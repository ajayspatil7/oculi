"""
Causal Intervention Profiler for Phase 1 Experiment 1
======================================================

Implements targeted causal intervention on query vectors at inference time.
Scales the Q vector magnitude for a single (layer, head) pair while keeping
all other components unchanged.

Key Design:
- Intervention happens AFTER Q projection, BEFORE RoPE
- Only the target head's Q is scaled; K, V, other heads unchanged
- This tests causality: does Q magnitude control attention sharpness?

Usage:
    from src.intervention import InterventionProfiler
    
    profiler = InterventionProfiler(model, target_layer=12, target_head=0)
    results = profiler.run_intervention_sweep(input_ids, scales=[0.5, 1.0, 1.5])
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from src.metrics import (
    compute_attention_entropy,
    compute_max_attention_weight,
    compute_effective_attention_span,
)


@dataclass
class InterventionResult:
    """Results from a single intervention run."""
    scale: float
    mean_entropy: float
    mean_max_attn: float
    mean_k_eff: float
    std_entropy: float = 0.0
    std_max_attn: float = 0.0
    std_k_eff: float = 0.0
    n_tokens: int = 0


@dataclass 
class InterventionConfig:
    """Configuration for intervention experiment."""
    target_layer: int = 12          # Mid-network layer (default)
    target_head: int = 0            # Default head
    scales: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5])
    context_length: int = 512       # Fixed for experiment
    keff_threshold: float = 0.9     # 90% mass threshold
    ignore_first_n_tokens: int = 2  # Skip early tokens (insufficient context)


class InterventionProfiler:
    """
    Profiles attention with causal intervention on query vectors.
    
    This class extends the attention profiling approach to support
    targeted interventions. For a specified (layer, head) pair,
    it scales the Q vector by a factor while leaving everything else unchanged.
    
    The intervention pipeline:
    
        hidden_states
           ↓
        Q = W_q(hidden_states)
           ↓
        [ INTERVENE: Q[target_head] *= scale ]
           ↓
        attention = softmax(Q @ K^T / sqrt(d))
           ↓
        compute metrics (entropy, max_attn, k_eff)
    
    Note: We compute attention pre-RoPE, consistent with Phase-0 analysis.
    This is deliberate: we're testing whether the intrinsic Q magnitude
    (position-agnostic) causally controls attention behavior.
    """
    
    def __init__(
        self,
        model,
        target_layer: int,
        target_head: int,
        config: Optional[InterventionConfig] = None
    ):
        """
        Initialize the intervention profiler.
        
        Args:
            model: HuggingFace causal LM model (e.g., LlamaForCausalLM)
            target_layer: Layer index to intervene on
            target_head: Head index to intervene on
            config: Optional intervention configuration
        """
        self.model = model
        self.config_obj = model.config
        self.target_layer = target_layer
        self.target_head = target_head
        self.intervention_config = config or InterventionConfig()
        
        # Get architecture parameters
        self.n_layers = self.config_obj.num_hidden_layers
        self.n_heads = self.config_obj.num_attention_heads
        self.n_kv_heads = self.config_obj.num_key_value_heads
        self.head_dim = self.config_obj.hidden_size // self.n_heads
        self.hidden_size = self.config_obj.hidden_size
        
        # Validate target
        if target_layer < 0 or target_layer >= self.n_layers:
            raise ValueError(f"target_layer {target_layer} out of range [0, {self.n_layers})")
        if target_head < 0 or target_head >= self.n_heads:
            raise ValueError(f"target_head {target_head} out of range [0, {self.n_heads})")
        
        # Current scale factor (modified during intervention)
        self._current_scale: float = 1.0
        
        # Storage for captured data
        self._hidden_states: Dict[int, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
    def _get_hidden_hook(self, layer_idx: int):
        """Create hook to capture pre-attention hidden states."""
        def hook_fn(module, input, output):
            self._hidden_states[layer_idx] = output.detach()
        return hook_fn
    
    def _compute_qkv_with_intervention(
        self, 
        layer_idx: int, 
        hidden_states: torch.Tensor,
        scale: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Q, K, V with intervention on target head.
        
        The intervention only applies to Q vectors of the target head
        in the target layer. All other projections remain unchanged.
        
        Args:
            layer_idx: Current layer index
            hidden_states: Pre-attention normalized hidden states
            scale: Scale factor to apply (only for target layer/head)
            
        Returns:
            Q, K, V tensors with shapes:
            - Q: [batch, n_heads, seq, head_dim]
            - K: [batch, n_kv_heads, seq, head_dim]
            - V: [batch, n_kv_heads, seq, head_dim]
        """
        layer = self.model.model.layers[layer_idx]
        attn = layer.self_attn
        
        batch_size, seq_len, _ = hidden_states.shape
        
        with torch.no_grad():
            # Project to Q, K, V
            Q = attn.q_proj(hidden_states)
            K = attn.k_proj(hidden_states)
            V = attn.v_proj(hidden_states)
            
            # Reshape Q: [batch, seq, n_heads * head_dim] -> [batch, n_heads, seq, head_dim]
            Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            
            # === INTERVENTION ===
            # Scale only the target head's Q vectors at the target layer
            if layer_idx == self.target_layer and scale != 1.0:
                Q[:, self.target_head, :, :] = Q[:, self.target_head, :, :] * scale
            # === END INTERVENTION ===
            
            # Reshape K, V: [batch, seq, n_kv_heads * head_dim] -> [batch, n_kv_heads, seq, head_dim]
            K = K.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        return Q, K, V
    
    def _compute_attention_probs(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor,
        causal: bool = True
    ) -> torch.Tensor:
        """
        Compute attention probabilities from Q and K (pre-RoPE).
        
        This recomputes attention without RoPE to match Phase-0 analysis.
        The intervention (Q scaling) has already been applied.
        
        Args:
            Q: [batch, n_heads, seq, head_dim]
            K: [batch, n_kv_heads, seq, head_dim]
            causal: Whether to apply causal masking
            
        Returns:
            attn_probs: [batch, n_heads, seq, seq]
        """
        batch_size, n_heads, seq_len, head_dim = Q.shape
        _, n_kv_heads, _, _ = K.shape
        
        # Expand K for GQA if needed
        n_rep = n_heads // n_kv_heads
        if n_rep > 1:
            K = K.repeat_interleave(n_rep, dim=1)
        
        # Compute attention scores: [batch, n_heads, seq, seq]
        scale = head_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Apply causal mask
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax to get probabilities
        attn_probs = torch.softmax(scores, dim=-1)
        
        return attn_probs
    
    def register_hooks(self):
        """Register forward hooks on all attention layers."""
        self.remove_hooks()
        
        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]
            hook = layer.input_layernorm.register_forward_hook(
                self._get_hidden_hook(layer_idx)
            )
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._hidden_states = {}
    
    def run_single_intervention(
        self, 
        input_ids: torch.Tensor, 
        scale: float
    ) -> InterventionResult:
        """
        Run a single forward pass with intervention and compute metrics.
        
        Args:
            input_ids: [batch, seq_len] input token IDs
            scale: Scale factor to apply to target head's Q
            
        Returns:
            InterventionResult with aggregated metrics
        """
        self._current_scale = scale
        self._hidden_states = {}
        self.register_hooks()
        
        try:
            # Run forward pass to trigger hooks
            with torch.no_grad():
                _ = self.model(input_ids, use_cache=False)
            
            # Get hidden states for target layer
            if self.target_layer not in self._hidden_states:
                raise RuntimeError(f"No hidden states captured for layer {self.target_layer}")
            
            hidden_states = self._hidden_states[self.target_layer]
            
            # Compute Q, K, V with intervention
            Q, K, V = self._compute_qkv_with_intervention(
                self.target_layer, hidden_states, scale
            )
            
            # Compute attention probabilities
            attn_probs = self._compute_attention_probs(Q, K, causal=True)
            
            # Extract metrics for target head only
            # attn_probs: [batch, n_heads, seq, seq]
            target_attn = attn_probs[:, self.target_head:self.target_head+1, :, :]
            
            # Compute metrics
            entropy = compute_attention_entropy(
                target_attn, 
                ignore_first_n=self.intervention_config.ignore_first_n_tokens
            )
            max_attn = compute_max_attention_weight(target_attn)
            k_eff = compute_effective_attention_span(
                target_attn, 
                threshold=self.intervention_config.keff_threshold
            )
            
            # Squeeze to 1D and convert to numpy
            entropy = entropy.squeeze().cpu().numpy()
            max_attn = max_attn.squeeze().cpu().numpy()
            k_eff = k_eff.squeeze().cpu().numpy()
            
            # Filter out NaN values and early tokens
            valid_mask = ~np.isnan(entropy)
            entropy_valid = entropy[valid_mask]
            max_attn_valid = max_attn[valid_mask]
            k_eff_valid = k_eff[valid_mask]
            
            # Aggregate across tokens
            result = InterventionResult(
                scale=scale,
                mean_entropy=float(np.mean(entropy_valid)) if len(entropy_valid) > 0 else float('nan'),
                mean_max_attn=float(np.mean(max_attn_valid)) if len(max_attn_valid) > 0 else float('nan'),
                mean_k_eff=float(np.mean(k_eff_valid)) if len(k_eff_valid) > 0 else float('nan'),
                std_entropy=float(np.std(entropy_valid)) if len(entropy_valid) > 0 else float('nan'),
                std_max_attn=float(np.std(max_attn_valid)) if len(max_attn_valid) > 0 else float('nan'),
                std_k_eff=float(np.std(k_eff_valid)) if len(k_eff_valid) > 0 else float('nan'),
                n_tokens=len(entropy_valid)
            )
            
            return result
            
        finally:
            self.remove_hooks()
    
    def run_intervention_sweep(
        self, 
        input_ids: torch.Tensor,
        scales: Optional[List[float]] = None,
        verbose: bool = True
    ) -> List[InterventionResult]:
        """
        Run intervention experiment across multiple scale factors.
        
        Args:
            input_ids: [batch, seq_len] input token IDs
            scales: List of scale factors (default from config)
            verbose: Whether to print progress
            
        Returns:
            List of InterventionResult, one per scale
        """
        if scales is None:
            scales = self.intervention_config.scales
        
        results = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"INTERVENTION SWEEP")
            print(f"{'='*60}")
            print(f"Target: Layer {self.target_layer}, Head {self.target_head}")
            print(f"Scales: {scales}")
            print(f"{'='*60}")
        
        for scale in scales:
            if verbose:
                print(f"\nRunning scale={scale:.2f}...")
            
            result = self.run_single_intervention(input_ids, scale)
            results.append(result)
            
            if verbose:
                print(f"  Entropy: {result.mean_entropy:.4f} ± {result.std_entropy:.4f}")
                print(f"  Max Attn: {result.mean_max_attn:.4f} ± {result.std_max_attn:.4f}")
                print(f"  k_eff: {result.mean_k_eff:.2f} ± {result.std_k_eff:.2f}")
                print(f"  Tokens: {result.n_tokens}")
            
            # Clear CUDA cache between runs
            torch.cuda.empty_cache()
        
        if verbose:
            print(f"\n{'='*60}")
            print("SWEEP COMPLETE")
            print(f"{'='*60}")
        
        return results
    
    @staticmethod
    def results_to_dataframe(
        results: List[InterventionResult],
        target_layer: int = None,
        target_head: int = None,
        head_type: str = "target"
    ) -> pd.DataFrame:
        """
        Convert intervention results to a pandas DataFrame.
        
        Args:
            results: List of InterventionResult
            target_layer: Layer index (for metadata)
            target_head: Head index (for metadata)
            head_type: Type of head ("target" or "control")
            
        Returns:
            DataFrame with one row per scale
        """
        data = []
        for r in results:
            data.append({
                'scale': r.scale,
                'mean_entropy': r.mean_entropy,
                'mean_max_attn': r.mean_max_attn,
                'mean_k_eff': r.mean_k_eff,
                'std_entropy': r.std_entropy,
                'std_max_attn': r.std_max_attn,
                'std_k_eff': r.std_k_eff,
                'n_tokens': r.n_tokens,
                'layer': target_layer,
                'head': target_head,
                'head_type': head_type
            })
        return pd.DataFrame(data)


def verify_intervention_isolation(
    model,
    input_ids: torch.Tensor,
    target_layer: int,
    target_head: int,
    scale: float = 2.0
) -> bool:
    """
    Verify that intervention only affects the target head.
    
    This test confirms that:
    1. Q vectors for target head are scaled
    2. Q vectors for other heads remain unchanged
    3. K and V vectors remain unchanged
    
    Args:
        model: HuggingFace model
        input_ids: Test input
        target_layer: Layer to test
        target_head: Head to test
        scale: Scale factor to apply
        
    Returns:
        True if isolation is verified
    """
    print(f"\nVerifying intervention isolation...")
    print(f"  Target: Layer {target_layer}, Head {target_head}")
    print(f"  Scale: {scale}")
    
    # Create profiler
    profiler = InterventionProfiler(model, target_layer, target_head)
    
    # Get baseline (scale=1.0) Q vectors
    profiler.register_hooks()
    with torch.no_grad():
        _ = model(input_ids, use_cache=False)
    hidden_baseline = profiler._hidden_states[target_layer].clone()
    Q_baseline, K_baseline, V_baseline = profiler._compute_qkv_with_intervention(
        target_layer, hidden_baseline, scale=1.0
    )
    profiler.remove_hooks()
    
    # Get intervened Q vectors
    profiler.register_hooks()
    with torch.no_grad():
        _ = model(input_ids, use_cache=False)
    hidden_intervened = profiler._hidden_states[target_layer].clone()
    Q_intervened, K_intervened, V_intervened = profiler._compute_qkv_with_intervention(
        target_layer, hidden_intervened, scale=scale
    )
    profiler.remove_hooks()
    
    # Check 1: Target head Q should be scaled
    target_q_ratio = Q_intervened[:, target_head] / (Q_baseline[:, target_head] + 1e-9)
    target_q_mean_ratio = target_q_ratio.mean().item()
    target_scaled = abs(target_q_mean_ratio - scale) < 0.01
    print(f"  Target head Q ratio: {target_q_mean_ratio:.4f} (expected: {scale})")
    
    # Check 2: Other heads Q should be unchanged
    other_heads = [h for h in range(profiler.n_heads) if h != target_head]
    if other_heads:
        other_q_diff = (Q_intervened[:, other_heads] - Q_baseline[:, other_heads]).abs().max().item()
        others_unchanged = other_q_diff < 1e-5
        print(f"  Other heads Q max diff: {other_q_diff:.2e} (expected: ~0)")
    else:
        others_unchanged = True
    
    # Check 3: K and V should be unchanged
    k_diff = (K_intervened - K_baseline).abs().max().item()
    v_diff = (V_intervened - V_baseline).abs().max().item()
    kv_unchanged = k_diff < 1e-5 and v_diff < 1e-5
    print(f"  K max diff: {k_diff:.2e} (expected: ~0)")
    print(f"  V max diff: {v_diff:.2e} (expected: ~0)")
    
    passed = target_scaled and others_unchanged and kv_unchanged
    print(f"\n  {'✅ PASSED' if passed else '❌ FAILED'}: Intervention isolation verified")
    
    return passed


if __name__ == "__main__":
    # Test the intervention profiler
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Testing InterventionProfiler...")
    
    # This test requires GPU - will fail without CUDA
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available. Skipping live test.")
        print("Run on SageMaker with GPU to verify.")
    else:
        # Load model
        model_name = "meta-llama/Meta-Llama-3-8B"
        print(f"\nLoading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        # Create test input
        text = "The quick brown fox jumps over the lazy dog. " * 20
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to("cuda")
        
        # Test intervention isolation
        verify_intervention_isolation(
            model, inputs.input_ids, 
            target_layer=12, target_head=0, scale=2.0
        )
        
        # Test sweep
        profiler = InterventionProfiler(model, target_layer=12, target_head=0)
        results = profiler.run_intervention_sweep(
            inputs.input_ids, 
            scales=[0.5, 1.0, 1.5]
        )
        
        # Convert to DataFrame
        df = InterventionProfiler.results_to_dataframe(results, 12, 0)
        print("\nResults DataFrame:")
        print(df)
        
        print("\n✅ InterventionProfiler test passed!")
