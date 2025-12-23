#!/usr/bin/env python3
"""
Q vs K Scaling Control Experiment
=================================

Demonstrates that query norm, not generic logit magnitude, functions as 
the primary control signal for attention entropy.

Key Insight:
- If Q-scaling and K-scaling produced identical entropy curves, then
  entropy changes would just be a generic softmax effect.
- If Q-scaling produces stronger/different effects than K-scaling, then
  query magnitude specifically controls attention sharpness.

Experimental Invariants (MUST NOT CHANGE):
- Same pretrained model
- Same input sequence
- Same attention head(s)
- Same layer(s)
- Same entropy computation
- Same scaling values
- V (value) vectors are NEVER modified

Only one thing changes per experiment: what gets scaled (Q or K).

Usage:
    python notebooks/key-scaling/run_qk_comparison.py \\
        --target-layer 12 --target-head 0

All outputs saved to: notebooks/key-scaling/
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_loader import load_long_context, load_from_shards
from src.metrics import (
    compute_attention_entropy,
    compute_max_attention_weight,
    compute_effective_attention_span,
)


# Output directory - ALL results go here
OUTPUT_DIR = Path("notebooks/key-scaling")


@dataclass
class ScalingResult:
    """Result from a single scaling experiment."""
    alpha: float
    scaling_type: str  # "Q" or "K"
    mean_entropy: float
    std_entropy: float
    mean_max_attn: float
    mean_k_eff: float
    n_tokens: int


class QKScalingProfiler:
    """
    Profiles attention with Q-only or K-only scaling.
    
    This extends the intervention approach to support both Q and K scaling
    for direct comparison of their effects on attention entropy.
    """
    
    def __init__(self, model, target_layer: int, target_head: int):
        """
        Initialize the profiler.
        
        Args:
            model: HuggingFace causal LM model
            target_layer: Layer index to intervene on
            target_head: Head index to intervene on
        """
        self.model = model
        self.config = model.config
        self.target_layer = target_layer
        self.target_head = target_head
        
        # Architecture parameters
        self.n_layers = self.config.num_hidden_layers
        self.n_heads = self.config.num_attention_heads
        self.n_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.hidden_size // self.n_heads
        
        # Storage
        self._hidden_states: Dict[int, torch.Tensor] = {}
        self.hooks: List = []
    
    def _get_hidden_hook(self, layer_idx: int):
        """
        Create hook to capture pre-attention hidden states.
        
        We capture hidden states after input LayerNorm, matching the inputs
        used for QKV projection in standard LLaMA blocks. This is the correct
        intervention point for analyzing attention behavior.
        """
        def hook_fn(module, input, output):
            self._hidden_states[layer_idx] = output.detach()
        return hook_fn
    
    def _compute_qkv_with_scaling(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        q_scale: float = 1.0,
        k_scale: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Q, K, V with optional scaling.
        
        Args:
            layer_idx: Layer index
            hidden_states: Pre-attention hidden states
            q_scale: Scale factor for Q (target head only)
            k_scale: Scale factor for K (target head only)
            
        Returns:
            Q, K, V tensors
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
            
            # Reshape K, V (for GQA)
            K = K.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
            
            # === SCALING ===
            # Only scale at the target layer, target head
            # All other layers and heads remain UNCHANGED
            # V (value) vectors are NEVER modified in this experiment
            if layer_idx == self.target_layer:
                if q_scale != 1.0:
                    Q[:, self.target_head, :, :] = Q[:, self.target_head, :, :] * q_scale
                
                # For K scaling, need to handle GQA (multiple Q heads share one K head)
                if k_scale != 1.0:
                    # In GQA, n_heads Q heads share n_kv_heads K heads
                    # Each K head is shared by (n_heads // n_kv_heads) Q heads
                    kv_head_idx = self.target_head % self.n_kv_heads
                    K[:, kv_head_idx, :, :] = K[:, kv_head_idx, :, :] * k_scale
        
        # NOTE: V is explicitly unchanged in all experiments
        return Q, K, V
    
    def _compute_attention_probs(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        causal: bool = True
    ) -> torch.Tensor:
        """Compute attention probabilities from Q and K."""
        batch_size, n_heads, seq_len, head_dim = Q.shape
        _, n_kv_heads, _, _ = K.shape
        
        # Expand K for GQA
        n_rep = n_heads // n_kv_heads
        if n_rep > 1:
            K = K.repeat_interleave(n_rep, dim=1)
        
        # Attention scores
        scale = head_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Causal mask
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax
        attn_probs = torch.softmax(scores, dim=-1)
        
        return attn_probs
    
    def register_hooks(self):
        """Register forward hooks."""
        self.remove_hooks()
        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]
            hook = layer.input_layernorm.register_forward_hook(
                self._get_hidden_hook(layer_idx)
            )
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._hidden_states = {}
    
    def run_single_scaling(
        self,
        input_ids: torch.Tensor,
        alpha: float,
        scaling_type: str  # "Q" or "K"
    ) -> ScalingResult:
        """
        Run a single scaling experiment.
        
        Args:
            input_ids: Input token IDs
            alpha: Scale factor
            scaling_type: "Q" for query scaling, "K" for key scaling
            
        Returns:
            ScalingResult with metrics
        """
        self._hidden_states = {}
        self.register_hooks()
        
        try:
            # Forward pass to capture hidden states
            with torch.no_grad():
                _ = self.model(input_ids, use_cache=False)
            
            # Get hidden states for target layer
            hidden_states = self._hidden_states[self.target_layer]
            
            # Compute Q, K, V with appropriate scaling
            if scaling_type == "Q":
                Q, K, V = self._compute_qkv_with_scaling(
                    self.target_layer, hidden_states, q_scale=alpha, k_scale=1.0
                )
            else:  # K scaling
                Q, K, V = self._compute_qkv_with_scaling(
                    self.target_layer, hidden_states, q_scale=1.0, k_scale=alpha
                )
            
            # Compute attention probabilities
            attn_probs = self._compute_attention_probs(Q, K, causal=True)
            
            # Extract target head
            target_attn = attn_probs[:, self.target_head:self.target_head+1, :, :]
            
            # Compute metrics
            entropy = compute_attention_entropy(target_attn, ignore_first_n=2)
            max_attn = compute_max_attention_weight(target_attn)
            k_eff = compute_effective_attention_span(target_attn, threshold=0.9)
            
            # Convert to numpy
            entropy_np = entropy.squeeze().cpu().numpy()
            max_attn_np = max_attn.squeeze().cpu().numpy()
            k_eff_np = k_eff.squeeze().cpu().numpy()
            
            # Filter valid tokens
            valid_mask = ~np.isnan(entropy_np)
            
            return ScalingResult(
                alpha=alpha,
                scaling_type=scaling_type,
                mean_entropy=float(np.mean(entropy_np[valid_mask])),
                std_entropy=float(np.std(entropy_np[valid_mask])),
                mean_max_attn=float(np.mean(max_attn_np[valid_mask])),
                mean_k_eff=float(np.mean(k_eff_np[valid_mask])),
                n_tokens=int(valid_mask.sum())
            )
            
        finally:
            self.remove_hooks()
    
    def run_comparison_sweep(
        self,
        input_ids: torch.Tensor,
        alphas: List[float],
        verbose: bool = True
    ) -> Tuple[List[ScalingResult], List[ScalingResult]]:
        """
        Run both Q-scaling and K-scaling for all alphas.
        
        Args:
            input_ids: Input token IDs
            alphas: List of scale factors
            verbose: Print progress
            
        Returns:
            (q_results, k_results) tuple
        """
        q_results = []
        k_results = []
        
        if verbose:
            print(f"\n{'='*60}")
            print("Q vs K SCALING COMPARISON")
            print(f"{'='*60}")
            print(f"Target: Layer {self.target_layer}, Head {self.target_head}")
            print(f"Alphas: {alphas}")
            print(f"")
            print(f"⚠️  Scaling applied ONLY at Layer {self.target_layer}, Head {self.target_head}")
            print(f"⚠️  V (value) vectors are NEVER modified")
            print(f"{'='*60}")
        
        # Q-scaling sweep
        if verbose:
            print("\n--- Q-SCALING ---")
        for alpha in alphas:
            result = self.run_single_scaling(input_ids, alpha, "Q")
            q_results.append(result)
            if verbose:
                print(f"  α={alpha:.2f}: entropy={result.mean_entropy:.4f}, "
                      f"max_attn={result.mean_max_attn:.4f}")
            torch.cuda.empty_cache()
        
        # K-scaling sweep
        if verbose:
            print("\n--- K-SCALING ---")
        for alpha in alphas:
            result = self.run_single_scaling(input_ids, alpha, "K")
            k_results.append(result)
            if verbose:
                print(f"  α={alpha:.2f}: entropy={result.mean_entropy:.4f}, "
                      f"max_attn={result.mean_max_attn:.4f}")
            torch.cuda.empty_cache()
        
        if verbose:
            print(f"\n{'='*60}")
            print("SWEEP COMPLETE")
            print(f"{'='*60}")
        
        return q_results, k_results


def plot_qk_comparison(
    q_results: List[ScalingResult],
    k_results: List[ScalingResult],
    output_path: Path,
    metric: str = "entropy",
    layer: int = None,
    head: int = None
):
    """
    Create comparison plot of Q vs K scaling.
    
    Args:
        q_results: Q-scaling results
        k_results: K-scaling results
        output_path: Where to save the plot
        metric: "entropy", "max_attn", or "k_eff"
        layer: Layer index for title
        head: Head index for title
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Extract data
    alphas = [r.alpha for r in q_results]
    
    if metric == "entropy":
        q_values = [r.mean_entropy for r in q_results]
        k_values = [r.mean_entropy for r in k_results]
        ylabel = "Attention Entropy (H)"
        title = "Entropy Response to Q vs K Scaling"
    elif metric == "max_attn":
        q_values = [r.mean_max_attn for r in q_results]
        k_values = [r.mean_max_attn for r in k_results]
        ylabel = "Max Attention Weight (α_max)"
        title = "Max Attention Response to Q vs K Scaling"
    else:  # k_eff
        q_values = [r.mean_k_eff for r in q_results]
        k_values = [r.mean_k_eff for r in k_results]
        ylabel = "Effective Span (k_eff)"
        title = "Effective Span Response to Q vs K Scaling"
    
    # Plot Q-scaling (solid, prominent)
    ax.plot(alphas, q_values, 'o-', 
            color='#3498db', linewidth=2.5, markersize=10,
            markeredgecolor='white', markeredgewidth=2,
            label='Q-scaling (Query)')
    
    # Plot K-scaling (dashed, secondary)
    ax.plot(alphas, k_values, 's--', 
            color='#e74c3c', linewidth=2.5, markersize=10,
            markeredgecolor='white', markeredgewidth=2,
            label='K-scaling (Key)')
    
    # Vertical line at α=1.0 (baseline)
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Baseline (α=1.0)')
    
    # Labels
    ax.set_xlabel('Scaling Factor (α)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if layer is not None and head is not None:
        ax.set_title(f"{title}\nLayer {layer}, Head {head}", fontsize=14, fontweight='bold')
    else:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure same x-axis for fair comparison
    ax.set_xticks(alphas)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_delta_entropy(
    q_results: List[ScalingResult],
    k_results: List[ScalingResult],
    output_path: Path,
    layer: int = None,
    head: int = None
):
    """
    Plot delta-normalized entropy: ΔH(α) = H(α) - H(α=1.0)
    
    This removes head-specific entropy scale and emphasizes the response,
    making comparisons cleaner for publication.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Extract data
    alphas = [r.alpha for r in q_results]
    q_entropies = [r.mean_entropy for r in q_results]
    k_entropies = [r.mean_entropy for r in k_results]
    
    # Find baseline (α=1.0) values
    baseline_idx = alphas.index(1.0) if 1.0 in alphas else len(alphas) // 2
    q_baseline = q_entropies[baseline_idx]
    k_baseline = k_entropies[baseline_idx]
    
    # Compute delta: ΔH = H(α) - H(α=1.0)
    q_delta = [h - q_baseline for h in q_entropies]
    k_delta = [h - k_baseline for h in k_entropies]
    
    # Plot Q-scaling delta
    ax.plot(alphas, q_delta, 'o-', 
            color='#3498db', linewidth=2.5, markersize=10,
            markeredgecolor='white', markeredgewidth=2,
            label='Q-scaling (Query)')
    
    # Plot K-scaling delta
    ax.plot(alphas, k_delta, 's--', 
            color='#e74c3c', linewidth=2.5, markersize=10,
            markeredgecolor='white', markeredgewidth=2,
            label='K-scaling (Key)')
    
    # Reference lines
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Labels
    ax.set_xlabel('Scaling Factor (α)', fontsize=12)
    ax.set_ylabel('ΔH = H(α) - H(α=1.0)', fontsize=12)
    
    if layer is not None and head is not None:
        ax.set_title(f"Delta Entropy Response to Q vs K Scaling\nLayer {layer}, Head {head}", 
                     fontsize=14, fontweight='bold')
    else:
        ax.set_title("Delta Entropy Response to Q vs K Scaling", fontsize=14, fontweight='bold')
    
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(alphas)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")




def save_results_csv(
    q_results: List[ScalingResult],
    k_results: List[ScalingResult],
    output_path: Path,
    layer: int,
    head: int
):
    """Save results to CSV."""
    data = []
    
    for r in q_results:
        data.append({
            'alpha': r.alpha,
            'scaling_type': r.scaling_type,
            'mean_entropy': r.mean_entropy,
            'std_entropy': r.std_entropy,
            'mean_max_attn': r.mean_max_attn,
            'mean_k_eff': r.mean_k_eff,
            'n_tokens': r.n_tokens,
            'layer': layer,
            'head': head
        })
    
    for r in k_results:
        data.append({
            'alpha': r.alpha,
            'scaling_type': r.scaling_type,
            'mean_entropy': r.mean_entropy,
            'std_entropy': r.std_entropy,
            'mean_max_attn': r.mean_max_attn,
            'mean_k_eff': r.mean_k_eff,
            'n_tokens': r.n_tokens,
            'layer': layer,
            'head': head
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    
    return df


def compute_sensitivity_ratio(
    q_results: List[ScalingResult],
    k_results: List[ScalingResult]
) -> Dict:
    """
    Compute sensitivity metrics comparing Q vs K.
    
    Returns dict with sensitivity ratios.
    """
    # Get entropy values
    q_entropies = np.array([r.mean_entropy for r in q_results])
    k_entropies = np.array([r.mean_entropy for r in k_results])
    
    # Compute range (max - min)
    q_range = np.max(q_entropies) - np.min(q_entropies)
    k_range = np.max(k_entropies) - np.min(k_entropies)
    
    # Compute slopes via linear regression on log(alpha)
    alphas = np.array([r.alpha for r in q_results])
    log_alphas = np.log(alphas)
    
    from scipy import stats
    q_slope, _, q_r2, _, _ = stats.linregress(log_alphas, q_entropies)
    k_slope, _, k_r2, _, _ = stats.linregress(log_alphas, k_entropies)
    
    return {
        'q_entropy_range': q_range,
        'k_entropy_range': k_range,
        'range_ratio': q_range / k_range if k_range > 0 else float('inf'),
        'q_slope': q_slope,
        'k_slope': k_slope,
        'slope_ratio': abs(q_slope) / abs(k_slope) if abs(k_slope) > 0 else float('inf'),
        'q_r2': q_r2,
        'k_r2': k_r2
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Q vs K Scaling Control Experiment"
    )
    parser.add_argument("--target-layer", type=int, default=12,
                        help="Layer index (default: 12)")
    parser.add_argument("--target-head", type=int, default=0,
                        help="Head index (default: 0)")
    parser.add_argument("--alphas", type=str, default="0.25,0.5,0.75,1.0,1.25,1.5,2.0",
                        help="Comma-separated scaling factors")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Sample index from shards")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Data directory")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Model name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Q vs K SCALING CONTROL EXPERIMENT")
    print("=" * 70)
    print("\nPurpose: Demonstrate that query norm, not generic logit magnitude,")
    print("         functions as the primary control signal for attention entropy.")
    print("=" * 70)
    
    print(f"\n--- Configuration ---")
    print(f"  Target Layer: {args.target_layer}")
    print(f"  Target Head: {args.target_head}")
    print(f"  Alphas: {args.alphas}")
    print(f"  Output: {OUTPUT_DIR}")
    
    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    
    print(f"\n--- GPU ---")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print(f"\n--- Loading Model ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    print(f"  Model loaded: {args.model}")
    
    # Load sample
    print(f"\n--- Loading Sample ---")
    samples = load_from_shards(args.data_dir, n_samples=args.sample_idx + 1, device="cuda")
    if samples and len(samples) > args.sample_idx:
        input_ids = samples[args.sample_idx]["input_ids"]
        print(f"  Source: Preprocessed shards")
    else:
        sample = load_long_context(tokenizer, target_length=512)
        input_ids = sample["input_ids"]
        print(f"  Source: Generated sample")
    print(f"  Tokens: {input_ids.shape[1]}")
    
    # Parse alphas
    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    
    # Create profiler
    profiler = QKScalingProfiler(model, args.target_layer, args.target_head)
    
    # Run comparison sweep
    q_results, k_results = profiler.run_comparison_sweep(input_ids, alphas, verbose=True)
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    csv_path = OUTPUT_DIR / f"qk_comparison_L{args.target_layer}_H{args.target_head}_{timestamp}.csv"
    df = save_results_csv(q_results, k_results, csv_path, args.target_layer, args.target_head)
    
    # Also save canonical version
    canonical_csv = OUTPUT_DIR / "qk_comparison_results.csv"
    df.to_csv(canonical_csv, index=False)
    print(f"  Saved: {canonical_csv}")
    
    # Generate plots
    print("\n--- Generating Plots ---")
    
    # Entropy plot (primary)
    entropy_plot = OUTPUT_DIR / f"qk_entropy_comparison_L{args.target_layer}_H{args.target_head}.png"
    plot_qk_comparison(q_results, k_results, entropy_plot, "entropy", args.target_layer, args.target_head)
    
    # Max attention plot
    max_attn_plot = OUTPUT_DIR / f"qk_max_attn_comparison_L{args.target_layer}_H{args.target_head}.png"
    plot_qk_comparison(q_results, k_results, max_attn_plot, "max_attn", args.target_layer, args.target_head)
    
    # k_eff plot
    keff_plot = OUTPUT_DIR / f"qk_keff_comparison_L{args.target_layer}_H{args.target_head}.png"
    plot_qk_comparison(q_results, k_results, keff_plot, "k_eff", args.target_layer, args.target_head)
    
    # Delta entropy plot (recommended for publication - removes head-specific scale)
    delta_plot = OUTPUT_DIR / f"qk_delta_entropy_L{args.target_layer}_H{args.target_head}.png"
    plot_delta_entropy(q_results, k_results, delta_plot, args.target_layer, args.target_head)
    
    # Compute sensitivity metrics
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    sensitivity = compute_sensitivity_ratio(q_results, k_results)
    
    print(f"\n  Q entropy range: {sensitivity['q_entropy_range']:.4f}")
    print(f"  K entropy range: {sensitivity['k_entropy_range']:.4f}")
    print(f"  Range ratio (Q/K): {sensitivity['range_ratio']:.2f}x")
    
    print(f"\n  Q slope (d(entropy)/d(log α)): {sensitivity['q_slope']:.4f}")
    print(f"  K slope (d(entropy)/d(log α)): {sensitivity['k_slope']:.4f}")
    print(f"  Slope ratio (|Q|/|K|): {sensitivity['slope_ratio']:.2f}x")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if sensitivity['range_ratio'] > 1.2:
        print(f"\n✅ Q-scaling shows {sensitivity['range_ratio']:.1f}x stronger entropy response than K-scaling.")
        print("\n   → Query magnitude functions as a primary control signal for attention sharpness,")
        print("     rather than entropy being a generic consequence of logit magnitude.")
    elif sensitivity['range_ratio'] < 0.8:
        print(f"\n⚠️ K-scaling shows stronger response than Q-scaling.")
        print("   → Unexpected result. Key magnitude may be more influential.")
    else:
        print(f"\n📊 Q and K scaling show similar effects (ratio ≈ 1).")
        print("   → Both affect attention similarly. Generic softmax effect.")
    
    # Save sensitivity summary
    sensitivity_df = pd.DataFrame([{
        'layer': args.target_layer,
        'head': args.target_head,
        **sensitivity
    }])
    sensitivity_path = OUTPUT_DIR / "sensitivity_summary.csv"
    sensitivity_df.to_csv(sensitivity_path, index=False)
    print(f"\n  Saved: {sensitivity_path}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
