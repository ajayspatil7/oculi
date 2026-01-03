"""
LLaMA Attention Adapter
=======================

Public adapter for LLaMA model family (LLaMA 2, LLaMA 3).

This adapter provides:
    - Explicit model loading (no magic auto-detection)
    - Attention capture with GQA handling
    - Hook management for interventions
    - Generation utilities

Supported Models:
    - meta-llama/Meta-Llama-3-8B (GQA: 32 Q heads, 8 KV heads)
    - meta-llama/Meta-Llama-3-70B
    - meta-llama/Llama-2-7b-hf (MHA: 32 Q heads, 32 KV heads)
    - meta-llama/Llama-2-13b-hf
    - meta-llama/Llama-2-70b-hf

Usage:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from oculi.models.llama import LlamaAttentionAdapter
    
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    adapter = LlamaAttentionAdapter(model, tokenizer)
    capture = adapter.capture(input_ids)
"""

from typing import Optional, Dict, List, Callable, Union
import torch

from oculi.models.base import AttentionAdapter, CaptureError
from oculi.capture.structures import (
    AttentionCapture,
    AttentionStructure,
    CaptureConfig,
    ResidualConfig,
    ResidualCapture,
    MLPConfig,
    MLPCapture,
    LogitConfig,
    LogitCapture,
    FullCapture,
)

# Import attention anatomy (the "where things are" documentation)
from oculi.models.llama.attention import (
    get_q_proj,
    get_k_proj,
    get_v_proj,
    get_o_proj,
    get_attention_module,
    expand_kv_for_gqa,
    create_capture_hook,
)


class LlamaAttentionAdapter(AttentionAdapter):
    """
    Attention adapter for LLaMA model family.
    
    This class wraps a pre-loaded LLaMA model and provides the Oculi
    capture/intervention interface. You must load the model yourself
    (no magic, no hidden state).
    
    Architecture (LLaMA-3-8B):
        - 32 layers
        - 32 query heads, 8 KV heads (GQA 4:1)
        - 128 head dimension
        - 4096 hidden size
        
    Args:
        model: Pre-loaded LlamaForCausalLM
        tokenizer: Pre-loaded tokenizer
        device: Override device (default: model's device)
        
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        >>> adapter = LlamaAttentionAdapter(model, tokenizer)
        >>> print(adapter.attention_structure())
        AttentionStructure(n_query_heads=32, n_kv_heads=8, head_dim=128)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[str] = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        
        # Determine device from model
        if device is not None:
            self._device = torch.device(device)
        else:
            self._device = next(model.parameters()).device
        
        # Cache architecture info from model config
        config = model.config
        self._model_name = getattr(config, '_name_or_path', 'llama')
        self._n_layers = config.num_hidden_layers
        self._n_heads = config.num_attention_heads
        self._n_kv_heads = getattr(config, 'num_key_value_heads', self._n_heads)
        self._head_dim = config.hidden_size // self._n_heads
        
        # Hook management
        self._hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._hook_counter = 0
        
        # Ensure model is in eval mode
        self._model.eval()
    
    # =========================================================================
    # ARCHITECTURE INTROSPECTION
    # =========================================================================
    
    def num_layers(self) -> int:
        return self._n_layers
    
    def num_heads(self, layer: int = 0) -> int:
        return self._n_heads
    
    def num_kv_heads(self, layer: int = 0) -> int:
        return self._n_kv_heads
    
    def head_dim(self, layer: int = 0) -> int:
        return self._head_dim
    
    def attention_structure(self, layer: int = 0) -> AttentionStructure:
        return AttentionStructure(
            n_query_heads=self._n_heads,
            n_kv_heads=self._n_kv_heads,
            head_dim=self._head_dim
        )
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    # =========================================================================
    # DIRECT MODEL ACCESS (for learning/debugging)
    # =========================================================================
    
    @property
    def model(self):
        """Direct access to underlying model (for learning/debugging)."""
        return self._model
    
    @property
    def tokenizer(self):
        """Direct access to tokenizer."""
        return self._tokenizer
    
    def get_attention_module(self, layer: int):
        """
        Get the attention module for a layer.
        
        Returns: LlamaAttention (or LlamaSdpaAttention/LlamaFlashAttention2)
        
        Useful for understanding the exact implementation:
            >>> attn = adapter.get_attention_module(0)
            >>> print(type(attn))  # See which implementation is used
            >>> print(attn.q_proj.weight.shape)  # Inspect weights
        """
        return get_attention_module(self._model, layer)
    
    def get_q_proj(self, layer: int):
        """Get Q projection module. See models/llama/attention.py for details."""
        return get_q_proj(self._model, layer)
    
    def get_k_proj(self, layer: int):
        """Get K projection module. See models/llama/attention.py for details."""
        return get_k_proj(self._model, layer)
    
    def get_v_proj(self, layer: int):
        """Get V projection module. See models/llama/attention.py for details."""
        return get_v_proj(self._model, layer)
    
    # =========================================================================
    # CAPTURE API
    # =========================================================================
    
    def capture(
        self,
        input_ids: torch.Tensor,
        config: Optional[CaptureConfig] = None
    ) -> AttentionCapture:
        """
        Capture attention data from a forward pass.
        
        This registers temporary hooks, runs the forward pass, and 
        assembles the captured data into an AttentionCapture.
        
        Note: For models using SDPA/FlashAttention, attention patterns
        are computed manually from Q and K vectors since those backends
        don't return attention weights.
        
        Args:
            input_ids: [1, seq_len] token IDs
            config: What to capture (default: all components, all layers)
            
        Returns:
            AttentionCapture with Q, K, V, and/or attention patterns
        """
        if config is None:
            config = CaptureConfig()
        
        config.validate(self._n_layers)
        
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must be shape [1, seq_len], got {input_ids.shape}"
            )
        
        n_tokens = input_ids.shape[1]
        layers_to_capture = config.get_layers(self._n_layers)
        
        # Storage for captured tensors
        captured_q = {}
        captured_k = {}
        captured_v = {}
        captured_patterns = {}
        
        # For pattern capture, we ALWAYS need Q and K (will compute patterns manually)
        need_q_for_patterns = config.capture_patterns
        need_k_for_patterns = config.capture_patterns
        
        # Register hooks
        handles = []
        
        for layer_idx in layers_to_capture:
            if config.capture_queries or need_q_for_patterns:
                hook = create_capture_hook(
                    captured_q, layer_idx, 'q', self._n_heads, self._head_dim
                )
                h = get_q_proj(self._model, layer_idx).register_forward_hook(hook)
                handles.append(h)
            
            if config.capture_keys or need_k_for_patterns:
                hook = create_capture_hook(
                    captured_k, layer_idx, 'k', self._n_kv_heads, self._head_dim
                )
                h = get_k_proj(self._model, layer_idx).register_forward_hook(hook)
                handles.append(h)
            
            if config.capture_values:
                hook = create_capture_hook(
                    captured_v, layer_idx, 'v', self._n_kv_heads, self._head_dim
                )
                h = get_v_proj(self._model, layer_idx).register_forward_hook(hook)
                handles.append(h)
        
        try:
            input_ids = input_ids.to(self._device)
            with torch.no_grad():
                outputs = self._model(
                    input_ids,
                    output_attentions=config.capture_patterns,  # Try to get native patterns
                    return_dict=True
                )
            
            # Check if we got attention patterns from the model
            native_patterns = outputs.attentions is not None
            
            if config.capture_patterns:
                if native_patterns:
                    # Model returned patterns (eager attention mode)
                    for layer_idx in layers_to_capture:
                        captured_patterns[layer_idx] = outputs.attentions[layer_idx].cpu()
                else:
                    # SDPA/FlashAttention - compute patterns manually from Q and K
                    captured_patterns = self._compute_attention_patterns(
                        captured_q, captured_k, layers_to_capture, n_tokens
                    )
        
        finally:
            for h in handles:
                h.remove()
        
        # Assemble tensors
        queries = self._assemble_tensor(
            captured_q, layers_to_capture, self._n_heads, n_tokens
        ) if config.capture_queries else None
        
        keys = self._assemble_tensor(
            captured_k, layers_to_capture, self._n_kv_heads, n_tokens
        ) if config.capture_keys else None
        
        values = self._assemble_tensor(
            captured_v, layers_to_capture, self._n_kv_heads, n_tokens
        ) if config.capture_values else None
        
        patterns = self._assemble_patterns(
            captured_patterns, layers_to_capture
        ) if config.capture_patterns else None
        
        return AttentionCapture(
            queries=queries,
            keys=keys,
            values=values,
            patterns=patterns,
            n_layers=len(layers_to_capture),
            n_heads=self._n_heads,
            n_kv_heads=self._n_kv_heads,
            n_tokens=n_tokens,
            head_dim=self._head_dim,
            model_name=self._model_name,
            qk_stage=config.qk_stage,
            captured_layers=tuple(layers_to_capture),
        )
    
    def _compute_attention_patterns(
        self,
        captured_q: dict,
        captured_k: dict,
        layers: List[int],
        n_tokens: int
    ) -> dict:
        """
        Compute attention patterns manually from Q and K.
        
        This is needed for SDPA/FlashAttention which don't return attention weights.
        
        Formula:
            attn = softmax(Q @ K^T / sqrt(head_dim)) with causal mask
        """
        patterns = {}
        scale = self._head_dim ** -0.5
        
        for layer_idx in layers:
            if layer_idx not in captured_q or layer_idx not in captured_k:
                continue
            
            # captured_q[layer_idx]: [batch, seq, n_heads, head_dim]
            q = captured_q[layer_idx].float()  # [1, T, H_q, D]
            k = captured_k[layer_idx].float()  # [1, T, H_kv, D]
            
            # Expand K for GQA: [1, T, H_kv, D] -> [1, T, H_q, D]
            if self._n_kv_heads != self._n_heads:
                k = expand_kv_for_gqa(k, self._n_heads, self._n_kv_heads)
            
            # Transpose to [1, H, T, D]
            q = q.permute(0, 2, 1, 3)  # [1, H, T, D]
            k = k.permute(0, 2, 1, 3)  # [1, H, T, D]
            
            # Compute attention scores: [1, H, T, T]
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Apply causal mask
            causal_mask = torch.triu(
                torch.ones(n_tokens, n_tokens, device=scores.device, dtype=torch.bool),
                diagonal=1
            )
            scores.masked_fill_(causal_mask, float('-inf'))
            
            # Softmax
            attn_probs = torch.softmax(scores, dim=-1)
            
            # Store [1, H, T, T]
            patterns[layer_idx] = attn_probs.cpu()
        
        return patterns
    
    def _assemble_tensor(
        self,
        storage: dict,
        layers: List[int],
        n_heads: int,
        n_tokens: int
    ) -> torch.Tensor:
        """Assemble captured data into [L, H, T, D] tensor."""
        result = torch.zeros(len(layers), n_heads, n_tokens, self._head_dim)
        for i, layer_idx in enumerate(layers):
            if layer_idx in storage:
                # storage is [batch, seq, heads, dim] -> [heads, seq, dim]
                result[i] = storage[layer_idx][0].permute(1, 0, 2)
        return result
    
    def _assemble_patterns(
        self,
        storage: dict,
        layers: List[int]
    ) -> Optional[torch.Tensor]:
        """Assemble attention patterns into [L, H, T, T] tensor."""
        if not storage:
            return None
        
        first = next(iter(storage.values()))
        _, n_heads, seq, _ = first.shape
        
        result = torch.zeros(len(layers), n_heads, seq, seq)
        for i, layer_idx in enumerate(layers):
            if layer_idx in storage:
                result[i] = storage[layer_idx][0]
        return result
    
    # =========================================================================
    # RESIDUAL STREAM CAPTURE API
    # =========================================================================
    
    def capture_residual(
        self,
        input_ids: torch.Tensor,
        config: Optional[ResidualConfig] = None
    ) -> ResidualCapture:
        """
        Capture residual stream activations during forward pass.
        
        Captures hidden states at four key points in each transformer block:
        - pre_attn: Block input (before input_layernorm)
        - post_attn: After attention (before residual add)
        - pre_mlp: After post_attention_layernorm (before MLP)
        - post_mlp: After MLP (before final residual add)
        
        Args:
            input_ids: [1, seq_len] token IDs
            config: What to capture (default: all positions, all layers)
            
        Returns:
            ResidualCapture with tensors of shape [L, T, H]
            
        Example:
            >>> residual = adapter.capture_residual(input_ids)
            >>> print(residual.pre_attn.shape)  # [32, 512, 4096]
            >>> layer_10 = residual.get_layer(10)
        """
        from oculi._private.hooks.residual import (
            register_residual_hooks,
            assemble_residual_capture,
        )
        
        if config is None:
            config = ResidualConfig()
        
        config.validate(self._n_layers)
        
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must be shape [1, seq_len], got {input_ids.shape}"
            )
        
        n_tokens = input_ids.shape[1]
        layers_to_capture = config.get_layers(self._n_layers)
        hidden_dim = self._model.config.hidden_size
        
        # Storage for captured tensors
        storage = {
            'pre_attn': {},
            'post_attn': {},
            'pre_mlp': {},
            'post_mlp': {},
        }
        
        # Register hooks
        handles = register_residual_hooks(
            self._model,
            layers_to_capture,
            storage,
            capture_pre_attn=config.capture_pre_attn,
            capture_post_attn=config.capture_post_attn,
            capture_pre_mlp=config.capture_pre_mlp,
            capture_post_mlp=config.capture_post_mlp,
            dtype=config.storage_dtype,
        )
        
        try:
            input_ids = input_ids.to(self._device)
            with torch.no_grad():
                self._model(input_ids, return_dict=True)
        finally:
            for h in handles:
                h.remove()
        
        return assemble_residual_capture(
            storage,
            layers_to_capture,
            n_tokens,
            hidden_dim,
            self._model_name,
        )
    
    # =========================================================================
    # MLP CAPTURE API
    # =========================================================================
    
    def capture_mlp(
        self,
        input_ids: torch.Tensor,
        config: Optional[MLPConfig] = None
    ) -> MLPCapture:
        """
        Capture MLP internals during forward pass.
        
        Captures:
        - pre_activation: Gate projection output (before SiLU)
        - post_activation: silu(gate) * up ("neuron activations")
        - gate_output: Gate projection output (optional)
        - mlp_output: Final MLP output
        
        Args:
            input_ids: [1, seq_len] token IDs
            config: What to capture (default: all except gate)
            
        Returns:
            MLPCapture with tensors of shape [L, T, I] or [L, T, H]
        """
        from oculi._private.hooks.mlp import (
            register_mlp_hooks,
            assemble_mlp_capture,
        )
        
        if config is None:
            config = MLPConfig()
        
        config.validate(self._n_layers)
        
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must be shape [1, seq_len], got {input_ids.shape}"
            )
        
        n_tokens = input_ids.shape[1]
        layers_to_capture = config.get_layers(self._n_layers)
        hidden_dim = self._model.config.hidden_size
        intermediate_dim = self._model.config.intermediate_size
        
        storage = {
            'pre_activation': {},
            'post_activation': {},
            'gate': {},
            'output': {},
        }
        
        handles = register_mlp_hooks(
            self._model,
            layers_to_capture,
            storage,
            capture_pre_activation=config.capture_pre_activation,
            capture_post_activation=config.capture_post_activation,
            capture_gate=config.capture_gate,
            capture_output=config.capture_output,
            dtype=config.storage_dtype,
        )
        
        try:
            input_ids = input_ids.to(self._device)
            with torch.no_grad():
                self._model(input_ids, return_dict=True)
        finally:
            for h in handles:
                h.remove()
        
        return assemble_mlp_capture(
            storage,
            layers_to_capture,
            n_tokens,
            hidden_dim,
            intermediate_dim,
            self._model_name,
        )
    
    # =========================================================================
    # LOGIT CAPTURE API (Logit Lens)
    # =========================================================================
    
    def capture_logits(
        self,
        input_ids: torch.Tensor,
        config: Optional[LogitConfig] = None
    ) -> LogitCapture:
        """
        Capture layer-wise logits for logit lens analysis.
        
        Applies the unembedding matrix to each layer's residual stream:
        logits[l] = layer_norm(residual[l]) @ lm_head.weight.T
        
        Args:
            input_ids: [1, seq_len] token IDs
            config: Configuration (top_k for memory efficiency)
            
        Returns:
            LogitCapture with logits or top-k logits
            
        Warning:
            Full logits for large models can be very memory-intensive.
            Use config.top_k for practical analysis.
        """
        if config is None:
            config = LogitConfig()
        
        config.validate(self._n_layers)
        
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must be shape [1, seq_len], got {input_ids.shape}"
            )
        
        # First capture residual post_mlp (the residual stream at each layer)
        residual_config = ResidualConfig(
            capture_pre_attn=False,
            capture_post_attn=False,
            capture_pre_mlp=False,
            capture_post_mlp=True,
            layers=config.layers,
            storage_dtype=config.storage_dtype,
        )
        
        residual = self.capture_residual(input_ids, residual_config)
        
        n_tokens = input_ids.shape[1]
        layers_to_capture = config.get_layers(self._n_layers)
        vocab_size = self._model.config.vocab_size
        
        # Get unembedding matrix and final layer norm
        lm_head = self._model.lm_head
        final_norm = self._model.model.norm
        
        # Compute logits at each layer
        with torch.no_grad():
            # residual.post_mlp: [L, T, H]
            post_mlp = residual.post_mlp.to(self._device)  # Move to device
            
            # Apply layer norm and compute logits
            # Note: We apply the final layer norm to approximate what the
            # model would predict at that layer
            normalized = final_norm(post_mlp)  # [L, T, H]
            all_logits = normalized @ lm_head.weight.T  # [L, T, V]
            all_logits = all_logits.to(config.storage_dtype).cpu()
        
        if config.top_k is not None:
            # Only store top-k
            top_k_logits, top_k_indices = torch.topk(all_logits, config.top_k, dim=-1)
            return LogitCapture(
                logits=None,
                top_k_logits=top_k_logits,
                top_k_indices=top_k_indices,
                n_layers=len(layers_to_capture),
                n_tokens=n_tokens,
                vocab_size=vocab_size,
                model_name=self._model_name,
                captured_layers=tuple(layers_to_capture),
            )
        else:
            return LogitCapture(
                logits=all_logits,
                top_k_logits=None,
                top_k_indices=None,
                n_layers=len(layers_to_capture),
                n_tokens=n_tokens,
                vocab_size=vocab_size,
                model_name=self._model_name,
                captured_layers=tuple(layers_to_capture),
            )
    
    # =========================================================================
    # FULL CAPTURE API
    # =========================================================================
    
    def capture_full(
        self,
        input_ids: torch.Tensor,
        attention_config: Optional[CaptureConfig] = None,
        residual_config: Optional[ResidualConfig] = None,
        mlp_config: Optional[MLPConfig] = None
    ) -> FullCapture:
        """
        Capture attention, residual, and MLP in a single forward pass.
        
        More efficient than calling capture(), capture_residual(), and
        capture_mlp() separately since it only runs one forward pass.
        
        Args:
            input_ids: [1, seq_len] token IDs
            attention_config: Config for attention capture (None = skip)
            residual_config: Config for residual capture (None = skip)
            mlp_config: Config for MLP capture (None = skip)
            
        Returns:
            FullCapture with all requested captures
        """
        from oculi._private.hooks.residual import (
            register_residual_hooks,
            assemble_residual_capture,
        )
        from oculi._private.hooks.mlp import (
            register_mlp_hooks,
            assemble_mlp_capture,
        )
        
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must be shape [1, seq_len], got {input_ids.shape}"
            )
        
        n_tokens = input_ids.shape[1]
        hidden_dim = self._model.config.hidden_size
        intermediate_dim = self._model.config.intermediate_size
        
        all_handles = []
        
        # Attention storage and hooks
        captured_q = {}
        captured_k = {}
        captured_v = {}
        attn_layers = []
        
        if attention_config is not None:
            attention_config.validate(self._n_layers)
            attn_layers = attention_config.get_layers(self._n_layers)
            
            for layer_idx in attn_layers:
                if attention_config.capture_queries:
                    from oculi.models.llama.attention import create_capture_hook
                    hook = create_capture_hook(
                        captured_q, layer_idx, 'q', self._n_heads, self._head_dim
                    )
                    h = get_q_proj(self._model, layer_idx).register_forward_hook(hook)
                    all_handles.append(h)
                
                if attention_config.capture_keys:
                    from oculi.models.llama.attention import create_capture_hook
                    hook = create_capture_hook(
                        captured_k, layer_idx, 'k', self._n_kv_heads, self._head_dim
                    )
                    h = get_k_proj(self._model, layer_idx).register_forward_hook(hook)
                    all_handles.append(h)
                
                if attention_config.capture_values:
                    from oculi.models.llama.attention import create_capture_hook
                    hook = create_capture_hook(
                        captured_v, layer_idx, 'v', self._n_kv_heads, self._head_dim
                    )
                    h = get_v_proj(self._model, layer_idx).register_forward_hook(hook)
                    all_handles.append(h)
        
        # Residual storage and hooks
        residual_storage = {
            'pre_attn': {},
            'post_attn': {},
            'pre_mlp': {},
            'post_mlp': {},
        }
        residual_layers = []
        
        if residual_config is not None:
            residual_config.validate(self._n_layers)
            residual_layers = residual_config.get_layers(self._n_layers)
            
            handles = register_residual_hooks(
                self._model,
                residual_layers,
                residual_storage,
                capture_pre_attn=residual_config.capture_pre_attn,
                capture_post_attn=residual_config.capture_post_attn,
                capture_pre_mlp=residual_config.capture_pre_mlp,
                capture_post_mlp=residual_config.capture_post_mlp,
                dtype=residual_config.storage_dtype,
            )
            all_handles.extend(handles)
        
        # MLP storage and hooks
        mlp_storage = {
            'pre_activation': {},
            'post_activation': {},
            'gate': {},
            'output': {},
        }
        mlp_layers = []
        
        if mlp_config is not None:
            mlp_config.validate(self._n_layers)
            mlp_layers = mlp_config.get_layers(self._n_layers)
            
            handles = register_mlp_hooks(
                self._model,
                mlp_layers,
                mlp_storage,
                capture_pre_activation=mlp_config.capture_pre_activation,
                capture_post_activation=mlp_config.capture_post_activation,
                capture_gate=mlp_config.capture_gate,
                capture_output=mlp_config.capture_output,
                dtype=mlp_config.storage_dtype,
            )
            all_handles.extend(handles)
        
        # Run forward pass
        try:
            input_ids = input_ids.to(self._device)
            with torch.no_grad():
                outputs = self._model(
                    input_ids,
                    output_attentions=attention_config.capture_patterns if attention_config else False,
                    return_dict=True
                )
            
            # Handle attention patterns
            captured_patterns = {}
            if attention_config and attention_config.capture_patterns:
                if outputs.attentions is not None:
                    for layer_idx in attn_layers:
                        captured_patterns[layer_idx] = outputs.attentions[layer_idx].cpu()
                else:
                    captured_patterns = self._compute_attention_patterns(
                        captured_q, captured_k, attn_layers, n_tokens
                    )
        finally:
            for h in all_handles:
                h.remove()
        
        # Assemble captures
        attention_capture = None
        if attention_config is not None:
            queries = self._assemble_tensor(
                captured_q, attn_layers, self._n_heads, n_tokens
            ) if attention_config.capture_queries else None
            
            keys = self._assemble_tensor(
                captured_k, attn_layers, self._n_kv_heads, n_tokens
            ) if attention_config.capture_keys else None
            
            values = self._assemble_tensor(
                captured_v, attn_layers, self._n_kv_heads, n_tokens
            ) if attention_config.capture_values else None
            
            patterns = self._assemble_patterns(
                captured_patterns, attn_layers
            ) if attention_config.capture_patterns else None
            
            attention_capture = AttentionCapture(
                queries=queries,
                keys=keys,
                values=values,
                patterns=patterns,
                n_layers=len(attn_layers),
                n_heads=self._n_heads,
                n_kv_heads=self._n_kv_heads,
                n_tokens=n_tokens,
                head_dim=self._head_dim,
                model_name=self._model_name,
                qk_stage=attention_config.qk_stage,
                captured_layers=tuple(attn_layers),
            )
        
        residual_capture = None
        if residual_config is not None:
            residual_capture = assemble_residual_capture(
                residual_storage,
                residual_layers,
                n_tokens,
                hidden_dim,
                self._model_name,
            )
        
        mlp_capture = None
        if mlp_config is not None:
            mlp_capture = assemble_mlp_capture(
                mlp_storage,
                mlp_layers,
                n_tokens,
                hidden_dim,
                intermediate_dim,
                self._model_name,
            )
        
        return FullCapture(
            attention=attention_capture,
            residual=residual_capture,
            mlp=mlp_capture,
        )
    
    # =========================================================================
    # GENERATION API
    # =========================================================================
    
    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        if isinstance(prompt, str):
            input_ids = self.tokenize(prompt)
        else:
            input_ids = prompt
        
        input_ids = input_ids.to(self._device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
                **kwargs
            )
        
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def tokenize(self, text: str) -> torch.Tensor:
        return self._tokenizer.encode(text, return_tensors='pt')
    
    def decode(self, token_ids: torch.Tensor) -> str:
        return self._tokenizer.decode(token_ids.squeeze(), skip_special_tokens=True)
    
    # =========================================================================
    # HOOK MANAGEMENT
    # =========================================================================
    
    def add_hook(
        self,
        hook_fn: Callable,
        layer: int,
        component: str,
        stage: str = "post_proj"
    ) -> str:
        """
        Add a forward hook at specified location.
        
        Components:
            - 'q': After Q projection (model.layers[l].self_attn.q_proj)
            - 'k': After K projection (model.layers[l].self_attn.k_proj)
            - 'v': After V projection (model.layers[l].self_attn.v_proj)
            - 'attn_out': After output projection (model.layers[l].self_attn.o_proj)
            - 'pattern': Full attention module (for patterns)
        """
        if component == 'q':
            module = get_q_proj(self._model, layer)
        elif component == 'k':
            module = get_k_proj(self._model, layer)
        elif component == 'v':
            module = get_v_proj(self._model, layer)
        elif component == 'attn_out':
            module = get_o_proj(self._model, layer)
        elif component == 'pattern':
            module = get_attention_module(self._model, layer)
        else:
            raise ValueError(
                f"Unknown component: {component}. "
                f"Must be one of: 'q', 'k', 'v', 'attn_out', 'pattern'"
            )
        
        handle = module.register_forward_hook(hook_fn)
        
        handle_id = f"hook_{self._hook_counter}"
        self._hook_counter += 1
        self._hooks[handle_id] = handle
        
        return handle_id
    
    def remove_hook(self, handle_id: str) -> None:
        if handle_id in self._hooks:
            self._hooks[handle_id].remove()
            del self._hooks[handle_id]
    
    def reset_hooks(self) -> None:
        for handle in self._hooks.values():
            handle.remove()
        self._hooks.clear()
