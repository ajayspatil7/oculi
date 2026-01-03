"""
Core Data Structures for Spectra
================================

Defines the public data structures used throughout the framework.
All structures follow the API contract in docs/API_CONTRACT.md.

These are IMMUTABLE containers — they hold data, not behavior.
"""

from dataclasses import dataclass
from typing import Optional, List, Literal
import torch


@dataclass(frozen=True)
class AttentionStructure:
    """
    Semantic description of a model's attention architecture.
    
    Abstracts away model-specific details into semantic categories.
    Use this to determine GQA/MQA structure without model-specific logic.
    
    Attributes:
        n_query_heads: Number of query attention heads
        n_kv_heads: Number of key/value attention heads
        head_dim: Dimension of each attention head
        
    Example:
        >>> struct = adapter.attention_structure(layer=0)
        >>> if struct.attention_type == "GQA":
        ...     print(f"GQA with {struct.gqa_ratio}:1 ratio")
    """
    n_query_heads: int
    n_kv_heads: int
    head_dim: int
    
    @property
    def attention_type(self) -> str:
        """
        Returns attention type: 'MHA', 'GQA', or 'MQA'.
        
        - MHA: Multi-Head Attention (n_query_heads == n_kv_heads)
        - GQA: Grouped Query Attention (n_query_heads > n_kv_heads > 1)
        - MQA: Multi-Query Attention (n_kv_heads == 1)
        """
        if self.n_query_heads == self.n_kv_heads:
            return "MHA"
        elif self.n_kv_heads == 1:
            return "MQA"
        else:
            return "GQA"
    
    @property
    def gqa_ratio(self) -> int:
        """Number of query heads per KV head."""
        return self.n_query_heads // self.n_kv_heads
    
    @property
    def is_gqa(self) -> bool:
        """True if model uses Grouped Query Attention."""
        return self.n_query_heads != self.n_kv_heads


@dataclass(frozen=True)
class AttentionCapture:
    """
    Immutable container for captured attention data from a forward pass.
    
    All tensors are detached from computation graph and on CPU.
    This is the primary output of the capture system.
    
    Shape Contracts (from API_CONTRACT.md):
        - queries: [n_layers, n_heads, n_tokens, head_dim]
        - keys: [n_layers, n_kv_heads, n_tokens, head_dim]
        - values: [n_layers, n_kv_heads, n_tokens, head_dim]
        - patterns: [n_layers, n_heads, n_tokens, n_tokens]
    
    Invariants:
        - All tensors are on CPU
        - All tensors are detached (no gradients)
        - patterns sums to 1.0 along last dimension
        - patterns[l, h, i, j] = 0 for j > i (causal masking)
        
    Example:
        >>> capture = model.capture(input_ids)
        >>> print(capture.queries.shape)  # [L, H, T, D]
        >>> print(capture.is_gqa)  # True for LLaMA-3
    """
    
    # Captured tensors (may be None if not requested in config)
    queries: Optional[torch.Tensor] = None   # [L, H_q, T, D]
    keys: Optional[torch.Tensor] = None      # [L, H_kv, T, D]
    values: Optional[torch.Tensor] = None    # [L, H_kv, T, D]
    patterns: Optional[torch.Tensor] = None  # [L, H_q, T, T]
    
    # Metadata
    n_layers: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    n_tokens: int = 0
    head_dim: int = 0
    model_name: str = ""
    
    # Capture configuration
    qk_stage: str = "pre_rope"  # 'pre_rope' or 'post_rope'
    captured_layers: tuple = ()  # Which layers were captured
    
    @property
    def is_gqa(self) -> bool:
        """True if model uses Grouped Query Attention."""
        return self.n_heads != self.n_kv_heads
    
    @property
    def gqa_ratio(self) -> int:
        """Number of query heads per KV head."""
        if self.n_kv_heads == 0:
            return 1
        return self.n_heads // self.n_kv_heads
    
    def __post_init__(self):
        """Validate tensor shapes match metadata."""
        # Note: frozen=True means we can't modify, but __post_init__ runs
        # before freezing, so we can validate
        if self.queries is not None:
            expected = (self.n_layers, self.n_heads, self.n_tokens, self.head_dim)
            if self.queries.shape != expected:
                raise ValueError(
                    f"queries shape {self.queries.shape} != expected {expected}"
                )


@dataclass
class CaptureConfig:
    """
    Configuration for what attention data to capture.
    
    Use to control memory usage by capturing only needed components.
    
    Attributes:
        layers: Which layers to capture (None = all)
        capture_queries: Whether to capture Q vectors
        capture_keys: Whether to capture K vectors
        capture_values: Whether to capture V vectors
        capture_patterns: Whether to capture attention patterns
        qk_stage: When to capture Q/K ('pre_rope' or 'post_rope')
        
    Example:
        >>> config = CaptureConfig(
        ...     layers=[20, 21, 22],  # Only last few layers
        ...     capture_values=False,  # Don't need V
        ...     qk_stage='pre_rope'    # Position-agnostic
        ... )
        >>> capture = model.capture(input_ids, config=config)
    """
    
    # Which layers to capture (None = all)
    layers: Optional[List[int]] = None
    
    # Which components to capture
    capture_queries: bool = True
    capture_keys: bool = True
    capture_values: bool = True
    capture_patterns: bool = True
    
    # Capture stage for Q/K vectors
    # 'pre_rope': Before rotary position embedding (position-agnostic)
    # 'post_rope': After rotary position embedding (position-aware)
    qk_stage: Literal['pre_rope', 'post_rope'] = 'pre_rope'
    
    def validate(self, num_layers: int) -> None:
        """
        Validate config against model.
        
        Args:
            num_layers: Total number of layers in model
            
        Raises:
            ValueError: If layers out of range or invalid configuration
        """
        # Check at least one capture flag is True
        if not any([
            self.capture_queries,
            self.capture_keys,
            self.capture_values,
            self.capture_patterns
        ]):
            raise ValueError("At least one capture flag must be True")
        
        # Check layer indices
        if self.layers is not None:
            for layer in self.layers:
                if not (0 <= layer < num_layers):
                    raise ValueError(
                        f"Layer {layer} out of range [0, {num_layers})"
                    )
        
        # Check qk_stage
        if self.qk_stage not in ('pre_rope', 'post_rope'):
            raise ValueError(
                f"qk_stage must be 'pre_rope' or 'post_rope', got '{self.qk_stage}'"
            )
    
    def get_layers(self, num_layers: int) -> List[int]:
        """Get list of layers to capture (resolves None to all)."""
        if self.layers is None:
            return list(range(num_layers))
        return list(self.layers)


# =============================================================================
# RESIDUAL STREAM CAPTURE
# =============================================================================

@dataclass
class ResidualConfig:
    """
    Configuration for residual stream capture.
    
    Controls which residual stream positions to capture and memory usage.
    
    Attributes:
        capture_pre_attn: Capture before attention (block input)
        capture_post_attn: Capture after attention (before residual add)
        capture_pre_mlp: Capture before MLP (after post_attention_layernorm)
        capture_post_mlp: Capture after MLP (before final residual add)
        layers: Which layers to capture (None = all)
        storage_dtype: Data type for storage (float32 or float16)
        
    Example:
        >>> config = ResidualConfig(
        ...     capture_pre_attn=False,  # Skip pre-attention
        ...     layers=(20, 21, 22),     # Only last few layers
        ...     storage_dtype=torch.float16  # Save memory
        ... )
        >>> residual = adapter.capture_residual(input_ids, config)
    """
    capture_pre_attn: bool = True
    capture_post_attn: bool = True
    capture_pre_mlp: bool = True
    capture_post_mlp: bool = True
    layers: Optional[tuple] = None  # Tuple for immutability
    storage_dtype: 'torch.dtype' = None  # Default to float32
    
    def __post_init__(self):
        if self.storage_dtype is None:
            self.storage_dtype = torch.float32
    
    def validate(self, num_layers: int) -> None:
        """Validate config against model."""
        if not any([
            self.capture_pre_attn,
            self.capture_post_attn,
            self.capture_pre_mlp,
            self.capture_post_mlp
        ]):
            raise ValueError("At least one capture flag must be True")
        
        if self.layers is not None:
            for layer in self.layers:
                if not (0 <= layer < num_layers):
                    raise ValueError(f"Layer {layer} out of range [0, {num_layers})")
    
    def get_layers(self, num_layers: int) -> List[int]:
        """Get list of layers to capture (resolves None to all)."""
        if self.layers is None:
            return list(range(num_layers))
        return list(self.layers)


@dataclass(frozen=True)
class ResidualCapture:
    """
    Immutable container for captured residual stream activations.
    
    Captures the hidden states at key points in the transformer block:
    - pre_attn: Before attention (block input after input_layernorm)
    - post_attn: After attention (attention output, before residual add)
    - pre_mlp: Before MLP (after post_attention_layernorm)
    - post_mlp: After MLP (MLP output, before final residual add)
    
    Shape Contracts:
        All tensors: [n_layers, n_tokens, hidden_dim]
        
    Invariants:
        - All tensors are on CPU
        - All tensors are detached (no gradients)
        - post_mlp[i] + residual ≈ pre_attn[i+1] (residual stream continuity)
        
    Example:
        >>> residual = adapter.capture_residual(input_ids)
        >>> print(residual.pre_attn.shape)  # [L, T, H]
        >>> layer_5 = residual.get_layer(5)
        >>> print(layer_5['pre_attn'].shape)  # [T, H]
    """
    
    # Captured tensors (may be None if not requested in config)
    pre_attn: Optional[torch.Tensor] = None   # [L, T, H]
    post_attn: Optional[torch.Tensor] = None  # [L, T, H]
    pre_mlp: Optional[torch.Tensor] = None    # [L, T, H]
    post_mlp: Optional[torch.Tensor] = None   # [L, T, H]
    
    # Metadata
    n_layers: int = 0
    n_tokens: int = 0
    hidden_dim: int = 0
    model_name: str = ""
    captured_layers: tuple = ()
    
    def get_layer(self, layer_idx: int) -> dict:
        """
        Get all captured tensors for a specific layer.
        
        Args:
            layer_idx: Layer index (must be in captured_layers)
            
        Returns:
            Dict with keys 'pre_attn', 'post_attn', 'pre_mlp', 'post_mlp'
            Each value is [T, H] tensor or None
            
        Raises:
            ValueError: If layer_idx not in captured_layers
        """
        if layer_idx not in self.captured_layers:
            raise ValueError(
                f"Layer {layer_idx} not captured. "
                f"Available: {self.captured_layers}"
            )
        
        idx = self.captured_layers.index(layer_idx)
        return {
            'pre_attn': self.pre_attn[idx] if self.pre_attn is not None else None,
            'post_attn': self.post_attn[idx] if self.post_attn is not None else None,
            'pre_mlp': self.pre_mlp[idx] if self.pre_mlp is not None else None,
            'post_mlp': self.post_mlp[idx] if self.post_mlp is not None else None,
        }
    
    def stream_at(self, position: str, layer: int) -> Optional[torch.Tensor]:
        """
        Get residual stream tensor at specific position and layer.
        
        Args:
            position: One of 'pre_attn', 'post_attn', 'pre_mlp', 'post_mlp'
            layer: Layer index
            
        Returns:
            [T, H] tensor or None if not captured
        """
        layer_data = self.get_layer(layer)
        return layer_data.get(position)
    
    def memory_usage(self) -> int:
        """
        Calculate total memory usage in bytes.
        
        Returns:
            Total bytes used by all captured tensors
        """
        total = 0
        for tensor in [self.pre_attn, self.post_attn, self.pre_mlp, self.post_mlp]:
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total


# =============================================================================
# MLP CAPTURE
# =============================================================================

@dataclass
class MLPConfig:
    """
    Configuration for MLP internals capture.
    
    LLaMA MLP structure:
        gate = gate_proj(x)           # [B, T, I]
        up = up_proj(x)               # [B, T, I]
        hidden = silu(gate) * up      # [B, T, I] - post_activation
        output = down_proj(hidden)    # [B, T, H]
    
    Attributes:
        capture_pre_activation: Capture before SiLU (gate_proj output)
        capture_post_activation: Capture after SiLU*up (neuron activations)
        capture_gate: Capture gate_proj output
        capture_output: Capture final MLP output
        layers: Which layers to capture (None = all)
        storage_dtype: Data type for storage
        
    Example:
        >>> config = MLPConfig(
        ...     capture_gate=True,           # Also get gate values
        ...     layers=(20, 21, 22),         # Only last few layers
        ... )
        >>> mlp = adapter.capture_mlp(input_ids, config)
    """
    capture_pre_activation: bool = True
    capture_post_activation: bool = True
    capture_gate: bool = False
    capture_output: bool = True
    layers: Optional[tuple] = None
    storage_dtype: 'torch.dtype' = None
    
    def __post_init__(self):
        if self.storage_dtype is None:
            self.storage_dtype = torch.float32
    
    def validate(self, num_layers: int) -> None:
        """Validate config against model."""
        if not any([
            self.capture_pre_activation,
            self.capture_post_activation,
            self.capture_gate,
            self.capture_output
        ]):
            raise ValueError("At least one capture flag must be True")
        
        if self.layers is not None:
            for layer in self.layers:
                if not (0 <= layer < num_layers):
                    raise ValueError(f"Layer {layer} out of range [0, {num_layers})")
    
    def get_layers(self, num_layers: int) -> List[int]:
        """Get list of layers to capture (resolves None to all)."""
        if self.layers is None:
            return list(range(num_layers))
        return list(self.layers)


@dataclass(frozen=True)
class MLPCapture:
    """
    Immutable container for captured MLP internals.
    
    LLaMA MLP computation:
        hidden = silu(gate_proj(x)) * up_proj(x)
        output = down_proj(hidden)
    
    Shape Contracts:
        - pre_activation: [L, T, I] (gate_proj output, before SiLU)
        - post_activation: [L, T, I] (silu(gate) * up, "neuron activations")
        - gate_output: [L, T, I] (gate_proj output)
        - mlp_output: [L, T, H] (final MLP output)
        
    Where I = intermediate_dim (14336 for LLaMA-3-8B)
    
    Example:
        >>> mlp = adapter.capture_mlp(input_ids)
        >>> print(mlp.post_activation.shape)  # [L, T, 14336]
    """
    
    # Captured tensors
    pre_activation: Optional[torch.Tensor] = None    # [L, T, I]
    post_activation: Optional[torch.Tensor] = None   # [L, T, I]
    gate_output: Optional[torch.Tensor] = None       # [L, T, I]
    mlp_output: Optional[torch.Tensor] = None        # [L, T, H]
    
    # Metadata
    n_layers: int = 0
    n_tokens: int = 0
    hidden_dim: int = 0
    intermediate_dim: int = 0
    model_name: str = ""
    captured_layers: tuple = ()
    
    def get_layer(self, layer_idx: int) -> dict:
        """Get all captured tensors for a specific layer."""
        if layer_idx not in self.captured_layers:
            raise ValueError(
                f"Layer {layer_idx} not captured. "
                f"Available: {self.captured_layers}"
            )
        
        idx = self.captured_layers.index(layer_idx)
        return {
            'pre_activation': self.pre_activation[idx] if self.pre_activation is not None else None,
            'post_activation': self.post_activation[idx] if self.post_activation is not None else None,
            'gate_output': self.gate_output[idx] if self.gate_output is not None else None,
            'mlp_output': self.mlp_output[idx] if self.mlp_output is not None else None,
        }
    
    def memory_usage(self) -> int:
        """Calculate total memory usage in bytes."""
        total = 0
        for tensor in [self.pre_activation, self.post_activation, self.gate_output, self.mlp_output]:
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total


# =============================================================================
# LOGIT CAPTURE (for Logit Lens)
# =============================================================================

@dataclass
class LogitConfig:
    """
    Configuration for layer-wise logit capture (logit lens).
    
    Logit lens applies the unembedding matrix to each layer's output
    to see what the model would predict at that layer.
    
    Attributes:
        layers: Which layers to capture logits at (None = all)
        top_k: Only store top-k logits per position (memory optimization)
        storage_dtype: Data type for storage
        
    Memory Note:
        Full logits for LLaMA-3-8B (32 layers, 512 tokens, 128k vocab) ≈ 8GB+
        Use top_k for practical analysis (e.g., top_k=100)
        
    Example:
        >>> config = LogitConfig(top_k=50)  # Only store top 50 per position
        >>> logits = adapter.capture_logits(input_ids, config)
    """
    layers: Optional[tuple] = None
    top_k: Optional[int] = None  # None = store all logits
    storage_dtype: 'torch.dtype' = None
    
    def __post_init__(self):
        if self.storage_dtype is None:
            self.storage_dtype = torch.float32
    
    def validate(self, num_layers: int) -> None:
        """Validate config against model."""
        if self.layers is not None:
            for layer in self.layers:
                if not (0 <= layer < num_layers):
                    raise ValueError(f"Layer {layer} out of range [0, {num_layers})")
        
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
    
    def get_layers(self, num_layers: int) -> List[int]:
        """Get list of layers to capture (resolves None to all)."""
        if self.layers is None:
            return list(range(num_layers))
        return list(self.layers)


@dataclass(frozen=True)
class LogitCapture:
    """
    Immutable container for layer-wise logits (logit lens).
    
    Logits are computed by applying the unembedding matrix to each layer's
    residual stream: logits[l] = residual[l] @ lm_head.weight.T
    
    Shape Contracts:
        - logits: [L, T, V] (full vocabulary logits)
        - top_k_logits: [L, T, K] (if top_k specified)
        - top_k_indices: [L, T, K] (token indices for top_k)
        
    Example:
        >>> logits = adapter.capture_logits(input_ids, LogitConfig(top_k=10))
        >>> print(logits.top_k_indices.shape)  # [L, T, 10]
    """
    
    # Full logits (memory-intensive)
    logits: Optional[torch.Tensor] = None           # [L, T, V]
    
    # Top-k logits (memory-efficient)
    top_k_logits: Optional[torch.Tensor] = None     # [L, T, K]
    top_k_indices: Optional[torch.Tensor] = None    # [L, T, K]
    
    # Metadata
    n_layers: int = 0
    n_tokens: int = 0
    vocab_size: int = 0
    model_name: str = ""
    captured_layers: tuple = ()
    
    def memory_usage(self) -> int:
        """Calculate total memory usage in bytes."""
        total = 0
        for tensor in [self.logits, self.top_k_logits, self.top_k_indices]:
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total


# =============================================================================
# FULL CAPTURE CONTAINER
# =============================================================================

@dataclass
class FullCapture:
    """
    Container for combined capture results.
    
    Allows capturing attention, residual, and MLP in a single forward pass
    for efficiency.
    
    Example:
        >>> full = adapter.capture_full(input_ids)
        >>> print(full.attention.queries.shape)
        >>> print(full.residual.pre_attn.shape)
        >>> print(full.mlp.post_activation.shape)
    """
    attention: Optional[AttentionCapture] = None
    residual: Optional[ResidualCapture] = None
    mlp: Optional[MLPCapture] = None

