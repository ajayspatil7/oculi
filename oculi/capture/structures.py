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
