"""
Model Adapter Interface
=======================

Defines the abstract interface that all model adapters must implement.
This is the PUBLIC API — implementation details live in _private/adapters/.

The adapter pattern provides:
1. Model-agnostic capture interface
2. Unified intervention API
3. Architecture introspection
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Union
import torch

from oculi.capture.structures import (
    AttentionCapture,
    AttentionStructure,
    CaptureConfig,
)


class ModelAdapter(ABC):
    """
    Model-agnostic interface for transformer instrumentation.
    
    This is the primary entry point for interacting with models.
    Concrete implementations live in oculi._private.adapters.
    
    Public methods define WHAT operations are available.
    Private adapters implement HOW for specific model families.
    
    Guarantees:
        - capture() is deterministic given same inputs
        - capture() does not modify model weights
        - All returned tensors are detached and on CPU
        
    Example:
        >>> adapter = oculi.load("meta-llama/Meta-Llama-3-8B")
        >>> capture = adapter.capture(input_ids)
        >>> print(adapter.num_layers())  # 32
        >>> print(adapter.attention_structure().attention_type)  # "GQA"
    """
    
    # =========================================================================
    # ARCHITECTURE INTROSPECTION
    # =========================================================================
    
    @abstractmethod
    def num_layers(self) -> int:
        """
        Total number of transformer layers.
        
        Returns:
            Number of layers (e.g., 32 for LLaMA-3-8B)
        """
        ...
    
    @abstractmethod
    def num_heads(self, layer: int = 0) -> int:
        """
        Number of query attention heads at given layer.
        
        Args:
            layer: Layer index (default 0, usually same for all layers)
            
        Returns:
            Number of query heads (e.g., 32 for LLaMA-3-8B)
        """
        ...
    
    @abstractmethod
    def num_kv_heads(self, layer: int = 0) -> int:
        """
        Number of key/value attention heads at given layer.
        
        Args:
            layer: Layer index
            
        Returns:
            Number of KV heads (e.g., 8 for LLaMA-3-8B with GQA)
        """
        ...
    
    @abstractmethod
    def head_dim(self, layer: int = 0) -> int:
        """
        Dimension of each attention head.
        
        Args:
            layer: Layer index
            
        Returns:
            Head dimension (e.g., 128)
        """
        ...
    
    @abstractmethod
    def attention_structure(self, layer: int = 0) -> AttentionStructure:
        """
        Returns semantic description of attention at given layer.
        
        Use this to determine GQA/MQA structure without model-specific logic.
        
        Args:
            layer: Layer index
            
        Returns:
            AttentionStructure with n_query_heads, n_kv_heads, head_dim
            
        Example:
            >>> struct = adapter.attention_structure()
            >>> if struct.attention_type == "GQA":
            ...     print(f"GQA ratio: {struct.gqa_ratio}:1")
        """
        ...
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Full model name/path."""
        ...
    
    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device model is loaded on."""
        ...
    
    # =========================================================================
    # CAPTURE API
    # =========================================================================
    
    @abstractmethod
    def capture(
        self,
        input_ids: torch.Tensor,
        config: Optional[CaptureConfig] = None
    ) -> AttentionCapture:
        """
        Run forward pass and capture attention data.
        
        This is the primary capture method. Registers temporary hooks,
        runs the forward pass, and returns captured data.
        
        Args:
            input_ids: Token IDs with shape [batch=1, seq_len]
                       Batch size must be 1 (single sequence capture)
            config: What to capture (default: all components, all layers)
            
        Returns:
            AttentionCapture with requested data (see API_CONTRACT.md)
            
        Raises:
            ValueError: If config validation fails
            ValueError: If batch size != 1
            RuntimeError: If capture hooks fail
            
        Guarantees:
            - Deterministic output for same input
            - Model weights unchanged
            - All tensors detached and on CPU
            
        Example:
            >>> capture = adapter.capture(
            ...     input_ids,
            ...     config=CaptureConfig(layers=[20, 21, 22])
            ... )
            >>> print(capture.patterns.shape)
        """
        ...
    
    # =========================================================================
    # GENERATION API
    # =========================================================================
    
    @abstractmethod
    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input text or token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = neutral)
            do_sample: Whether to sample (vs greedy decoding)
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text (including prompt)
            
        Example:
            >>> output = adapter.generate(
            ...     "The capital of France is",
            ...     max_new_tokens=20,
            ...     temperature=0.0  # Greedy
            ... )
        """
        ...
    
    @abstractmethod
    def tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize text to input IDs.
        
        Args:
            text: Input text
            
        Returns:
            Token IDs with shape [1, seq_len]
        """
        ...
    
    @abstractmethod
    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs tensor
            
        Returns:
            Decoded text string
        """
        ...
    
    # =========================================================================
    # INTERVENTION API (Hooks management)
    # =========================================================================
    
    @abstractmethod
    def add_hook(
        self,
        hook_fn,
        layer: int,
        component: str,
        stage: str = "post_proj"
    ) -> str:
        """
        Add a forward hook at specified location.
        
        This is a low-level method. Prefer using InterventionContext.
        
        Args:
            hook_fn: Hook function matching PyTorch signature
            layer: Layer index
            component: 'q', 'k', 'v', 'attn_out', 'pattern'
            stage: 'pre_proj', 'post_proj', 'pre_rope', 'post_rope'
            
        Returns:
            Hook handle ID (for removal)
            
        Note:
            The actual hook point name is model-specific and resolved
            by the private adapter implementation.
        """
        ...
    
    @abstractmethod
    def remove_hook(self, handle_id: str) -> None:
        """Remove a specific hook by handle ID."""
        ...
    
    @abstractmethod
    def reset_hooks(self) -> None:
        """Remove all registered hooks."""
        ...


# =============================================================================
# EXCEPTIONS
# =============================================================================

class UnsupportedModelError(Exception):
    """Raised when no adapter exists for a model."""
    pass


class CaptureError(RuntimeError):
    """Raised when capture hooks fail."""
    pass
