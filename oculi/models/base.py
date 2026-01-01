"""
Base Adapter Contract
=====================

Abstract interface that all model adapters must implement.
This defines WHAT operations are available — model families implement HOW.

This is PUBLIC API and executable documentation of attention anatomy.

Example:
    from oculi.models.llama import LlamaAttentionAdapter
    adapter = LlamaAttentionAdapter(model)
    capture = adapter.capture(input_ids)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Union, Callable
import torch

from oculi.capture.structures import (
    AttentionCapture,
    AttentionStructure,
    CaptureConfig,
)


class AttentionAdapter(ABC):
    """
    Abstract contract for model-specific attention adapters.
    
    Each adapter answers the question:
    "How does this model implement attention?"
    
    Subclasses must implement:
        - Architecture introspection (num_layers, num_heads, etc.)
        - Tensor capture (capture)
        - Hook management (add_hook, remove_hook)
        - Generation (generate, tokenize, decode)
    
    Implementation lives in:
        oculi/models/{model_family}/adapter.py
    
    Anatomy details live in:
        oculi/models/{model_family}/attention.py
    
    Guarantees:
        - capture() is deterministic given same inputs
        - capture() does not modify model weights
        - All returned tensors are detached and on CPU
    """
    
    # =========================================================================
    # ARCHITECTURE INTROSPECTION
    # =========================================================================
    
    @abstractmethod
    def num_layers(self) -> int:
        """Total number of transformer layers."""
        ...
    
    @abstractmethod
    def num_heads(self, layer: int = 0) -> int:
        """Number of query attention heads at given layer."""
        ...
    
    @abstractmethod
    def num_kv_heads(self, layer: int = 0) -> int:
        """Number of key/value attention heads (for GQA/MQA)."""
        ...
    
    @abstractmethod
    def head_dim(self, layer: int = 0) -> int:
        """Dimension of each attention head."""
        ...
    
    @abstractmethod
    def attention_structure(self, layer: int = 0) -> AttentionStructure:
        """Returns semantic description of attention (MHA/GQA/MQA)."""
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
        
        Args:
            input_ids: Token IDs with shape [batch=1, seq_len]
            config: What to capture (default: all components, all layers)
            
        Returns:
            AttentionCapture with requested data
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
        """Generate text completion."""
        ...
    
    @abstractmethod
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text to input IDs [1, seq_len]."""
        ...
    
    @abstractmethod
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        ...
    
    # =========================================================================
    # HOOK MANAGEMENT
    # =========================================================================
    
    @abstractmethod
    def add_hook(
        self,
        hook_fn: Callable,
        layer: int,
        component: str,
        stage: str = "post_proj"
    ) -> str:
        """
        Add a forward hook at specified location.
        
        Args:
            hook_fn: Hook function (module, input, output) -> None
            layer: Layer index (0 to num_layers-1)
            component: 'q', 'k', 'v', 'attn_out', or 'pattern'
            stage: Hook stage (model-specific)
            
        Returns:
            Hook handle ID (for removal)
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
