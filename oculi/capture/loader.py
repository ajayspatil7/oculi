"""
Model Loader
============

Provides the main entry point for loading models with auto-detection.
"""

from typing import Optional
import torch

from oculi.capture.adapter import ModelAdapter, UnsupportedModelError


# =============================================================================
# MODEL REGISTRY
# =============================================================================

# Maps model name patterns to adapter classes
# Actual adapter implementations are in oculi._private.adapters
_MODEL_REGISTRY = {}


def register_adapter(pattern: str, adapter_class: type) -> None:
    """
    Register an adapter class for a model pattern.
    
    Called by _private.adapters during initialization.
    
    Args:
        pattern: Glob pattern for model names (e.g., "meta-llama/*")
        adapter_class: ModelAdapter subclass
    """
    _MODEL_REGISTRY[pattern] = adapter_class


def _match_pattern(model_name: str, pattern: str) -> bool:
    """Simple glob matching for model names."""
    import fnmatch
    return fnmatch.fnmatch(model_name.lower(), pattern.lower())


def _find_adapter_class(model_name: str) -> type:
    """Find the appropriate adapter class for a model name."""
    for pattern, adapter_class in _MODEL_REGISTRY.items():
        if _match_pattern(model_name, pattern):
            return adapter_class
    
    raise UnsupportedModelError(
        f"No adapter registered for model: {model_name}\n"
        f"Supported patterns: {list(_MODEL_REGISTRY.keys())}"
    )


# =============================================================================
# PUBLIC LOADER
# =============================================================================

def load(
    model_name: str,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs
) -> ModelAdapter:
    """
    Load a model with auto-detected adapter.
    
    This is the primary entry point for using Spectra.
    
    Args:
        model_name: HuggingFace model name or path
            Examples:
                - "meta-llama/Meta-Llama-3-8B"
                - "mistralai/Mistral-7B-v0.1"
                - "Qwen/Qwen2.5-7B-Instruct"
        device: Device to load model on (default: auto-detect GPU)
        dtype: Model dtype (default: float16 for GPU, float32 for CPU)
        **kwargs: Additional arguments passed to adapter
        
    Returns:
        ModelAdapter instance ready for capture/generation
        
    Raises:
        UnsupportedModelError: If no adapter exists for model
        
    Example:
        >>> import oculi
        >>> model = oculi.load("meta-llama/Meta-Llama-3-8B")
        >>> capture = model.capture(input_ids)
    """
    # Import private adapters to trigger registration
    # This is the ONLY place where _private is imported from public code
    from oculi._private import adapters as _  # noqa: F401
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto-detect dtype
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Find and instantiate adapter
    adapter_class = _find_adapter_class(model_name)
    
    return adapter_class(
        model_name=model_name,
        device=device,
        dtype=dtype,
        **kwargs
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_supported_models() -> list:
    """Return list of supported model patterns."""
    # Import to ensure registration
    from oculi._private import adapters as _  # noqa: F401
    return list(_MODEL_REGISTRY.keys())


def is_model_supported(model_name: str) -> bool:
    """Check if a model is supported."""
    try:
        _find_adapter_class(model_name)
        return True
    except UnsupportedModelError:
        return False
