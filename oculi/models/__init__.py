"""
Oculi Model Adapters
====================

Public model-specific adapters. Each adapter is *executable documentation*
of how a model implements attention.

Supported Models:
    - LlamaAttentionAdapter: LLaMA 2/3 family (with GQA support)

Usage:
    from oculi.models.llama import LlamaAttentionAdapter
    
    adapter = LlamaAttentionAdapter(model, tokenizer)
    capture = adapter.capture(input_ids)
"""

from oculi.models.base import (
    AttentionAdapter,
    UnsupportedModelError,
    CaptureError,
)

# Model-specific adapters
from oculi.models.llama import LlamaAttentionAdapter

__all__ = [
    # Base
    "AttentionAdapter",
    "UnsupportedModelError", 
    "CaptureError",
    
    # LLaMA
    "LlamaAttentionAdapter",
]
