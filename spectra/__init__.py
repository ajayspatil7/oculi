"""
Spectra Package
===============

Unified experiment infrastructure for transformer attention analysis.
"""

from .models import ModelAdapter, ModelInfo, load_model
from .hooks import UnifiedHooks
from .metrics import (
    compute_attention_entropy,
    compute_max_attention_weight,
    compute_effective_attention_span,
    compute_query_norms,
    compute_linear_gain,
)

__version__ = "1.0.0"
__all__ = [
    "ModelAdapter",
    "ModelInfo", 
    "load_model",
    "UnifiedHooks",
    "compute_attention_entropy",
    "compute_max_attention_weight",
    "compute_effective_attention_span",
    "compute_query_norms",
    "compute_linear_gain",
]
