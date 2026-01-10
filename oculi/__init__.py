"""
Oculi: Mechanistic Interpretability Toolkit
============================================

A low-level, surgical instrumentation layer for LLMs.

Public API
----------
Model Adapters (explicit, no magic):
    from oculi.models.llama import LlamaAttentionAdapter

Data Structures:
    - AttentionCapture: Captured attention data
    - AttentionStructure: Model architecture info
    - CaptureConfig: Capture configuration
    - ResidualCapture: Residual stream activations
    - MLPCapture: MLP internals
    - LogitCapture: Layer-wise logits
    - FullCapture: Combined capture container

Modules:
    - oculi.models: Model-specific adapters
    - oculi.capture: Capture utilities
    - oculi.analysis: Analysis functions
    - oculi.intervention: Intervention classes

Example
-------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from oculi.models.llama import LlamaAttentionAdapter
    
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    adapter = LlamaAttentionAdapter(model, tokenizer)
    capture = adapter.capture(input_ids)

Version: 0.3.0-dev
"""

__version__ = "0.3.0-dev"
__author__ = "Ajay S Patil"

# =============================================================================
# PUBLIC API — Core Data Structures
# =============================================================================

from oculi.capture.structures import (
    AttentionCapture,
    AttentionStructure,
    CaptureConfig,
    ResidualCapture,
    ResidualConfig,
    MLPCapture,
    MLPConfig,
    LogitCapture,
    LogitConfig,
    FullCapture,
)

# Device utilities
from oculi.utils.device import (
    get_default_device,
    get_device_info,
    is_mps_available,
    is_cuda_available,
    auto_select_device,
)

# Activation Patching (Phase 2.3 - v0.6.0)
from oculi.intervention import (
    PatchConfig,
    ActivationPatch,
    PatchingResult,
    PatchingSweepResult,
    PatchingContext,
    CausalTracer,
)

# Base adapter contract
from oculi.models.base import (
    AttentionAdapter,
    UnsupportedModelError,
    CaptureError,
)

# =============================================================================
# PUBLIC API — Modules (imported as namespaces)
# =============================================================================

from oculi import models
from oculi import analysis
from oculi import intervention
from oculi import capture

# =============================================================================
# PUBLIC API LIST
# =============================================================================

__all__ = [
    # Version
    "__version__",

    # Core structures
    "AttentionCapture",
    "AttentionStructure",
    "CaptureConfig",

    # Phase 1 structures
    "ResidualCapture",
    "ResidualConfig",
    "MLPCapture",
    "MLPConfig",
    "LogitCapture",
    "LogitConfig",
    "FullCapture",

    # Base adapter
    "AttentionAdapter",
    "UnsupportedModelError",
    "CaptureError",

    # Device utilities (MPS/CUDA/CPU support)
    "get_default_device",
    "get_device_info",
    "is_mps_available",
    "is_cuda_available",
    "auto_select_device",

    # Activation Patching (Phase 2.3)
    "PatchConfig",
    "ActivationPatch",
    "PatchingResult",
    "PatchingSweepResult",
    "PatchingContext",
    "CausalTracer",

    # Modules
    "models",
    "analysis",
    "intervention",
    "capture",
]

