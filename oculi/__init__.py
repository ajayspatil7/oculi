"""
Oculi: Mechanistic Interpretability Toolkit
============================================

A low-level, surgical instrumentation layer for LLMs.

Public API Exports
------------------
- load: Load a model with auto-detected adapter
- ModelAdapter: Abstract model interface
- AttentionCapture: Captured attention data container
- CaptureConfig: Capture configuration

Modules
-------
- oculi.capture: Capture system
- oculi.analysis: Analysis functions
- oculi.intervention: Intervention classes
- oculi.visualize: Visualization utilities

Example
-------
>>> import oculi
>>> model = oculi.load("meta-llama/Meta-Llama-3-8B")
>>> capture = model.capture(input_ids)
>>> entropy = oculi.analysis.EntropyAnalysis.token_entropy(capture)

Version: 0.1.0-dev
"""

__version__ = "0.1.0-dev"
__author__ = "Ajay S Patil"

# =============================================================================
# PUBLIC API — These are the ONLY exports
# =============================================================================

# Core data structures
from oculi.capture.structures import (
    AttentionCapture,
    AttentionStructure,
    CaptureConfig,
)

# Model adapter interface
from oculi.capture.adapter import ModelAdapter

# Convenience loader
from oculi.capture.loader import load

# Analysis module (import as namespace)
from oculi import analysis

# Intervention module (import as namespace)
from oculi import intervention

# Visualization module (import as namespace)
from oculi import visualize

# =============================================================================
# PUBLIC API LIST — For documentation and linting
# =============================================================================

__all__ = [
    # Version
    "__version__",
    
    # Core structures
    "AttentionCapture",
    "AttentionStructure",
    "CaptureConfig",
    
    # Model interface
    "ModelAdapter",
    
    # Loader
    "load",
    
    # Modules (as namespaces)
    "analysis",
    "intervention",
    "visualize",
]

# =============================================================================
# INTERNAL — Do not import from _private directly
# =============================================================================
# The _private package contains implementation details:
# - _private.adapters: Model-specific adapter implementations
# - _private.hooks: PyTorch hook machinery
# - _private.cache: Memory optimization
# - _private.validation: Internal checks
#
# These may change without notice. Use public API only.
