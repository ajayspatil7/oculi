"""
Oculi Intervention Module
===========================

Public API for applying interventions during model inference.

Intervention Semantics (from API_CONTRACT.md):
    - QScaler: Q_new = alpha * Q
    - KScaler: K_new = alpha * K  
    - SpectraScaler: Q_new = √α * Q, K_new = √α * K (net effect: α on logits)
    - HeadAblation: Zero out head output

Activation Patching (Phase 2.3):
    - ActivationPatch: Replace activations from source run
    - PatchingContext: Context manager for applying patches
    - CausalTracer: Systematic patching experiments

Metrics:
    - InterventionMetrics: delta_entropy, effect_size, accuracy_delta

Usage:
    >>> from oculi.intervention import SpectraScaler, InterventionContext
    >>> scaler = SpectraScaler(layer=23, head=5, alpha=1.5)
    >>> with InterventionContext(adapter, [scaler]):
    ...     output = adapter.generate(prompt)
    
    >>> from oculi.intervention import CausalTracer
    >>> tracer = CausalTracer(adapter)
    >>> results = tracer.trace(clean_ids, corrupt_ids, metric_fn)
"""

from oculi.intervention.scalers import (
    QScaler,
    KScaler,
    SpectraScaler,
)
from oculi.intervention.ablation import HeadAblation
from oculi.intervention.context import InterventionContext
from oculi.intervention.metrics import InterventionMetrics, EffectMeasurement

# Phase 2.3: Activation Patching
from oculi.intervention.patching import (
    PatchConfig,
    ActivationPatch,
    PatchingResult,
    PatchingSweepResult,
    VALID_COMPONENTS,
)
from oculi.intervention.experiments import (
    PatchingContext,
    patching_context,
    CausalTracer,
)

__all__ = [
    # Scalers
    "QScaler",
    "KScaler",
    "SpectraScaler",
    # Ablation
    "HeadAblation",
    # Context managers
    "InterventionContext",
    "PatchingContext",
    "patching_context",
    # Metrics
    "InterventionMetrics",
    "EffectMeasurement",
    # Patching (Phase 2.3)
    "PatchConfig",
    "ActivationPatch",
    "PatchingResult",
    "PatchingSweepResult",
    "VALID_COMPONENTS",
    "CausalTracer",
]

