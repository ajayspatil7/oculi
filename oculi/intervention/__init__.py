"""
Oculi Intervention Module
===========================

Public API for applying interventions during model inference.

Intervention Semantics (from API_CONTRACT.md):
    - QScaler: Q_new = alpha * Q
    - KScaler: K_new = alpha * K  
    - SpectraScaler: Q_new = √α * Q, K_new = √α * K (net effect: α on logits)
    - HeadAblation: Zero out head output

Metrics:
    - InterventionMetrics: delta_entropy, effect_size, accuracy_delta

Usage:
    >>> from oculi.intervention import SpectraScaler, InterventionContext
    >>> scaler = SpectraScaler(layer=23, head=5, alpha=1.5)
    >>> with InterventionContext(adapter, [scaler]):
    ...     output = adapter.generate(prompt)
"""

from oculi.intervention.scalers import (
    QScaler,
    KScaler,
    SpectraScaler,
)
from oculi.intervention.ablation import HeadAblation
from oculi.intervention.context import InterventionContext
from oculi.intervention.metrics import InterventionMetrics, EffectMeasurement

__all__ = [
    "QScaler",
    "KScaler",
    "SpectraScaler",
    "HeadAblation",
    "InterventionContext",
    "InterventionMetrics",
    "EffectMeasurement",
]
