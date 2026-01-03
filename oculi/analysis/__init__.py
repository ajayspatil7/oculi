"""
Oculi Analysis Module
=======================

Public API for analyzing captured attention data.

All analysis functions follow the contract:
    f(AttentionCapture, **params) -> torch.Tensor

Pure functions:
    - No side effects
    - No model access
    - No plotting (see oculi.visualize)
    - Deterministic output

Classes
-------
- NormAnalysis: Q/K/V vector norm computations
- EntropyAnalysis: Attention entropy metrics
- AttentionAnalysis: Pattern-based metrics
- CorrelationAnalysis: Statistical analysis with p-values
- StratifiedView: Layer/head/token slicing helpers
- LogitLensAnalysis: Layer-by-layer prediction analysis
- CircuitDetection: Canonical circuit pattern detection
"""

from oculi.analysis.norms import NormAnalysis
from oculi.analysis.entropy import EntropyAnalysis
from oculi.analysis.attention import AttentionAnalysis
from oculi.analysis.correlation import CorrelationAnalysis
from oculi.analysis.stratified import (
    StratifiedView,
    StratifiedResult,
    find_extreme_heads,
)
from oculi.analysis.logit_lens import LogitLensAnalysis
from oculi.analysis.circuits import CircuitDetection

__all__ = [
    "NormAnalysis",
    "EntropyAnalysis",
    "AttentionAnalysis",
    "CorrelationAnalysis",
    "StratifiedView",
    "StratifiedResult",
    "find_extreme_heads",
    "LogitLensAnalysis",
    "CircuitDetection",
]

