"""
Experiments Module for MATS 10.0
=================================

Contains all experiment implementations for sycophancy entropy analysis.

Experiments:
- sanity: Phase 1 critical gate (ΔEntropy detection)
- rationalization: EXP1 - Identify Logic/Sycophancy heads
- restoration: EXP2 - Logic head sharpening intervention
- jamming: EXP3 - Sycophancy head flattening
- control: EXP4 - Baseline preservation check
"""

from .sanity import run_sanity_check
from .rationalization import run_rationalization_profile
from .restoration import run_restoration_experiment
from .jamming import run_jamming_experiment
from .control import run_control_experiment

__all__ = [
    "run_sanity_check",
    "run_rationalization_profile",
    "run_restoration_experiment",
    "run_jamming_experiment",
    "run_control_experiment",
]
