"""
MATS 10.0: Sycophancy Entropy Control
======================================

Core module for investigating attention entropy as causal bottleneck for sycophancy.

Hypothesis: When rationalizing, Logic Heads blur (high entropy) while 
Sycophancy Heads sharpen (low entropy).
"""

from .model import load_model, get_model_info
from .hooks import get_spectra_hook, add_scaling_hooks, reset_hooks
from .entropy import calculate_entropy, compute_delta_entropy
from .registry import experiment, get_experiment, list_experiments
from .utils import set_seed, load_config, save_results

__version__ = "0.1.0"
__all__ = [
    "load_model",
    "get_model_info",
    "get_spectra_hook", 
    "add_scaling_hooks",
    "reset_hooks",
    "calculate_entropy",
    "compute_delta_entropy",
    "experiment",
    "get_experiment",
    "list_experiments",
    "set_seed",
    "load_config",
    "save_results",
]
