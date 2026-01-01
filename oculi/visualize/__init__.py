"""
Spectra Visualization Module
============================

Research-quality plotting utilities.

Design Contract:
    - All functions return matplotlib.Figure objects
    - User controls saving/display
    - No side effects (no plt.show(), no file I/O)
    - Uses consistent Spectra color scheme

Usage:
    >>> from oculi.visualize import EntropyPlots
    >>> fig = EntropyPlots.heatmap(mean_entropy)
    >>> fig.savefig("entropy_heatmap.png", dpi=150)
"""

from oculi.visualize.entropy import EntropyPlots
from oculi.visualize.intervention import InterventionPlots
from oculi.visualize.correlation import CorrelationPlots

__all__ = [
    "EntropyPlots",
    "InterventionPlots",
    "CorrelationPlots",
]
