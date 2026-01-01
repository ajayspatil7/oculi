"""
Private Hooks Module
====================

PyTorch hook machinery for capture and intervention.
"""

from oculi._private.hooks.intervention import (
    create_intervention_hook,
    create_spectra_hook,
)

__all__ = [
    "create_intervention_hook",
    "create_spectra_hook",
]
