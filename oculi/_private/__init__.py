"""
Spectra Private Layer
=====================

INTERNAL IMPLEMENTATION DETAILS — NOT PART OF PUBLIC API

This package contains:
- Model-specific adapter implementations
- PyTorch hook machinery
- Memory optimization
- Internal validation

DO NOT IMPORT FROM THIS PACKAGE DIRECTLY.
Use the public API in spectra/ instead.

Changes to this package do NOT trigger version bumps
(unless they affect public API behavior).
"""

# Note: adapters have been moved to oculi.models
# Hooks are in oculi._private.hooks

__all__ = []  # Nothing exported publicly

