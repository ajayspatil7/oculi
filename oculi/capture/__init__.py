"""
Oculi Capture Module
======================

Public API for capturing attention data from transformer models.

Classes
-------
- AttentionCapture: Immutable container for captured data
- AttentionStructure: Model attention architecture description
- CaptureConfig: Configuration for capture operation
- ModelAdapter: Abstract model interface
- AttentionHook: Hook registration and management

Functions
---------
- load: Load model with auto-detected adapter
"""

from oculi.capture.structures import (
    AttentionCapture,
    AttentionStructure,
    CaptureConfig,
)
from oculi.capture.adapter import ModelAdapter
from oculi.capture.loader import load
from oculi.capture.hooks import (
    AttentionHook,
    HookHandle,
    CapturedData,
    assemble_capture_tensor,
)

__all__ = [
    "AttentionCapture",
    "AttentionStructure",
    "CaptureConfig",
    "ModelAdapter",
    "load",
    "AttentionHook",
    "HookHandle",
    "CapturedData",
    "assemble_capture_tensor",
]
