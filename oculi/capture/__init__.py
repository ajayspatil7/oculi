"""
Oculi Capture Module
====================

Utilities for capturing attention data from transformer models.

Core Classes:
    - AttentionCapture: Immutable container for captured data
    - AttentionStructure: Model attention architecture description
    - CaptureConfig: Configuration for capture operation

Hook Utilities:
    - AttentionHook: Hook registration and management
    
Note: Model adapters now live in oculi.models (explicit, not auto-detected).
"""

from oculi.capture.structures import (
    AttentionCapture,
    AttentionStructure,
    CaptureConfig,
)
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
    "AttentionHook",
    "HookHandle",
    "CapturedData",
    "assemble_capture_tensor",
]
