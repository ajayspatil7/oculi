"""
Oculi Utilities
================

Utility functions and helpers for Oculi.
"""

from oculi.utils.device import (
    get_default_device,
    get_device_info,
    is_mps_available,
    is_cuda_available,
    auto_select_device,
    DeviceInfo,
)

__all__ = [
    "get_default_device",
    "get_device_info",
    "is_mps_available",
    "is_cuda_available",
    "auto_select_device",
    "DeviceInfo",
]
