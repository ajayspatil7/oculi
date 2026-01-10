"""
Device Utilities
=================

Cross-platform device detection and selection for CUDA, MPS (Apple Silicon), and CPU.

This module provides utilities to automatically detect and select the best available
device for running models, with support for:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon - M1/M2/M3/M4)
- CPU (fallback)

Usage:
    >>> from oculi.utils import get_default_device, get_device_info
    >>> device = get_default_device()
    >>> print(f"Using device: {device}")
    >>> info = get_device_info()
    >>> print(f"Device info: {info}")
"""

from dataclasses import dataclass
from typing import Optional, Literal
import torch
import warnings


DeviceType = Literal["cuda", "mps", "cpu"]


@dataclass
class DeviceInfo:
    """
    Information about available compute devices.

    Attributes:
        device_type: Type of device ("cuda", "mps", or "cpu")
        device: PyTorch device object
        device_name: Human-readable device name
        has_cuda: Whether CUDA is available
        has_mps: Whether MPS is available
        cuda_device_count: Number of CUDA devices
        pytorch_version: PyTorch version string
    """
    device_type: DeviceType
    device: torch.device
    device_name: str
    has_cuda: bool
    has_mps: bool
    cuda_device_count: int
    pytorch_version: str

    def __repr__(self) -> str:
        return (
            f"DeviceInfo(device_type='{self.device_type}', "
            f"device={self.device}, "
            f"device_name='{self.device_name}')"
        )


def is_cuda_available() -> bool:
    """
    Check if CUDA is available.

    Returns:
        True if CUDA-capable GPU is available

    Example:
        >>> if is_cuda_available():
        ...     print("CUDA available!")
    """
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available.

    MPS provides GPU acceleration on Apple Silicon (M1/M2/M3/M4).
    Requires PyTorch 1.12+ and macOS 12.3+.

    Returns:
        True if MPS is available and functional

    Example:
        >>> if is_mps_available():
        ...     print("Running on Apple Silicon with MPS!")
    """
    # Check if MPS is available (requires PyTorch 1.12+)
    if not hasattr(torch.backends, 'mps'):
        return False

    if not torch.backends.mps.is_available():
        return False

    # Additional check: MPS must be built
    try:
        if not torch.backends.mps.is_built():
            return False
    except AttributeError:
        # Older PyTorch versions don't have is_built()
        pass

    return True


def get_device_name(device: torch.device) -> str:
    """
    Get human-readable device name.

    Args:
        device: PyTorch device

    Returns:
        Device name string

    Example:
        >>> device = torch.device("cuda:0")
        >>> print(get_device_name(device))
        'NVIDIA GeForce RTX 3090'
    """
    if device.type == "cuda":
        try:
            return torch.cuda.get_device_name(device)
        except (RuntimeError, AssertionError):
            return "CUDA Device"
    elif device.type == "mps":
        return "Apple Silicon (MPS)"
    else:
        return "CPU"


def auto_select_device(prefer_cuda: bool = True, verbose: bool = False) -> torch.device:
    """
    Automatically select the best available device.

    Selection priority:
    1. CUDA (if prefer_cuda=True and available)
    2. MPS (if available on Apple Silicon)
    3. CPU (fallback)

    Args:
        prefer_cuda: Prefer CUDA over MPS if both available (default: True)
        verbose: Print device selection reasoning (default: False)

    Returns:
        Selected PyTorch device

    Example:
        >>> device = auto_select_device(verbose=True)
        Selected device: cuda:0 (NVIDIA GeForce RTX 3090)
        >>> device = auto_select_device(prefer_cuda=False)  # Prefer MPS on Mac
    """
    has_cuda = is_cuda_available()
    has_mps = is_mps_available()

    if prefer_cuda and has_cuda:
        device = torch.device("cuda")
        if verbose:
            print(f"Selected device: cuda:0 ({get_device_name(device)})")
    elif has_mps:
        device = torch.device("mps")
        if verbose:
            print(f"Selected device: mps ({get_device_name(device)})")

        # Warn about MPS limitations
        if verbose:
            warnings.warn(
                "Using MPS (Apple Silicon). Note: Some operations may fall back to CPU. "
                "Set PYTORCH_ENABLE_MPS_FALLBACK=1 to enable automatic fallback.",
                UserWarning
            )
    elif has_cuda:
        device = torch.device("cuda")
        if verbose:
            print(f"Selected device: cuda:0 ({get_device_name(device)})")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Selected device: cpu (no GPU acceleration available)")

    return device


def get_default_device() -> torch.device:
    """
    Get the default device for Oculi operations.

    This is the recommended way to select a device. It automatically
    detects and uses the best available hardware.

    Priority: CUDA > MPS > CPU

    Returns:
        Default PyTorch device

    Example:
        >>> from oculi.utils import get_default_device
        >>> device = get_default_device()
        >>> model = model.to(device)
    """
    return auto_select_device(prefer_cuda=True, verbose=False)


def get_device_info() -> DeviceInfo:
    """
    Get comprehensive information about available devices.

    Returns:
        DeviceInfo object with device details

    Example:
        >>> from oculi.utils import get_device_info
        >>> info = get_device_info()
        >>> print(f"Running on: {info.device_name}")
        >>> print(f"Device type: {info.device_type}")
        >>> print(f"PyTorch version: {info.pytorch_version}")
    """
    device = get_default_device()
    has_cuda = is_cuda_available()
    has_mps = is_mps_available()

    # Determine device type
    device_type: DeviceType
    if device.type == "cuda":
        device_type = "cuda"
    elif device.type == "mps":
        device_type = "mps"
    else:
        device_type = "cpu"

    return DeviceInfo(
        device_type=device_type,
        device=device,
        device_name=get_device_name(device),
        has_cuda=has_cuda,
        has_mps=has_mps,
        cuda_device_count=torch.cuda.device_count() if has_cuda else 0,
        pytorch_version=torch.__version__,
    )


def validate_device_compatibility(device: torch.device, operation: str = "operation") -> None:
    """
    Validate that a device supports required operations.

    Some operations may not be supported on all devices (especially MPS).
    This function can be used to validate device compatibility before
    running expensive operations.

    Args:
        device: Device to validate
        operation: Name of operation for error message

    Raises:
        RuntimeError: If device doesn't support the operation

    Example:
        >>> device = torch.device("mps")
        >>> validate_device_compatibility(device, "attention computation")
    """
    if device.type == "mps":
        # Check PyTorch version for MPS
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version < (2, 0):
            raise RuntimeError(
                f"MPS support for {operation} requires PyTorch 2.0+. "
                f"Current version: {torch.__version__}"
            )

        # Warn about potential fallback
        warnings.warn(
            f"Running {operation} on MPS. Some operations may automatically "
            f"fall back to CPU. Set PYTORCH_ENABLE_MPS_FALLBACK=1 environment "
            f"variable to enable automatic fallback.",
            UserWarning,
            stacklevel=2
        )


def recommend_device_for_model(model_size_gb: float, verbose: bool = False) -> torch.device:
    """
    Recommend device based on model size and available memory.

    Args:
        model_size_gb: Approximate model size in GB
        verbose: Print recommendation reasoning

    Returns:
        Recommended device

    Example:
        >>> # For LLaMA-3-8B (~16GB)
        >>> device = recommend_device_for_model(16.0, verbose=True)
    """
    device = auto_select_device(verbose=False)

    if device.type == "cuda":
        try:
            # Check CUDA memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            if verbose:
                print(f"CUDA device has {total_memory:.1f}GB memory")

            if model_size_gb * 1.5 > total_memory:  # 1.5x for activations
                if verbose:
                    print(f"Warning: Model ({model_size_gb:.1f}GB) may not fit. "
                          f"Consider using CPU or quantization.")
        except Exception:
            pass

    elif device.type == "mps":
        # MPS shares memory with system
        if verbose:
            print("Using MPS (Apple Silicon). Memory is shared with system.")
            if model_size_gb > 8:
                print(f"Warning: Large model ({model_size_gb:.1f}GB). "
                      f"Monitor system memory usage.")

    return device
