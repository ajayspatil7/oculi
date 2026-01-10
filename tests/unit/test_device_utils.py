"""
Test Device Utilities
======================

Tests for cross-platform device detection (CUDA/MPS/CPU).
"""

import pytest
import torch
from oculi.utils.device import (
    is_cuda_available,
    is_mps_available,
    get_default_device,
    get_device_info,
    auto_select_device,
    get_device_name,
    DeviceInfo,
)


class TestDeviceDetection:
    """Test device detection functions."""

    def test_is_cuda_available(self):
        """Test CUDA availability check."""
        result = is_cuda_available()
        assert isinstance(result, bool)
        # Should match torch.cuda.is_available()
        assert result == torch.cuda.is_available()

    def test_is_mps_available(self):
        """Test MPS availability check."""
        result = is_mps_available()
        assert isinstance(result, bool)

        # If torch has mps backend, should return correct value
        if hasattr(torch.backends, 'mps'):
            assert result == torch.backends.mps.is_available()

    def test_get_default_device(self):
        """Test default device selection."""
        device = get_default_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]

    def test_auto_select_device_prefer_cuda(self):
        """Test device selection with CUDA preference."""
        device = auto_select_device(prefer_cuda=True)
        assert isinstance(device, torch.device)

        # If CUDA available, should prefer it
        if torch.cuda.is_available():
            assert device.type == "cuda"

    def test_auto_select_device_prefer_mps(self):
        """Test device selection preferring MPS over CUDA."""
        device = auto_select_device(prefer_cuda=False)
        assert isinstance(device, torch.device)

        # If MPS available, should use it when prefer_cuda=False
        if is_mps_available():
            assert device.type == "mps"

    def test_get_device_name(self):
        """Test device name retrieval."""
        # Test CPU
        cpu_device = torch.device("cpu")
        name = get_device_name(cpu_device)
        assert name == "CPU"
        assert isinstance(name, str)

        # Test CUDA if available
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda:0")
            name = get_device_name(cuda_device)
            assert isinstance(name, str)
            assert len(name) > 0

        # Test MPS if available
        if is_mps_available():
            mps_device = torch.device("mps")
            name = get_device_name(mps_device)
            assert name == "Apple Silicon (MPS)"

    def test_get_device_info(self):
        """Test comprehensive device info."""
        info = get_device_info()
        assert isinstance(info, DeviceInfo)
        assert info.device_type in ["cuda", "mps", "cpu"]
        assert isinstance(info.device, torch.device)
        assert isinstance(info.device_name, str)
        assert isinstance(info.has_cuda, bool)
        assert isinstance(info.has_mps, bool)
        assert isinstance(info.cuda_device_count, int)
        assert isinstance(info.pytorch_version, str)

        # Validate consistency
        assert info.has_cuda == torch.cuda.is_available()
        if hasattr(torch.backends, 'mps'):
            assert info.has_mps == is_mps_available()

    def test_device_info_repr(self):
        """Test DeviceInfo string representation."""
        info = get_device_info()
        repr_str = repr(info)
        assert "DeviceInfo" in repr_str
        assert info.device_type in repr_str

    def test_device_consistency(self):
        """Test that device detection is consistent."""
        device1 = get_default_device()
        device2 = get_default_device()
        assert device1.type == device2.type


class TestDeviceCompatibility:
    """Test device compatibility across operations."""

    def test_tensor_creation_on_default_device(self):
        """Test that we can create tensors on the default device."""
        device = get_default_device()

        # Create tensor on device
        tensor = torch.randn(10, 10, device=device)
        assert tensor.device.type == device.type

    def test_tensor_operations_on_device(self):
        """Test basic tensor operations work on selected device."""
        device = get_default_device()

        a = torch.randn(5, 5, device=device)
        b = torch.randn(5, 5, device=device)

        # Basic operations should work
        c = a + b
        d = torch.matmul(a, b)
        e = torch.softmax(a, dim=-1)

        assert c.device.type == device.type
        assert d.device.type == device.type
        assert e.device.type == device.type

    def test_cpu_fallback(self):
        """Test that CPU always works as fallback."""
        cpu_device = torch.device("cpu")

        # Should always be able to create tensors on CPU
        tensor = torch.randn(10, 10, device=cpu_device)
        assert tensor.device.type == "cpu"

        # Operations should work
        result = torch.matmul(tensor, tensor)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(not is_mps_available(), reason="MPS not available")
    def test_mps_basic_operations(self):
        """Test basic operations on MPS if available."""
        mps_device = torch.device("mps")

        # Create tensors
        a = torch.randn(10, 10, device=mps_device)
        b = torch.randn(10, 10, device=mps_device)

        # Basic operations
        c = a + b
        d = torch.matmul(a, b)
        e = torch.softmax(a, dim=-1)

        assert c.device.type == "mps"
        assert d.device.type == "mps"
        assert e.device.type == "mps"

        # Transfer to CPU should work
        c_cpu = c.cpu()
        assert c_cpu.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_basic_operations(self):
        """Test basic operations on CUDA if available."""
        cuda_device = torch.device("cuda")

        # Create tensors
        a = torch.randn(10, 10, device=cuda_device)
        b = torch.randn(10, 10, device=cuda_device)

        # Basic operations
        c = a + b
        d = torch.matmul(a, b)
        e = torch.softmax(a, dim=-1)

        assert c.device.type == "cuda"
        assert d.device.type == "cuda"
        assert e.device.type == "cuda"

        # Transfer to CPU should work
        c_cpu = c.cpu()
        assert c_cpu.device.type == "cpu"


class TestDeviceFromAdapter:
    """Test device usage in adapter context."""

    def test_adapter_device_detection(self):
        """Test that adapters can use device detection."""
        from tests.mocks import MockLlamaAdapter

        device = get_default_device()
        adapter = MockLlamaAdapter(device=str(device))

        # Adapter should report correct device
        assert adapter.device.type == device.type

    def test_adapter_cpu_mode(self):
        """Test adapter works on CPU."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter(device="cpu")
        assert adapter.device.type == "cpu"

        # Test capture works
        input_ids = adapter.tokenize("Test input")
        capture = adapter.capture(input_ids)

        assert capture.queries is not None
        assert capture.patterns is not None

    @pytest.mark.skipif(not is_mps_available(), reason="MPS not available")
    def test_adapter_mps_mode(self):
        """Test adapter works on MPS."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter(device="mps")
        assert adapter.device.type == "mps"

        # Test capture works
        input_ids = adapter.tokenize("Test input")
        capture = adapter.capture(input_ids)

        assert capture.queries is not None
        assert capture.patterns is not None
