"""
Phase 1 Shape Contract Tests
============================

Contract tests for ResidualCapture, MLPCapture, LogitCapture, and FullCapture.
Uses mock fixtures - no GPU required.
"""

import pytest
import torch
from oculi.capture.structures import (
    ResidualCapture,
    ResidualConfig,
    MLPCapture,
    MLPConfig,
    LogitCapture,
    LogitConfig,
    FullCapture,
    AttentionCapture,
)


# =============================================================================
# FIXTURES
# =============================================================================

def mock_residual_capture(
    n_layers: int = 4,
    n_tokens: int = 16,
    hidden_dim: int = 256
) -> ResidualCapture:
    """Create a mock ResidualCapture for testing."""
    torch.manual_seed(42)
    return ResidualCapture(
        pre_attn=torch.randn(n_layers, n_tokens, hidden_dim),
        post_attn=torch.randn(n_layers, n_tokens, hidden_dim),
        pre_mlp=torch.randn(n_layers, n_tokens, hidden_dim),
        post_mlp=torch.randn(n_layers, n_tokens, hidden_dim),
        n_layers=n_layers,
        n_tokens=n_tokens,
        hidden_dim=hidden_dim,
        model_name="test-model",
        captured_layers=tuple(range(n_layers)),
    )


def mock_mlp_capture(
    n_layers: int = 4,
    n_tokens: int = 16,
    hidden_dim: int = 256,
    intermediate_dim: int = 512
) -> MLPCapture:
    """Create a mock MLPCapture for testing."""
    torch.manual_seed(42)
    return MLPCapture(
        pre_activation=torch.randn(n_layers, n_tokens, intermediate_dim),
        post_activation=torch.randn(n_layers, n_tokens, intermediate_dim),
        gate_output=torch.randn(n_layers, n_tokens, intermediate_dim),
        mlp_output=torch.randn(n_layers, n_tokens, hidden_dim),
        n_layers=n_layers,
        n_tokens=n_tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        model_name="test-model",
        captured_layers=tuple(range(n_layers)),
    )


def mock_logit_capture(
    n_layers: int = 4,
    n_tokens: int = 16,
    vocab_size: int = 1000
) -> LogitCapture:
    """Create a mock LogitCapture for testing."""
    torch.manual_seed(42)
    return LogitCapture(
        logits=torch.randn(n_layers, n_tokens, vocab_size),
        top_k_logits=None,
        top_k_indices=None,
        n_layers=n_layers,
        n_tokens=n_tokens,
        vocab_size=vocab_size,
        model_name="test-model",
        captured_layers=tuple(range(n_layers)),
    )


# =============================================================================
# RESIDUAL CAPTURE SHAPE TESTS
# =============================================================================

class TestResidualCaptureShapes:
    """Shape contract tests for ResidualCapture."""
    
    def test_residual_shapes_match_metadata(self):
        """All tensors should match n_layers, n_tokens, hidden_dim."""
        capture = mock_residual_capture(n_layers=4, n_tokens=16, hidden_dim=256)
        
        assert capture.pre_attn.shape == (4, 16, 256)
        assert capture.post_attn.shape == (4, 16, 256)
        assert capture.pre_mlp.shape == (4, 16, 256)
        assert capture.post_mlp.shape == (4, 16, 256)
    
    def test_residual_layer_count_matches(self):
        """n_layers should match tensor first dimension."""
        capture = mock_residual_capture(n_layers=8)
        
        assert capture.n_layers == 8
        assert capture.pre_attn.shape[0] == 8
    
    def test_residual_token_dim_matches(self):
        """n_tokens should match tensor second dimension."""
        capture = mock_residual_capture(n_tokens=32)
        
        assert capture.n_tokens == 32
        assert capture.pre_attn.shape[1] == 32
    
    def test_residual_hidden_dim_matches(self):
        """hidden_dim should match tensor third dimension."""
        capture = mock_residual_capture(hidden_dim=512)
        
        assert capture.hidden_dim == 512
        assert capture.pre_attn.shape[2] == 512
    
    def test_captured_layers_tuple(self):
        """captured_layers should be a tuple of ints."""
        capture = mock_residual_capture(n_layers=4)
        
        assert isinstance(capture.captured_layers, tuple)
        assert len(capture.captured_layers) == 4
        assert all(isinstance(x, int) for x in capture.captured_layers)


# =============================================================================
# RESIDUAL CAPTURE HELPER TESTS
# =============================================================================

class TestResidualCaptureHelpers:
    """Test ResidualCapture helper methods."""
    
    def test_get_layer_valid(self):
        """get_layer should return correct tensors."""
        capture = mock_residual_capture(n_layers=4)
        
        layer_data = capture.get_layer(2)
        
        assert 'pre_attn' in layer_data
        assert layer_data['pre_attn'].shape == (16, 256)
    
    def test_get_layer_invalid(self):
        """get_layer should raise for invalid layer."""
        capture = mock_residual_capture(n_layers=4)
        
        with pytest.raises(ValueError, match="not captured"):
            capture.get_layer(10)
    
    def test_stream_at(self):
        """stream_at should return correct tensor."""
        capture = mock_residual_capture(n_layers=4)
        
        tensor = capture.stream_at('pre_attn', 2)
        
        assert tensor.shape == (16, 256)
    
    def test_memory_usage(self):
        """memory_usage should return positive integer."""
        capture = mock_residual_capture()
        
        usage = capture.memory_usage()
        
        assert usage > 0
        # 4 tensors * 4 * 16 * 256 * 4 bytes = 262144
        expected = 4 * 4 * 16 * 256 * 4
        assert usage == expected


# =============================================================================
# MLP CAPTURE SHAPE TESTS
# =============================================================================

class TestMLPCaptureShapes:
    """Shape contract tests for MLPCapture."""
    
    def test_mlp_shapes_match_metadata(self):
        """Tensor shapes should match metadata."""
        capture = mock_mlp_capture(
            n_layers=4, n_tokens=16, hidden_dim=256, intermediate_dim=512
        )
        
        assert capture.pre_activation.shape == (4, 16, 512)
        assert capture.post_activation.shape == (4, 16, 512)
        assert capture.gate_output.shape == (4, 16, 512)
        assert capture.mlp_output.shape == (4, 16, 256)
    
    def test_intermediate_dim_correct(self):
        """Intermediate tensors should use intermediate_dim."""
        capture = mock_mlp_capture(intermediate_dim=1024)
        
        assert capture.intermediate_dim == 1024
        assert capture.pre_activation.shape[2] == 1024


# =============================================================================
# LOGIT CAPTURE SHAPE TESTS
# =============================================================================

class TestLogitCaptureShapes:
    """Shape contract tests for LogitCapture."""
    
    def test_logit_shapes_match_metadata(self):
        """Logits shape should be [L, T, V]."""
        capture = mock_logit_capture(n_layers=4, n_tokens=16, vocab_size=1000)
        
        assert capture.logits.shape == (4, 16, 1000)
    
    def test_top_k_shapes(self):
        """Top-k tensors should be [L, T, K]."""
        top_k = 10
        capture = LogitCapture(
            logits=None,
            top_k_logits=torch.randn(4, 16, top_k),
            top_k_indices=torch.randint(0, 1000, (4, 16, top_k)),
            n_layers=4,
            n_tokens=16,
            vocab_size=1000,
            model_name="test",
            captured_layers=tuple(range(4)),
        )
        
        assert capture.top_k_logits.shape == (4, 16, 10)
        assert capture.top_k_indices.shape == (4, 16, 10)


# =============================================================================
# CONFIG VALIDATION TESTS
# =============================================================================

class TestConfigValidation:
    """Test config validation methods."""
    
    def test_residual_config_valid(self):
        """Valid ResidualConfig should not raise."""
        config = ResidualConfig(layers=(0, 1, 2))
        config.validate(num_layers=4)  # Should not raise
    
    def test_residual_config_invalid_layer(self):
        """Invalid layer should raise ValueError."""
        config = ResidualConfig(layers=(10,))
        
        with pytest.raises(ValueError, match="out of range"):
            config.validate(num_layers=4)
    
    def test_residual_config_no_captures(self):
        """No captures enabled should raise ValueError."""
        config = ResidualConfig(
            capture_pre_attn=False,
            capture_post_attn=False,
            capture_pre_mlp=False,
            capture_post_mlp=False,
        )
        
        with pytest.raises(ValueError, match="At least one"):
            config.validate(num_layers=4)
    
    def test_mlp_config_valid(self):
        """Valid MLPConfig should not raise."""
        config = MLPConfig(layers=(0, 1))
        config.validate(num_layers=4)
    
    def test_logit_config_valid(self):
        """Valid LogitConfig should not raise."""
        config = LogitConfig(layers=(0, 1), top_k=10)
        config.validate(num_layers=4)
    
    def test_logit_config_invalid_top_k(self):
        """Invalid top_k should raise ValueError."""
        config = LogitConfig(top_k=-1)
        
        with pytest.raises(ValueError, match="top_k must be positive"):
            config.validate(num_layers=4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
