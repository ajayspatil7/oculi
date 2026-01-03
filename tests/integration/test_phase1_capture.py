"""
Phase 1 Integration Tests
=========================

Integration tests for ResidualCapture, MLPCapture, LogitCapture using MockLlamaAdapter.
Tests actual capture functionality with a tiny model.
"""

import pytest
import sys
sys.path.insert(0, '/Users/ajaysp/oculi')

import torch
from tests.mocks.mock_llama import MockLlamaAdapter, MockLlamaConfig


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def adapter():
    """Create a MockLlamaAdapter for testing."""
    return MockLlamaAdapter()


@pytest.fixture
def sample_input(adapter):
    """Create sample input IDs."""
    return adapter.tokenize("Hello world, this is a test!")


# =============================================================================
# RESIDUAL CAPTURE TESTS
# =============================================================================

class TestResidualCaptureIntegration:
    """Integration tests for residual stream capture."""
    
    def test_capture_residual_basic(self, adapter, sample_input):
        """Basic residual capture should work."""
        from oculi.capture.structures import ResidualConfig
        
        residual = adapter.capture_residual(sample_input)
        
        assert residual is not None
        assert residual.pre_attn is not None
        assert residual.post_attn is not None
        assert residual.pre_mlp is not None
        assert residual.post_mlp is not None
    
    def test_capture_residual_shapes(self, adapter, sample_input):
        """Captured tensors should have correct shapes."""
        residual = adapter.capture_residual(sample_input)
        
        n_layers = adapter.num_layers()
        n_tokens = sample_input.shape[1]
        hidden_dim = adapter._config.hidden_size
        
        assert residual.pre_attn.shape == (n_layers, n_tokens, hidden_dim)
        assert residual.n_layers == n_layers
        assert residual.n_tokens == n_tokens
    
    def test_capture_residual_selective(self, adapter, sample_input):
        """Selective capture should only capture requested positions."""
        from oculi.capture.structures import ResidualConfig
        
        config = ResidualConfig(
            capture_pre_attn=True,
            capture_post_attn=False,
            capture_pre_mlp=False,
            capture_post_mlp=True,
        )
        
        residual = adapter.capture_residual(sample_input, config)
        
        assert residual.pre_attn is not None
        assert residual.post_attn is None
        assert residual.pre_mlp is None
        assert residual.post_mlp is not None
    
    def test_capture_residual_layer_subset(self, adapter, sample_input):
        """Layer subset capture should work."""
        from oculi.capture.structures import ResidualConfig
        
        config = ResidualConfig(layers=(0, 2))
        
        residual = adapter.capture_residual(sample_input, config)
        
        assert residual.n_layers == 2
        assert residual.captured_layers == (0, 2)


# =============================================================================
# MLP CAPTURE TESTS
# =============================================================================

class TestMLPCaptureIntegration:
    """Integration tests for MLP capture."""
    
    def test_capture_mlp_basic(self, adapter, sample_input):
        """Basic MLP capture should work."""
        mlp = adapter.capture_mlp(sample_input)
        
        assert mlp is not None
        assert mlp.pre_activation is not None
        assert mlp.mlp_output is not None
    
    def test_capture_mlp_shapes(self, adapter, sample_input):
        """MLP tensors should have correct shapes."""
        mlp = adapter.capture_mlp(sample_input)
        
        n_layers = adapter.num_layers()
        n_tokens = sample_input.shape[1]
        hidden_dim = adapter._config.hidden_size
        intermediate_dim = adapter._config.intermediate_size
        
        assert mlp.pre_activation.shape == (n_layers, n_tokens, intermediate_dim)
        assert mlp.mlp_output.shape == (n_layers, n_tokens, hidden_dim)


# =============================================================================
# FULL CAPTURE TESTS
# =============================================================================

class TestFullCaptureIntegration:
    """Integration tests for combined capture."""
    
    def test_capture_full_all(self, adapter, sample_input):
        """Full capture with all configs should work."""
        from oculi.capture.structures import CaptureConfig, ResidualConfig, MLPConfig
        
        full = adapter.capture_full(
            sample_input,
            attention_config=CaptureConfig(),
            residual_config=ResidualConfig(),
            mlp_config=MLPConfig(),
        )
        
        assert full.attention is not None
        assert full.residual is not None
        assert full.mlp is not None
    
    def test_capture_full_partial(self, adapter, sample_input):
        """Full capture with some configs None should work."""
        from oculi.capture.structures import ResidualConfig
        
        full = adapter.capture_full(
            sample_input,
            attention_config=None,
            residual_config=ResidualConfig(),
            mlp_config=None,
        )
        
        assert full.attention is None
        assert full.residual is not None
        assert full.mlp is None


# =============================================================================
# CIRCUIT DETECTION TESTS
# =============================================================================

class TestCircuitDetection:
    """Integration tests for circuit detection."""
    
    def test_detect_induction_heads(self, adapter, sample_input):
        """Induction head detection should run without error."""
        from oculi.analysis import CircuitDetection
        
        capture = adapter.capture(sample_input)
        scores = CircuitDetection.detect_induction_heads(capture)
        
        assert scores.shape == (capture.n_layers, capture.n_heads)
        assert (scores >= 0).all()
        assert (scores <= 1).all()
    
    def test_detect_previous_token_heads(self, adapter, sample_input):
        """Previous token head detection should run."""
        from oculi.analysis import CircuitDetection
        
        capture = adapter.capture(sample_input)
        mask = CircuitDetection.detect_previous_token_heads(capture)
        
        assert mask.shape == (capture.n_layers, capture.n_heads)
        assert mask.dtype == torch.bool
    
    def test_attention_entropy(self, adapter, sample_input):
        """Attention entropy should have correct shape."""
        from oculi.analysis import CircuitDetection
        
        capture = adapter.capture(sample_input)
        entropy = CircuitDetection.attention_entropy(capture)
        
        assert entropy.shape == (capture.n_layers, capture.n_heads, capture.n_tokens)
    
    def test_summarize_circuits(self, adapter, sample_input):
        """Circuit summary should return valid dict."""
        from oculi.analysis import CircuitDetection
        
        capture = adapter.capture(sample_input)
        summary = CircuitDetection.summarize_circuits(capture)
        
        assert 'induction_heads' in summary
        assert 'previous_token_heads' in summary
        assert 'total_heads' in summary
        assert summary['total_heads'] == capture.n_layers * capture.n_heads


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
