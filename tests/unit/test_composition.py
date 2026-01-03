"""
Head Composition Analysis Unit Tests
=====================================

Tests for head composition analysis methods.
Uses mock fixtures - no GPU required.
"""

import pytest
import torch
from oculi.capture.structures import AttentionCapture, ResidualCapture
from oculi.analysis.composition import CompositionAnalysis, CompositionResult


# =============================================================================
# FIXTURES
# =============================================================================

def mock_attention_capture(
    n_layers: int = 4,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    n_tokens: int = 16,
    head_dim: int = 64
) -> AttentionCapture:
    """Create mock AttentionCapture for testing."""
    torch.manual_seed(42)
    
    # Create attention patterns that sum to 1 (valid probabilities)
    patterns = torch.rand(n_layers, n_heads, n_tokens, n_tokens)
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)
    
    # Create Q/K/V tensors
    queries = torch.randn(n_layers, n_heads, n_tokens, head_dim)
    keys = torch.randn(n_layers, n_kv_heads, n_tokens, head_dim)
    values = torch.randn(n_layers, n_kv_heads, n_tokens, head_dim)
    
    return AttentionCapture(
        queries=queries,
        keys=keys,
        values=values,
        patterns=patterns,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_tokens=n_tokens,
        head_dim=head_dim,
        model_name="test-model",
        qk_stage="post_rope",
        captured_layers=tuple(range(n_layers)),
    )


def mock_residual_capture(
    n_layers: int = 4,
    n_tokens: int = 16,
    hidden_dim: int = 256
) -> ResidualCapture:
    """Create mock ResidualCapture for testing."""
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


# =============================================================================
# QK COMPOSITION TESTS
# =============================================================================

class TestQKComposition:
    """Tests for qk_composition method."""
    
    def test_output_shape(self):
        """Should return [T] scores."""
        capture = mock_attention_capture(n_tokens=16)
        
        result = CompositionAnalysis.qk_composition(
            capture, source=(0, 0), target=(2, 1)
        )
        
        assert result.values.shape == (16,)
    
    def test_method_name(self):
        """Result should have correct method name."""
        capture = mock_attention_capture()
        
        result = CompositionAnalysis.qk_composition(
            capture, source=(0, 0), target=(1, 0)
        )
        
        assert result.method == "qk_composition"
    
    def test_source_must_be_before_target(self):
        """Should raise if source layer >= target layer."""
        capture = mock_attention_capture()
        
        with pytest.raises(ValueError, match="must be before"):
            CompositionAnalysis.qk_composition(
                capture, source=(2, 0), target=(1, 0)
            )
    
    def test_normalized_values(self):
        """With normalize=True, values should be in [0, 1]."""
        capture = mock_attention_capture()
        
        result = CompositionAnalysis.qk_composition(
            capture, source=(0, 0), target=(2, 0), normalize=True
        )
        
        assert (result.values >= 0).all()
        assert (result.values <= 1).all()


# =============================================================================
# OV COMPOSITION TESTS
# =============================================================================

class TestOVComposition:
    """Tests for ov_composition method."""
    
    def test_output_shape(self):
        """Should return [T] scores."""
        capture = mock_attention_capture(n_tokens=16)
        
        result = CompositionAnalysis.ov_composition(
            capture, source=(0, 0), target=(2, 1)
        )
        
        assert result.values.shape == (16,)
    
    def test_requires_values(self):
        """Should raise if values not present."""
        capture = mock_attention_capture()
        capture_no_values = AttentionCapture(
            queries=capture.queries,
            keys=capture.keys,
            values=None,  # No values
            patterns=capture.patterns,
            n_layers=capture.n_layers,
            n_heads=capture.n_heads,
            n_kv_heads=capture.n_kv_heads,
            n_tokens=capture.n_tokens,
            head_dim=capture.head_dim,
            model_name=capture.model_name,
            qk_stage=capture.qk_stage,
            captured_layers=capture.captured_layers,
        )
        
        with pytest.raises(ValueError, match="must have values"):
            CompositionAnalysis.ov_composition(
                capture_no_values, source=(0, 0), target=(1, 0)
            )


# =============================================================================
# VIRTUAL ATTENTION TESTS
# =============================================================================

class TestVirtualAttention:
    """Tests for virtual_attention method."""
    
    def test_output_shape(self):
        """Should return [T, T] matrix."""
        capture = mock_attention_capture(n_tokens=16)
        
        result = CompositionAnalysis.virtual_attention(
            capture, heads=[(0, 0), (1, 0), (2, 0)]
        )
        
        assert result.values.shape == (16, 16)
    
    def test_requires_multiple_heads(self):
        """Should raise if less than 2 heads."""
        capture = mock_attention_capture()
        
        with pytest.raises(ValueError, match="at least 2 heads"):
            CompositionAnalysis.virtual_attention(capture, heads=[(0, 0)])
    
    def test_requires_causal_order(self):
        """Should raise if heads not in causal order."""
        capture = mock_attention_capture()
        
        with pytest.raises(ValueError, match="causal order"):
            CompositionAnalysis.virtual_attention(
                capture, heads=[(2, 0), (1, 0)]
            )
    
    def test_two_heads_composition(self):
        """Two head composition should equal matrix product."""
        capture = mock_attention_capture(n_tokens=8)
        
        result = CompositionAnalysis.virtual_attention(
            capture, heads=[(0, 0), (1, 0)]
        )
        
        # Manual computation
        A0 = capture.patterns[0, 0].float()  # [T, T]
        A1 = capture.patterns[1, 0].float()  # [T, T]
        expected = torch.mm(A1, A0)
        expected = expected / expected.sum(dim=-1, keepdim=True)  # Normalized
        
        torch.testing.assert_close(
            result.values.float(),
            expected.cpu(),
            atol=1e-5,
            rtol=1e-5
        )


# =============================================================================
# PATH PATCHING SCORE TESTS
# =============================================================================

class TestPathPatchingScore:
    """Tests for path_patching_score method."""
    
    def test_returns_scalar(self):
        """Should return single score."""
        capture = mock_attention_capture()
        residual = mock_residual_capture()
        
        result = CompositionAnalysis.path_patching_score(
            capture, residual, path=[(0, 0), (2, 0)]
        )
        
        assert result.values.numel() == 1
    
    def test_metadata_contains_path(self):
        """Metadata should contain path info."""
        capture = mock_attention_capture()
        residual = mock_residual_capture()
        
        path = [(0, 0), (1, 1), (2, 0)]
        result = CompositionAnalysis.path_patching_score(
            capture, residual, path=path
        )
        
        assert result.metadata["path"] == path


# =============================================================================
# COMPOSITION MATRIX TESTS
# =============================================================================

class TestCompositionMatrix:
    """Tests for composition_matrix method."""
    
    def test_output_shape(self):
        """Should be [L*H, L*H]."""
        capture = mock_attention_capture(n_layers=3, n_heads=2)
        
        result = CompositionAnalysis.composition_matrix(capture, method="qk")
        
        assert result.values.shape == (6, 6)  # 3*2 = 6
    
    def test_upper_triangular(self):
        """Only later heads can read from earlier ones."""
        capture = mock_attention_capture(n_layers=4, n_heads=2)
        
        result = CompositionAnalysis.composition_matrix(capture, method="qk")
        
        # Diagonal and below should be zero (can't compose with self or future)
        for i in range(result.values.shape[0]):
            for j in range(i, result.values.shape[1]):
                # Skip blocks where j's layer < i's layer
                i_layer = i // 2
                j_layer = j // 2
                if j_layer >= i_layer:
                    assert result.values[i, j] == 0


# =============================================================================  
# INDUCTION CIRCUIT DETECTION TESTS
# =============================================================================

class TestDetectInductionCircuit:
    """Tests for detect_induction_circuit method."""
    
    def test_returns_result(self):
        """Should return CompositionResult."""
        capture = mock_attention_capture()
        
        result = CompositionAnalysis.detect_induction_circuit(capture)
        
        assert isinstance(result, CompositionResult)
        assert result.method == "detect_induction_circuit"
    
    def test_metadata_contains_circuits(self):
        """Metadata should contain detected circuits."""
        capture = mock_attention_capture()
        
        result = CompositionAnalysis.detect_induction_circuit(capture)
        
        assert "circuits" in result.metadata
        assert isinstance(result.metadata["circuits"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
