"""
Attribution Methods Unit Tests
==============================

Tests for attribution analysis methods.
Uses mock fixtures - no GPU required.
"""

import pytest
import torch
from oculi.capture.structures import (
    AttentionCapture,
    ResidualCapture,
    MLPCapture,
)
from oculi.analysis.attribution import AttributionMethods, AttributionResult


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


def mock_unembed_matrix(vocab_size: int = 1000, hidden_dim: int = 256):
    """Create mock unembedding matrix."""
    torch.manual_seed(42)
    return torch.randn(vocab_size, hidden_dim)


# =============================================================================
# ATTENTION FLOW TESTS
# =============================================================================

class TestAttentionFlow:
    """Tests for attention_flow method."""
    
    def test_output_shape(self):
        """Flow should be [L, H, T, T]."""
        capture = mock_attention_capture(n_layers=4, n_heads=4, n_tokens=16)
        
        result = AttributionMethods.attention_flow(capture)
        
        assert result.values.shape == (4, 4, 16, 16)
    
    def test_method_name(self):
        """Result should have correct method name."""
        capture = mock_attention_capture()
        
        result = AttributionMethods.attention_flow(capture)
        
        assert result.method == "attention_flow"
    
    def test_identity_attention_layer0(self):
        """Layer 0 flow should equal attention pattern."""
        capture = mock_attention_capture()
        
        result = AttributionMethods.attention_flow(capture, normalize=False)
        
        torch.testing.assert_close(
            result.values[0].float(),
            capture.patterns[0].float(),
            atol=1e-5,
            rtol=1e-5
        )
    
    def test_flow_is_normalized(self):
        """With normalize=True, flow rows should sum to ~1."""
        capture = mock_attention_capture()
        
        result = AttributionMethods.attention_flow(capture, normalize=True)
        
        row_sums = result.values.sum(dim=-1)
        # All row sums should be approximately 1
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    
    def test_requires_patterns(self):
        """Should raise if patterns not present."""
        capture = mock_attention_capture()
        capture = AttentionCapture(
            queries=capture.queries,
            keys=capture.keys,
            values=capture.values,
            patterns=None,  # No patterns
            n_layers=capture.n_layers,
            n_heads=capture.n_heads,
            n_kv_heads=capture.n_kv_heads,
            n_tokens=capture.n_tokens,
            head_dim=capture.head_dim,
            model_name=capture.model_name,
            qk_stage=capture.qk_stage,
            captured_layers=capture.captured_layers,
        )
        
        with pytest.raises(ValueError, match="must have patterns"):
            AttributionMethods.attention_flow(capture)


# =============================================================================
# VALUE-WEIGHTED ATTENTION TESTS
# =============================================================================

class TestValueWeightedAttention:
    """Tests for value_weighted_attention method."""
    
    def test_output_shape(self):
        """Should match attention pattern shape."""
        capture = mock_attention_capture(n_layers=4, n_heads=4, n_tokens=16)
        
        result = AttributionMethods.value_weighted_attention(capture)
        
        assert result.values.shape == (4, 4, 16, 16)
    
    def test_weights_sum_to_one(self):
        """Weighted attention should sum to 1 per query."""
        capture = mock_attention_capture()
        
        result = AttributionMethods.value_weighted_attention(capture)
        
        row_sums = result.values.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    
    def test_zero_values_zero_weight(self):
        """Positions with zero value vectors get zero weight after normalization."""
        capture = mock_attention_capture(n_tokens=8)
        
        # Zero out values at position 3
        values_zeroed = capture.values.clone()
        values_zeroed[:, :, 3, :] = 0
        
        capture_mod = AttentionCapture(
            queries=capture.queries,
            keys=capture.keys,
            values=values_zeroed,
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
        
        result = AttributionMethods.value_weighted_attention(capture_mod)
        
        # Position 3 should have ~0 weight (after normalization)
        pos3_weight = result.values[:, :, :, 3].mean()
        assert pos3_weight < 0.001
    
    def test_different_norm_types(self):
        """Should work with different norm types."""
        capture = mock_attention_capture()
        
        for norm_type in ["l1", "l2", "linf"]:
            result = AttributionMethods.value_weighted_attention(capture, norm_type=norm_type)
            assert result.values.shape == capture.patterns.shape
            assert result.metadata["norm_type"] == norm_type


# =============================================================================
# DIRECT LOGIT ATTRIBUTION TESTS
# =============================================================================

class TestDirectLogitAttribution:
    """Tests for direct_logit_attribution method."""
    
    def test_output_shape(self):
        """Should be [L] for layers."""
        residual = mock_residual_capture(n_layers=4)
        unembed = mock_unembed_matrix(hidden_dim=residual.hidden_dim)
        
        result = AttributionMethods.direct_logit_attribution(
            residual, unembed, target_token_id=42
        )
        
        assert result.values.shape == (4,)
    
    def test_position_indexing(self):
        """position=-1 should give last token's attribution."""
        residual = mock_residual_capture(n_tokens=16)
        unembed = mock_unembed_matrix(hidden_dim=residual.hidden_dim)
        
        result_neg = AttributionMethods.direct_logit_attribution(
            residual, unembed, target_token_id=42, position=-1
        )
        result_pos = AttributionMethods.direct_logit_attribution(
            residual, unembed, target_token_id=42, position=15
        )
        
        torch.testing.assert_close(result_neg.values, result_pos.values)
    
    def test_method_name_and_target(self):
        """Result should have correct metadata."""
        residual = mock_residual_capture()
        unembed = mock_unembed_matrix(hidden_dim=residual.hidden_dim)
        
        result = AttributionMethods.direct_logit_attribution(
            residual, unembed, target_token_id=123
        )
        
        assert result.method == "direct_logit_attribution"
        assert result.target == "token_123"
        assert result.metadata["target_token_id"] == 123
    
    def test_unembed_shape_handling(self):
        """Should handle both [V, H] and [H, V] shaped unembed."""
        residual = mock_residual_capture()
        unembed_vh = mock_unembed_matrix(hidden_dim=residual.hidden_dim)  # [V, H]
        unembed_hv = unembed_vh.T  # [H, V]
        
        result_vh = AttributionMethods.direct_logit_attribution(
            residual, unembed_vh, target_token_id=42
        )
        result_hv = AttributionMethods.direct_logit_attribution(
            residual, unembed_hv, target_token_id=42
        )
        
        torch.testing.assert_close(result_vh.values, result_hv.values)


# =============================================================================
# COMPONENT ATTRIBUTION TESTS
# =============================================================================

class TestComponentAttribution:
    """Tests for component_attribution method."""
    
    def test_output_shape(self):
        """Should be [L, 2]."""
        residual = mock_residual_capture(n_layers=4)
        unembed = mock_unembed_matrix(hidden_dim=residual.hidden_dim)
        
        result = AttributionMethods.component_attribution(
            residual, None, unembed, target_token_id=42
        )
        
        assert result.values.shape == (4, 2)
    
    def test_components_metadata(self):
        """Metadata should specify component names."""
        residual = mock_residual_capture()
        unembed = mock_unembed_matrix(hidden_dim=residual.hidden_dim)
        
        result = AttributionMethods.component_attribution(
            residual, None, unembed, target_token_id=42
        )
        
        assert result.metadata["components"] == ["attention", "mlp"]


# =============================================================================
# TOP ATTRIBUTIONS TESTS
# =============================================================================

class TestTopAttributions:
    """Tests for top_attributions utility."""
    
    def test_returns_correct_count(self):
        """Should return k items."""
        result = AttributionResult(
            values=torch.randn(4, 4),
            method="test"
        )
        
        top = AttributionMethods.top_attributions(result, k=5)
        
        assert len(top) == 5
    
    def test_sorted_by_absolute_value(self):
        """Results should be sorted by |value| descending."""
        values = torch.tensor([[1.0, -3.0], [2.0, -0.5]])
        result = AttributionResult(values=values, method="test")
        
        top = AttributionMethods.top_attributions(result, k=4)
        
        abs_values = [abs(v) for _, v in top]
        assert abs_values == sorted(abs_values, reverse=True)
    
    def test_returns_correct_indices(self):
        """Should return correct multi-dimensional indices."""
        values = torch.zeros(2, 3)
        values[1, 2] = 10.0  # Max value
        result = AttributionResult(values=values, method="test")
        
        top = AttributionMethods.top_attributions(result, k=1)
        
        assert top[0][0] == (1, 2)
        assert top[0][1] == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
