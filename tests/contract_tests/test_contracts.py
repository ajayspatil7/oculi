"""
Contract Tests for Spectra API
==============================

These tests DEFINE THE CONTRACT independent of implementation.

If implementation and tests disagree, the tests are the source of truth
(as documented in API_CONTRACT.md).

Run with: pytest tests/contract_tests/
"""

import pytest
import torch


# =============================================================================
# MOCK FIXTURES
# =============================================================================

def mock_capture(
    n_layers: int = 4,
    n_heads: int = 8,
    n_kv_heads: int = 8,
    n_tokens: int = 16,
    head_dim: int = 64
):
    """Create a mock AttentionCapture for testing."""
    from oculi.capture.structures import AttentionCapture
    
    # Random but deterministic data
    torch.manual_seed(42)
    
    queries = torch.randn(n_layers, n_heads, n_tokens, head_dim)
    keys = torch.randn(n_layers, n_kv_heads, n_tokens, head_dim)
    values = torch.randn(n_layers, n_kv_heads, n_tokens, head_dim)
    
    # Create valid attention patterns (sum to 1, causal masked)
    patterns = torch.rand(n_layers, n_heads, n_tokens, n_tokens)
    # Apply causal mask
    causal_mask = torch.triu(torch.ones(n_tokens, n_tokens), diagonal=1).bool()
    patterns.masked_fill_(causal_mask, 0)
    # Normalize
    patterns = patterns / patterns.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    
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
        qk_stage="pre_rope",
        captured_layers=tuple(range(n_layers)),
    )


# =============================================================================
# SHAPE CONTRACT TESTS
# =============================================================================

class TestShapeContracts:
    """Verify output shapes match API contract."""
    
    def test_q_norms_shape(self):
        """q_norms returns [n_layers, n_heads, n_tokens]."""
        from oculi.analysis import NormAnalysis
        
        capture = mock_capture(n_layers=4, n_heads=8, n_tokens=16)
        norms = NormAnalysis.q_norms(capture)
        
        assert norms.shape == (4, 8, 16), f"Expected (4, 8, 16), got {norms.shape}"
    
    def test_k_norms_shape(self):
        """k_norms returns [n_layers, n_kv_heads, n_tokens]."""
        from oculi.analysis import NormAnalysis
        
        capture = mock_capture(n_layers=4, n_heads=8, n_kv_heads=2, n_tokens=16)
        norms = NormAnalysis.k_norms(capture)
        
        assert norms.shape == (4, 2, 16), f"Expected (4, 2, 16), got {norms.shape}"
    
    def test_token_entropy_shape(self):
        """token_entropy returns [n_layers, n_heads, n_tokens]."""
        from oculi.analysis import EntropyAnalysis
        
        capture = mock_capture(n_layers=4, n_heads=8, n_tokens=16)
        entropy = EntropyAnalysis.token_entropy(capture)
        
        assert entropy.shape == (4, 8, 16), f"Expected (4, 8, 16), got {entropy.shape}"
    
    def test_mean_entropy_shape(self):
        """mean_entropy returns [n_layers, n_heads]."""
        from oculi.analysis import EntropyAnalysis
        
        capture = mock_capture(n_layers=4, n_heads=8, n_tokens=16)
        entropy = EntropyAnalysis.mean_entropy(capture)
        
        assert entropy.shape == (4, 8), f"Expected (4, 8), got {entropy.shape}"
    
    def test_delta_entropy_shape(self):
        """delta_entropy returns [n_layers, n_heads]."""
        from oculi.analysis import EntropyAnalysis
        
        capture1 = mock_capture(n_layers=4, n_heads=8, n_tokens=16)
        capture2 = mock_capture(n_layers=4, n_heads=8, n_tokens=20)
        
        delta = EntropyAnalysis.delta_entropy(capture1, capture2)
        
        assert delta.shape == (4, 8), f"Expected (4, 8), got {delta.shape}"
    
    def test_max_weight_shape(self):
        """max_weight returns [n_layers, n_heads, n_tokens]."""
        from oculi.analysis import AttentionAnalysis
        
        capture = mock_capture(n_layers=4, n_heads=8, n_tokens=16)
        max_w = AttentionAnalysis.max_weight(capture)
        
        assert max_w.shape == (4, 8, 16), f"Expected (4, 8, 16), got {max_w.shape}"
    
    def test_effective_span_shape(self):
        """effective_span returns [n_layers, n_heads, n_tokens]."""
        from oculi.analysis import AttentionAnalysis
        
        capture = mock_capture(n_layers=4, n_heads=8, n_tokens=16)
        k_eff = AttentionAnalysis.effective_span(capture)
        
        assert k_eff.shape == (4, 8, 16), f"Expected (4, 8, 16), got {k_eff.shape}"
    
    def test_correlation_shape(self):
        """norm_entropy_correlation returns [n_layers, n_heads]."""
        from oculi.analysis import CorrelationAnalysis
        
        capture = mock_capture(n_layers=4, n_heads=8, n_tokens=16)
        corr = CorrelationAnalysis.norm_entropy_correlation(capture)
        
        assert corr.shape == (4, 8), f"Expected (4, 8), got {corr.shape}"


# =============================================================================
# SEMANTIC CONTRACT TESTS
# =============================================================================

class TestSemanticContracts:
    """Verify mathematical semantics match API contract."""
    
    def test_entropy_causal_masking(self):
        """Entropy respects causal structure."""
        from oculi.capture.structures import AttentionCapture
        from oculi.analysis import EntropyAnalysis
        
        # Create known attention pattern
        # Token 0: attends only to itself (entropy = 0)
        # Token 1: uniform over [0, 1] (entropy = log(2) ≈ 0.693)
        # Token 2: uniform over [0, 1, 2] (entropy = log(3) ≈ 1.099)
        patterns = torch.tensor([
            [  # layer 0
                [  # head 0
                    [1.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [1/3, 1/3, 1/3]
                ]
            ]
        ])  # [1, 1, 3, 3]
        
        capture = AttentionCapture(
            patterns=patterns,
            n_layers=1,
            n_heads=1,
            n_kv_heads=1,
            n_tokens=3,
            head_dim=64,
            model_name="test",
        )
        
        entropy = EntropyAnalysis.token_entropy(capture, ignore_first=0)
        
        # Check values
        assert torch.isclose(entropy[0, 0, 0], torch.tensor(0.0), atol=1e-3).item(), \
            f"Token 0 entropy should be 0, got {entropy[0, 0, 0]}"
        assert torch.isclose(entropy[0, 0, 1], torch.tensor(0.693), atol=1e-2).item(), \
            f"Token 1 entropy should be log(2)≈0.693, got {entropy[0, 0, 1]}"
        assert torch.isclose(entropy[0, 0, 2], torch.tensor(1.099), atol=1e-2).item(), \
            f"Token 2 entropy should be log(3)≈1.099, got {entropy[0, 0, 2]}"
    
    def test_entropy_first_tokens_nan(self):
        """First tokens should be NaN with default ignore_first=2."""
        from oculi.analysis import EntropyAnalysis
        
        capture = mock_capture(n_tokens=10)
        entropy = EntropyAnalysis.token_entropy(capture, ignore_first=2)
        
        assert torch.isnan(entropy[..., 0]).all(), "Token 0 should be NaN"
        assert torch.isnan(entropy[..., 1]).all(), "Token 1 should be NaN"
        assert not torch.isnan(entropy[..., 2]).any(), "Token 2 should not be NaN"
    
    def test_norm_is_positive(self):
        """Norms should always be non-negative."""
        from oculi.analysis import NormAnalysis
        
        capture = mock_capture()
        norms = NormAnalysis.q_norms(capture)
        
        assert (norms >= 0).all(), "All norms should be >= 0"
    
    def test_max_weight_in_range(self):
        """Max weight should be in [0, 1]."""
        from oculi.analysis import AttentionAnalysis
        
        capture = mock_capture()
        max_w = AttentionAnalysis.max_weight(capture)
        
        assert (max_w >= 0).all(), "Max weight should be >= 0"
        assert (max_w <= 1).all(), "Max weight should be <= 1"
    
    def test_effective_span_minimum_one(self):
        """Effective span should be at least 1."""
        from oculi.analysis import AttentionAnalysis
        
        capture = mock_capture()
        k_eff = AttentionAnalysis.effective_span(capture)
        
        assert (k_eff >= 1).all(), "Effective span should be >= 1"
    
    def test_effective_rank_exp_entropy(self):
        """Effective rank = exp(entropy)."""
        from oculi.analysis import EntropyAnalysis
        
        capture = mock_capture()
        entropy = EntropyAnalysis.token_entropy(capture, ignore_first=0)
        eff_rank = EntropyAnalysis.effective_rank(capture, ignore_first=0)
        
        # Check relationship
        expected = torch.exp(entropy)
        assert torch.allclose(eff_rank, expected, atol=1e-5), \
            "effective_rank should equal exp(entropy)"
    
    def test_correlation_in_range(self):
        """Pearson correlation should be in [-1, 1]."""
        from oculi.analysis import CorrelationAnalysis
        
        x = torch.randn(100)
        y = torch.randn(100)
        r = CorrelationAnalysis.pearson(x, y)
        
        assert -1 <= r <= 1, f"Correlation should be in [-1, 1], got {r}"


# =============================================================================
# INTERVENTION CONTRACT TESTS
# =============================================================================

class TestInterventionContracts:
    """Verify intervention semantics."""
    
    def test_spectra_scaler_sqrt_decomposition(self):
        """SpectraScaler applies √α to Q and K."""
        from oculi.intervention import SpectraScaler
        
        scaler = SpectraScaler(layer=5, head=3, alpha=4.0)
        
        assert scaler.q_scale_factor == 2.0, "√4 = 2"
        assert scaler.k_scale_factor == 2.0, "√4 = 2"
    
    def test_scaler_validation(self):
        """Scalers should reject invalid parameters."""
        from oculi.intervention import QScaler
        
        scaler = QScaler(layer=5, head=3, alpha=-1.0)
        
        # Mock adapter
        class MockAdapter:
            def num_layers(self): return 10
            def num_heads(self, layer): return 8
        
        with pytest.raises(ValueError, match="Alpha must be positive"):
            scaler.validate(MockAdapter())
    
    def test_layer_out_of_range(self):
        """Should reject layer >= num_layers."""
        from oculi.intervention import QScaler
        
        scaler = QScaler(layer=20, head=3, alpha=1.5)
        
        class MockAdapter:
            def num_layers(self): return 10
            def num_heads(self, layer): return 8
        
        with pytest.raises(ValueError, match="out of range"):
            scaler.validate(MockAdapter())


# =============================================================================
# STRUCTURE CONTRACT TESTS
# =============================================================================

class TestStructureContracts:
    """Verify data structure invariants."""
    
    def test_attention_structure_gqa_detection(self):
        """AttentionStructure correctly identifies GQA."""
        from oculi.capture.structures import AttentionStructure
        
        # MHA case
        mha = AttentionStructure(n_query_heads=32, n_kv_heads=32, head_dim=128)
        assert mha.attention_type == "MHA"
        assert not mha.is_gqa
        
        # GQA case
        gqa = AttentionStructure(n_query_heads=32, n_kv_heads=8, head_dim=128)
        assert gqa.attention_type == "GQA"
        assert gqa.is_gqa
        assert gqa.gqa_ratio == 4
        
        # MQA case
        mqa = AttentionStructure(n_query_heads=32, n_kv_heads=1, head_dim=128)
        assert mqa.attention_type == "MQA"
        assert mqa.is_gqa
    
    def test_capture_config_validation(self):
        """CaptureConfig validates correctly."""
        from oculi.capture.structures import CaptureConfig
        
        # Valid config
        config = CaptureConfig(layers=[0, 1, 2])
        config.validate(num_layers=10)  # Should not raise
        
        # Invalid layer
        config = CaptureConfig(layers=[15])
        with pytest.raises(ValueError, match="out of range"):
            config.validate(num_layers=10)
        
        # No captures
        config = CaptureConfig(
            capture_queries=False,
            capture_keys=False,
            capture_values=False,
            capture_patterns=False
        )
        with pytest.raises(ValueError, match="At least one"):
            config.validate(num_layers=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
