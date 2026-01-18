"""
Unit Tests for Activation Patching
====================================

Tests for PatchConfig, ActivationPatch, PatchingResult, and PatchingSweepResult.
"""

import pytest
import torch
from oculi.intervention.patching import (
    PatchConfig,
    ActivationPatch,
    PatchingResult,
    PatchingSweepResult,
    VALID_COMPONENTS,
)


class TestPatchConfig:
    """Test PatchConfig validation and configuration."""

    def test_valid_config(self):
        """Test creating valid PatchConfig."""
        config = PatchConfig(layer=20, component='mlp_out')
        assert config.layer == 20
        assert config.component == 'mlp_out'
        assert config.head is None
        assert config.tokens is None

    def test_head_component_requires_head(self):
        """Test that head component requires head parameter."""
        with pytest.raises(ValueError, match="head parameter is required"):
            PatchConfig(layer=20, component='head')

        # Should work with head
        config = PatchConfig(layer=20, component='head', head=5)
        assert config.head == 5

    def test_non_head_component_rejects_head(self):
        """Test that non-head components reject head parameter."""
        with pytest.raises(ValueError, match="head parameter should only be set"):
            PatchConfig(layer=20, component='mlp_out', head=5)

    def test_invalid_component(self):
        """Test that invalid component names are rejected."""
        with pytest.raises(ValueError, match="Invalid component"):
            PatchConfig(layer=20, component='invalid_component')

    def test_all_valid_components(self):
        """Test all valid component types."""
        for comp in VALID_COMPONENTS:
            if comp == 'head':
                config = PatchConfig(layer=0, component=comp, head=0)
            else:
                config = PatchConfig(layer=0, component=comp)
            assert config.component == comp

    def test_token_selection(self):
        """Test token-level patching configuration."""
        config = PatchConfig(layer=20, component='mlp_out', tokens=[2, 3, 4])
        assert config.tokens == [2, 3, 4]

    def test_validate_against_adapter(self):
        """Test validation against mock adapter."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter()

        # Valid config
        config = PatchConfig(layer=2, component='mlp_out')
        config.validate(adapter)  # Should not raise

        # Invalid layer (out of range)
        config_bad = PatchConfig(layer=999, component='mlp_out')
        with pytest.raises(ValueError, match="out of range"):
            config_bad.validate(adapter)

        # Invalid head
        config_bad_head = PatchConfig(layer=2, component='head', head=999)
        with pytest.raises(ValueError, match="out of range"):
            config_bad_head.validate(adapter)


class TestActivationPatch:
    """Test ActivationPatch creation and validation."""

    def test_create_patch(self):
        """Test creating a basic patch."""
        config = PatchConfig(layer=20, component='mlp_out')
        activation = torch.randn(10, 512)  # [T, H]

        patch = ActivationPatch(
            config=config,
            source_activation=activation
        )

        assert patch.layer == 20
        assert patch.component == 'mlp_out'
        assert patch.head is None
        assert patch.source_activation.shape == (10, 512)

    def test_patch_convenience_properties(self):
        """Test convenience property accessors."""
        config = PatchConfig(layer=20, component='head', head=5, tokens=[2, 3])
        activation = torch.randn(10, 128)

        patch = ActivationPatch(config=config, source_activation=activation)

        assert patch.layer == 20
        assert patch.component == 'head'
        assert patch.head == 5
        assert patch.tokens == [2, 3]

    def test_validate_shape_mismatch(self):
        """Test that shape mismatches are caught."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter()

        # Wrong shape for mlp_out (should be [T, hidden_size])
        config = PatchConfig(layer=2, component='mlp_out')
        wrong_activation = torch.randn(10, 999)  # Wrong hidden size

        patch = ActivationPatch(config=config, source_activation=wrong_activation)

        with pytest.raises(ValueError, match="shape mismatch"):
            patch.validate(adapter)

    def test_validate_device_mismatch(self):
        """Test that device mismatches are caught."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter(device='cpu')

        config = PatchConfig(layer=5, component='mlp_out')

        # Create activation on wrong device
        if torch.cuda.is_available():
            wrong_device_activation = torch.randn(10, 512, device='cuda')
            patch = ActivationPatch(config=config, source_activation=wrong_device_activation)

            with pytest.raises(ValueError, match="device"):
                patch.validate(adapter)

    def test_apply_full_replacement(self):
        """Test applying patch to replace all tokens."""
        config = PatchConfig(layer=20, component='mlp_out')
        source = torch.randn(10, 512)
        target = torch.randn(10, 512)

        patch = ActivationPatch(config=config, source_activation=source)
        result = patch.apply(target)

        # Should replace entire activation
        assert torch.allclose(result, source)
        assert not torch.allclose(result, target)

    def test_apply_token_level_replacement(self):
        """Test applying patch to specific tokens only."""
        config = PatchConfig(layer=20, component='mlp_out', tokens=[2, 3])
        source = torch.randn(10, 512)
        target = torch.randn(10, 512)

        patch = ActivationPatch(config=config, source_activation=source)
        result = patch.apply(target)

        # Tokens 2 and 3 should match source
        assert torch.allclose(result[2], source[2])
        assert torch.allclose(result[3], source[3])

        # Other tokens should match target
        assert torch.allclose(result[0], target[0])
        assert torch.allclose(result[1], target[1])
        assert torch.allclose(result[4], target[4])


class TestPatchingResult:
    """Test PatchingResult and recovery computation."""

    def test_create_result(self):
        """Test creating a basic result."""
        config = PatchConfig(layer=20, component='mlp_out')
        result = PatchingResult(
            config=config,
            metric_clean=1.0,
            metric_corrupted=0.0,
            metric_patched=0.8,
            recovery=0.8,
        )

        assert result.config.layer == 20
        assert result.metric_clean == 1.0
        assert result.recovery == 0.8

    def test_compute_recovery_full(self):
        """Test computing full recovery (recovery = 1.0)."""
        config = PatchConfig(layer=20, component='mlp_out')

        result = PatchingResult.compute(
            config=config,
            metric_clean=1.0,
            metric_corrupted=0.0,
            metric_patched=1.0,  # Fully recovered
        )

        assert result.recovery == pytest.approx(1.0)

    def test_compute_recovery_none(self):
        """Test computing no recovery (recovery = 0.0)."""
        config = PatchConfig(layer=20, component='mlp_out')

        result = PatchingResult.compute(
            config=config,
            metric_clean=1.0,
            metric_corrupted=0.0,
            metric_patched=0.0,  # No recovery
        )

        assert result.recovery == pytest.approx(0.0)

    def test_compute_recovery_partial(self):
        """Test computing partial recovery."""
        config = PatchConfig(layer=20, component='mlp_out')

        result = PatchingResult.compute(
            config=config,
            metric_clean=1.0,
            metric_corrupted=0.0,
            metric_patched=0.5,  # 50% recovery
        )

        assert result.recovery == pytest.approx(0.5)

    def test_compute_recovery_over(self):
        """Test computing over-recovery (recovery > 1.0)."""
        config = PatchConfig(layer=20, component='mlp_out')

        result = PatchingResult.compute(
            config=config,
            metric_clean=1.0,
            metric_corrupted=0.0,
            metric_patched=1.2,  # Over-compensated
        )

        assert result.recovery == pytest.approx(1.2)

    def test_compute_recovery_negative(self):
        """Test computing negative recovery (made worse)."""
        config = PatchConfig(layer=20, component='mlp_out')

        result = PatchingResult.compute(
            config=config,
            metric_clean=1.0,
            metric_corrupted=0.5,
            metric_patched=0.0,  # Made worse
        )

        assert result.recovery < 0.0

    def test_compute_recovery_no_difference(self):
        """Test when clean and corrupted are the same."""
        config = PatchConfig(layer=20, component='mlp_out')

        result = PatchingResult.compute(
            config=config,
            metric_clean=1.0,
            metric_corrupted=1.0,  # No difference
            metric_patched=1.0,
        )

        # Should return 0.0 when no difference
        assert result.recovery == 0.0

    def test_result_repr(self):
        """Test string representation."""
        config = PatchConfig(layer=20, component='mlp_out')
        result = PatchingResult(
            config=config,
            metric_clean=1.0,
            metric_corrupted=0.0,
            metric_patched=0.75,
            recovery=0.75,
        )

        repr_str = repr(result)
        assert 'layer=20' in repr_str
        assert 'mlp_out' in repr_str
        assert '0.75' in repr_str


class TestPatchingSweepResult:
    """Test PatchingSweepResult aggregation and analysis."""

    def test_create_sweep_result(self):
        """Test creating a sweep result."""
        results = [
            PatchingResult(
                config=PatchConfig(layer=i, component='mlp_out'),
                metric_clean=1.0,
                metric_corrupted=0.0,
                metric_patched=0.5,
                recovery=0.5,
            )
            for i in range(5)
        ]

        sweep = PatchingSweepResult(
            results=results,
            layers=list(range(5)),
            components=['mlp_out'],
        )

        assert len(sweep.results) == 5
        assert sweep.layers == list(range(5))
        assert sweep.components == ['mlp_out']

    def test_recovery_matrix(self):
        """Test generating recovery matrix."""
        results = []
        layers = [0, 1, 2]
        components = ['mlp_out', 'attn_out']

        for layer in layers:
            for comp in components:
                results.append(
                    PatchingResult(
                        config=PatchConfig(layer=layer, component=comp),
                        metric_clean=1.0,
                        metric_corrupted=0.0,
                        metric_patched=layer * 0.3,  # Varying recovery
                        recovery=layer * 0.3,
                    )
                )

        sweep = PatchingSweepResult(
            results=results,
            layers=layers,
            components=components,
        )

        matrix = sweep.recovery_matrix()

        # Shape should be [layers x components]
        assert matrix.shape == (3, 2)

        # Check values
        assert matrix[0, 0] == pytest.approx(0.0)  # Layer 0, mlp_out
        assert matrix[1, 0] == pytest.approx(0.3)  # Layer 1, mlp_out
        assert matrix[2, 0] == pytest.approx(0.6)  # Layer 2, mlp_out

    def test_top_results(self):
        """Test getting top-k results by recovery."""
        results = [
            PatchingResult(
                config=PatchConfig(layer=i, component='mlp_out'),
                metric_clean=1.0,
                metric_corrupted=0.0,
                metric_patched=i * 0.1,
                recovery=i * 0.1,
            )
            for i in range(10)
        ]

        sweep = PatchingSweepResult(
            results=results,
            layers=list(range(10)),
            components=['mlp_out'],
        )

        top_3 = sweep.top_results(3)

        assert len(top_3) == 3
        # Should be sorted by recovery (descending)
        assert top_3[0].recovery >= top_3[1].recovery >= top_3[2].recovery
        assert top_3[0].config.layer == 9  # Highest recovery

    def test_by_layer(self):
        """Test filtering results by layer."""
        results = []
        for layer in [0, 1, 2]:
            for comp in ['mlp_out', 'attn_out']:
                results.append(
                    PatchingResult(
                        config=PatchConfig(layer=layer, component=comp),
                        metric_clean=1.0,
                        metric_corrupted=0.0,
                        metric_patched=0.5,
                        recovery=0.5,
                    )
                )

        sweep = PatchingSweepResult(
            results=results,
            layers=[0, 1, 2],
            components=['mlp_out', 'attn_out'],
        )

        layer_1_results = sweep.by_layer(1)

        assert len(layer_1_results) == 2  # mlp_out and attn_out
        assert all(r.config.layer == 1 for r in layer_1_results)

    def test_by_component(self):
        """Test filtering results by component."""
        results = []
        for layer in [0, 1, 2]:
            for comp in ['mlp_out', 'attn_out']:
                results.append(
                    PatchingResult(
                        config=PatchConfig(layer=layer, component=comp),
                        metric_clean=1.0,
                        metric_corrupted=0.0,
                        metric_patched=0.5,
                        recovery=0.5,
                    )
                )

        sweep = PatchingSweepResult(
            results=results,
            layers=[0, 1, 2],
            components=['mlp_out', 'attn_out'],
        )

        mlp_results = sweep.by_component('mlp_out')

        assert len(mlp_results) == 3  # Layers 0, 1, 2
        assert all(r.config.component == 'mlp_out' for r in mlp_results)
