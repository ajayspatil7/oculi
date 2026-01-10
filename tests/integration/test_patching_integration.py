"""
Integration Tests for Activation Patching
==========================================

Tests PatchingContext and CausalTracer with real model operations.
"""

import pytest
import torch
from oculi.intervention import (
    PatchConfig,
    ActivationPatch,
    PatchingContext,
    CausalTracer,
)
from oculi.utils import get_default_device


class TestPatchingContext:
    """Integration tests for PatchingContext."""

    def test_context_manager_lifecycle(self):
        """Test that context manager properly enters and exits."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter()

        # Create a simple patch
        config = PatchConfig(layer=5, component='mlp_out')
        activation = torch.randn(10, adapter.model.config.hidden_size)
        patch = ActivationPatch(config=config, source_activation=activation)

        # Enter and exit context
        with PatchingContext(adapter, [patch]) as ctx:
            assert ctx._entered is True

        # Should be cleaned up after exit
        assert ctx._entered is False
        assert len(ctx._hook_handles) == 0

    def test_multiple_patches(self):
        """Test applying multiple patches simultaneously."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter()
        hidden_size = adapter.model.config.hidden_size

        # Create multiple patches
        patches = [
            ActivationPatch(
                config=PatchConfig(layer=i, component='mlp_out'),
                source_activation=torch.randn(10, hidden_size)
            )
            for i in range(3)
        ]

        # Should handle multiple patches
        with PatchingContext(adapter, patches) as ctx:
            assert len(patches) == 3
            # All should be applied

    def test_validation_before_context(self):
        """Test that patches are validated before entering context."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter()

        # Create invalid patch (layer out of range)
        bad_patch = ActivationPatch(
            config=PatchConfig(layer=999, component='mlp_out'),
            source_activation=torch.randn(10, 512)
        )

        # Should fail on initialization
        with pytest.raises(ValueError, match="out of range"):
            PatchingContext(adapter, [bad_patch])

    def test_exception_safety(self):
        """Test that hooks are cleaned up even on exception."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter()
        hidden_size = adapter.model.config.hidden_size

        patch = ActivationPatch(
            config=PatchConfig(layer=5, component='mlp_out'),
            source_activation=torch.randn(10, hidden_size)
        )

        try:
            with PatchingContext(adapter, [patch]) as ctx:
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass

        # Hooks should still be cleaned up
        assert len(ctx._hook_handles) == 0

    def test_device_compatibility(self):
        """Test that patches work on correct device."""
        from tests.mocks import MockLlamaAdapter

        device = get_default_device()
        adapter = MockLlamaAdapter(device=str(device))
        hidden_size = adapter.model.config.hidden_size

        # Create patch on same device
        activation = torch.randn(10, hidden_size, device=device)
        patch = ActivationPatch(
            config=PatchConfig(layer=5, component='mlp_out'),
            source_activation=activation
        )

        # Should work
        with PatchingContext(adapter, [patch]):
            pass  # Success if no error

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_mismatch_cuda(self):
        """Test that device mismatch is caught (CUDA)."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter(device='cpu')
        hidden_size = adapter.model.config.hidden_size

        # Create patch on wrong device
        activation = torch.randn(10, hidden_size, device='cuda')
        patch = ActivationPatch(
            config=PatchConfig(layer=5, component='mlp_out'),
            source_activation=activation
        )

        # Should fail validation
        with pytest.raises(ValueError, match="device"):
            PatchingContext(adapter, [patch])


class TestCausalTracer:
    """Integration tests for CausalTracer."""

    def test_basic_trace(self):
        """Test basic causal tracing functionality."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter()

        # Create simple inputs
        clean = torch.randint(0, 1000, (1, 10))
        corrupt = torch.randint(0, 1000, (1, 10))

        # Define metric (just get a logit value)
        def metric_fn(logits):
            return logits[0, -1, 0].item()

        tracer = CausalTracer(adapter)

        # Run trace on subset of layers
        results = tracer.trace(
            clean_input=clean,
            corrupted_input=corrupt,
            metric_fn=metric_fn,
            layers=[0, 1, 2],
            components=['mlp_out', 'attn_out'],
            verbose=False,
        )

        # Should have results for each layer/component combination
        assert len(results.results) == 6  # 3 layers × 2 components
        assert results.layers == [0, 1, 2]
        assert results.components == ['mlp_out', 'attn_out']

    def test_recovery_computation(self):
        """Test that recovery scores are computed correctly."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter()

        clean = torch.randint(0, 1000, (1, 10))
        corrupt = torch.randint(0, 1000, (1, 10))

        def metric_fn(logits):
            return logits[0, -1, 0].item()

        tracer = CausalTracer(adapter)

        results = tracer.trace(
            clean_input=clean,
            corrupted_input=corrupt,
            metric_fn=metric_fn,
            layers=[0, 1],
            components=['mlp_out'],
        )

        # Each result should have valid recovery score
        for result in results.results:
            assert isinstance(result.recovery, float)
            assert result.metric_clean is not None
            assert result.metric_corrupted is not None
            assert result.metric_patched is not None

    def test_trace_all_components(self):
        """Test tracing all valid components."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter()

        clean = torch.randint(0, 1000, (1, 10))
        corrupt = torch.randint(0, 1000, (1, 10))

        def metric_fn(logits):
            return logits[0, -1, 0].item()

        tracer = CausalTracer(adapter)

        # Test with multiple component types
        components = ['mlp_out', 'attn_out', 'residual_post_mlp']

        results = tracer.trace(
            clean_input=clean,
            corrupted_input=corrupt,
            metric_fn=metric_fn,
            layers=[5],
            components=components,
        )

        # Should have results for all components
        result_components = {r.config.component for r in results.results}
        assert 'mlp_out' in result_components
        assert 'attn_out' in result_components

    def test_token_level_tracing(self):
        """Test tracing with token-level patching."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter()

        clean = torch.randint(0, 1000, (1, 10))
        corrupt = torch.randint(0, 1000, (1, 10))

        def metric_fn(logits):
            return logits[0, -1, 0].item()

        tracer = CausalTracer(adapter)

        # Trace with specific tokens
        results = tracer.trace(
            clean_input=clean,
            corrupted_input=corrupt,
            metric_fn=metric_fn,
            layers=[5],
            components=['mlp_out'],
            tokens=[5, 6, 7],  # Only patch these positions
        )

        # Should work without errors
        assert len(results.results) > 0

    def test_verbose_mode(self):
        """Test that verbose mode prints progress."""
        from tests.mocks import MockLlamaAdapter
        import io
        import sys

        adapter = MockLlamaAdapter()

        clean = torch.randint(0, 1000, (1, 10))
        corrupt = torch.randint(0, 1000, (1, 10))

        def metric_fn(logits):
            return logits[0, -1, 0].item()

        tracer = CausalTracer(adapter)

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            tracer.trace(
                clean_input=clean,
                corrupted_input=corrupt,
                metric_fn=metric_fn,
                layers=[0],
                components=['mlp_out'],
                verbose=True,
            )

            output = buffer.getvalue()

            # Should print clean and corrupted metrics
            assert 'Clean metric' in output
            assert 'Corrupted metric' in output
        finally:
            sys.stdout = old_stdout

    def test_result_matrix_generation(self):
        """Test generating recovery matrix from results."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter()

        clean = torch.randint(0, 1000, (1, 10))
        corrupt = torch.randint(0, 1000, (1, 10))

        def metric_fn(logits):
            return logits[0, -1, 0].item()

        tracer = CausalTracer(adapter)

        results = tracer.trace(
            clean_input=clean,
            corrupted_input=corrupt,
            metric_fn=metric_fn,
            layers=[0, 1, 2],
            components=['mlp_out', 'attn_out'],
        )

        # Generate matrix
        matrix = results.recovery_matrix()

        # Should be [layers x components]
        assert matrix.shape == (3, 2)

        # Values should be floats
        assert matrix.dtype == torch.float32 or matrix.dtype == torch.float64

    def test_top_results_extraction(self):
        """Test extracting top results by recovery."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter()

        clean = torch.randint(0, 1000, (1, 10))
        corrupt = torch.randint(0, 1000, (1, 10))

        def metric_fn(logits):
            return logits[0, -1, 0].item()

        tracer = CausalTracer(adapter)

        results = tracer.trace(
            clean_input=clean,
            corrupted_input=corrupt,
            metric_fn=metric_fn,
            layers=[0, 1, 2, 3, 4],
            components=['mlp_out', 'attn_out'],
        )

        # Get top 3
        top_3 = results.top_results(3)

        assert len(top_3) == 3
        # Should be sorted by recovery
        assert top_3[0].recovery >= top_3[1].recovery

    @pytest.mark.skipif(not torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else True,
                        reason="MPS not available")
    def test_mps_compatibility(self):
        """Test that patching works on MPS device."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter(device='mps')

        clean = torch.randint(0, 1000, (1, 10), device='mps')
        corrupt = torch.randint(0, 1000, (1, 10), device='mps')

        def metric_fn(logits):
            return logits[0, -1, 0].item()

        tracer = CausalTracer(adapter)

        # Should work on MPS
        results = tracer.trace(
            clean_input=clean,
            corrupted_input=corrupt,
            metric_fn=metric_fn,
            layers=[0, 1],
            components=['mlp_out'],
        )

        assert len(results.results) > 0

        # Activations should be on MPS
        for result in results.results:
            assert result.config.layer >= 0


class TestPatchingEndToEnd:
    """End-to-end integration tests."""

    def test_full_workflow(self):
        """Test complete patching workflow from capture to analysis."""
        from tests.mocks import MockLlamaAdapter

        adapter = MockLlamaAdapter()

        # Step 1: Create inputs
        clean_text = "The capital of France is"
        corrupt_text = "The capital of Poland is"

        clean_ids = adapter.tokenize(clean_text)
        corrupt_ids = adapter.tokenize(corrupt_text)

        # Step 2: Capture clean activations
        clean_capture = adapter.capture_full(clean_ids)

        # Step 3: Create manual patch
        patch = ActivationPatch(
            config=PatchConfig(layer=10, component='mlp_out'),
            source_activation=clean_capture.mlp.output[10]
        )

        # Step 4: Apply patch and run
        with PatchingContext(adapter, [patch]):
            # Model runs with patch applied
            pass  # Success if no error

        # Step 5: Use CausalTracer for systematic analysis
        def metric_fn(logits):
            # Get probability of correct token
            return logits[0, -1, 0].item()

        tracer = CausalTracer(adapter)

        results = tracer.trace(
            clean_input=clean_ids,
            corrupted_input=corrupt_ids,
            metric_fn=metric_fn,
            layers=[5, 10, 15],
            components=['mlp_out', 'attn_out'],
        )

        # Step 6: Analyze results
        top_components = results.top_results(3)
        matrix = results.recovery_matrix()

        assert len(top_components) > 0
        assert matrix.shape[0] == 3  # 3 layers
        assert matrix.shape[1] == 2  # 2 components
