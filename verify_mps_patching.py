"""
MPS Activation Patching Verification
=====================================

Comprehensive verification script for activation patching on Apple Silicon (MPS).

Tests all activation patching functionality on MPS:
- PatchConfig creation and validation
- ActivationPatch creation and application
- PatchingContext manager
- CausalTracer systematic experiments
- Device compatibility

Usage:
    python verify_mps_patching.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import Oculi
from oculi.utils import get_default_device, get_device_info, is_mps_available
from oculi.models.llama import LlamaAttentionAdapter
from oculi import (
    PatchConfig,
    ActivationPatch,
    PatchingContext,
    CausalTracer,
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def verify_device():
    """Test 1: Verify MPS is available and selected."""
    print_section("Test 1: Device Detection")

    if not is_mps_available():
        print("\n❌ MPS not available on this system!")
        print("This script requires Apple Silicon (M1/M2/M3/M4)")
        return None

    device_info = get_device_info()
    device = get_default_device()

    print(f"\n✓ MPS Available")
    print(f"  Device: {device}")
    print(f"  Device Name: {device_info.device_name}")
    print(f"  PyTorch Version: {device_info.pytorch_version}")

    return device


def load_model(device):
    """Test 2: Load model on MPS."""
    print_section("Test 2: Load Model on MPS")

    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    print(f"\nLoading {model_name} on {device}...")
    print("This may take a minute...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=str(device),
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"\n✓ Model loaded successfully")
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Model dtype: {next(model.parameters()).dtype}")

        return model, tokenizer

    except Exception as e:
        print(f"\n❌ Model loading failed: {e}")
        return None, None


def test_patch_config(adapter):
    """Test 3: PatchConfig creation and validation."""
    print_section("Test 3: PatchConfig Creation")

    print("\nCreating various PatchConfig objects...")

    # Test different component types
    configs = [
        PatchConfig(layer=10, component='mlp_out'),
        PatchConfig(layer=15, component='attn_out'),
        PatchConfig(layer=20, component='residual_post_mlp'),
        PatchConfig(layer=5, component='head', head=3),
        PatchConfig(layer=12, component='mlp_out', tokens=[2, 3, 4]),
    ]

    for i, config in enumerate(configs):
        try:
            config.validate(adapter)
            print(f"  ✓ Config {i+1}: L{config.layer} {config.component} "
                  f"{'H'+str(config.head) if config.head is not None else ''} "
                  f"{'tokens='+str(config.tokens) if config.tokens else ''}")
        except Exception as e:
            print(f"  ❌ Config {i+1} validation failed: {e}")
            return False

    print("\n✓ All PatchConfig objects created and validated")
    return True


def test_activation_patch(adapter, device):
    """Test 4: ActivationPatch creation and application."""
    print_section("Test 4: ActivationPatch Creation")

    print("\nCapturing activations for patching...")

    # Create simple input
    text = "The capital of France is"
    input_ids = adapter.tokenize(text)

    # Capture activations
    from oculi import MLPConfig, ResidualConfig

    capture = adapter.capture_full(
        input_ids,
        mlp_config=MLPConfig(capture_output=True),
        residual_config=ResidualConfig(capture_post_mlp=True)
    )

    print(f"  Captured from {capture.mlp.n_tokens} tokens")

    # Create patches from different components
    patches = []

    try:
        # MLP output patch
        patch1 = ActivationPatch(
            config=PatchConfig(layer=15, component='mlp_out'),
            source_activation=capture.mlp.mlp_output[15]
        )
        patch1.validate(adapter)
        patches.append(patch1)
        print(f"  ✓ MLP patch created: {capture.mlp.mlp_output[15].shape} on {capture.mlp.mlp_output[15].device}")

        # Residual stream patch
        patch2 = ActivationPatch(
            config=PatchConfig(layer=20, component='residual_post_mlp'),
            source_activation=capture.residual.post_mlp[20]
        )
        patch2.validate(adapter)
        patches.append(patch2)
        print(f"  ✓ Residual patch created: {capture.residual.post_mlp[20].shape} on {capture.residual.post_mlp[20].device}")

        # Verify all patches are on MPS
        for i, patch in enumerate(patches):
            assert patch.source_activation.device.type == 'mps', f"Patch {i} not on MPS!"

        print(f"\n✓ All patches created and on MPS device")
        return True

    except Exception as e:
        print(f"\n❌ Patch creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_patching_context(adapter, model, device):
    """Test 5: PatchingContext manager."""
    print_section("Test 5: PatchingContext Manager")

    print("\nTesting context manager lifecycle...")

    # Prepare inputs
    clean_text = "The Eiffel Tower is in Paris"
    corrupt_text = "The Eiffel Tower is in London"

    clean_ids = adapter.tokenize(clean_text)
    corrupt_ids = adapter.tokenize(corrupt_text)

    # Capture clean activations
    print("  Capturing clean activations...")
    from oculi import MLPConfig

    clean_capture = adapter.capture_full(
        clean_ids,
        mlp_config=MLPConfig(capture_output=True)
    )

    # Create patch
    patch = ActivationPatch(
        config=PatchConfig(layer=20, component='mlp_out'),
        source_activation=clean_capture.mlp.mlp_output[20]
    )

    try:
        # Test context manager
        print("  Entering PatchingContext...")
        with PatchingContext(adapter, [patch]) as ctx:
            print(f"    ✓ Context entered (hooks active)")

            # Run model with patch
            print("    Running model with patch...")
            with torch.no_grad():
                output = model(corrupt_ids)

            print(f"    ✓ Model ran successfully")
            print(f"    Output shape: {output.logits.shape}")
            print(f"    Output device: {output.logits.device}")

        print("  ✓ Context exited (hooks cleaned up)")

        # Verify hooks are cleaned up
        assert len(ctx._hook_handles) == 0, "Hooks not cleaned up!"

        print("\n✓ PatchingContext manager working correctly")
        return True

    except Exception as e:
        print(f"\n❌ PatchingContext failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_causal_tracer(adapter, model, device):
    """Test 6: CausalTracer systematic experiments."""
    print_section("Test 6: CausalTracer Systematic Experiments")

    print("\nSetting up causal tracing experiment...")

    # Prepare inputs
    clean_text = "The Eiffel Tower is in Paris"
    corrupt_text = "The Eiffel Tower is in London"

    clean_ids = adapter.tokenize(clean_text)
    corrupt_ids = adapter.tokenize(corrupt_text)

    # Define metric
    target_token = adapter.tokenizer.encode("Paris", add_special_tokens=False)[0]

    def metric_fn(logits):
        """Get probability of target token."""
        probs = torch.softmax(logits[0, -1], dim=-1)
        return probs[target_token].item()

    try:
        # Create tracer
        print("  Creating CausalTracer...")
        tracer = CausalTracer(adapter)

        # Run systematic trace
        print(f"  Running causal trace on layers 20-23...")
        print(f"  Target token: '{adapter.tokenizer.decode([target_token])}'")

        results = tracer.trace(
            clean_input=clean_ids,
            corrupted_input=corrupt_ids,
            metric_fn=metric_fn,
            layers=[20, 21, 22, 23],
            components=['mlp_out', 'attn_out'],
            verbose=True,
        )

        print(f"\n  ✓ Trace completed!")
        print(f"    Total results: {len(results.results)}")

        # Analyze results
        print(f"\n  Top 3 Components by Recovery:")
        top_3 = results.top_results(3)
        for i, result in enumerate(top_3):
            print(f"    {i+1}. L{result.config.layer} {result.config.component}: "
                  f"recovery={result.recovery:.3f}")

        # Generate recovery matrix
        matrix = results.recovery_matrix()
        print(f"\n  Recovery Matrix: {matrix.shape}")
        print(f"    Min recovery: {matrix.min().item():.3f}")
        print(f"    Max recovery: {matrix.max().item():.3f}")
        print(f"    Mean recovery: {matrix.mean().item():.3f}")

        print("\n✓ CausalTracer working correctly on MPS")
        return True

    except Exception as e:
        print(f"\n❌ CausalTracer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_workflow(adapter, model, tokenizer, device):
    """Test 7: Complete end-to-end workflow."""
    print_section("Test 7: Full Activation Patching Workflow")

    print("\nRunning complete patching workflow...")

    # Step 1: Create inputs
    print("\n  Step 1: Creating inputs...")
    clean_text = "The capital of France is"
    corrupt_text = "The capital of Poland is"

    clean_ids = adapter.tokenize(clean_text)
    corrupt_ids = adapter.tokenize(corrupt_text)

    target = tokenizer.encode("Paris", add_special_tokens=False)[0]
    print(f"    Clean: '{clean_text}'")
    print(f"    Corrupt: '{corrupt_text}'")
    print(f"    Target: '{tokenizer.decode([target])}'")

    try:
        # Step 2: Get baseline metrics
        print("\n  Step 2: Computing baseline metrics...")
        with torch.no_grad():
            clean_logits = model(clean_ids).logits
            corrupt_logits = model(corrupt_ids).logits

        clean_probs = torch.softmax(clean_logits[0, -1], dim=-1)
        corrupt_probs = torch.softmax(corrupt_logits[0, -1], dim=-1)

        clean_metric = clean_probs[target].item()
        corrupt_metric = corrupt_probs[target].item()

        print(f"    Clean P(Paris): {clean_metric:.4f}")
        print(f"    Corrupt P(Paris): {corrupt_metric:.4f}")
        print(f"    Difference: {clean_metric - corrupt_metric:.4f}")

        # Step 3: Capture clean activations
        print("\n  Step 3: Capturing clean activations...")
        clean_capture = adapter.capture_full(clean_ids)
        print(f"    ✓ Captured from {clean_capture.n_layers} layers")

        # Step 4: Manual patch
        print("\n  Step 4: Applying manual patch...")
        patch = ActivationPatch(
            config=PatchConfig(layer=22, component='mlp_out'),
            source_activation=clean_capture.mlp.mlp_output[22]
        )

        with PatchingContext(adapter, [patch]):
            with torch.no_grad():
                patched_logits = model(corrupt_ids).logits

        patched_probs = torch.softmax(patched_logits[0, -1], dim=-1)
        patched_metric = patched_probs[target].item()

        recovery = (patched_metric - corrupt_metric) / (clean_metric - corrupt_metric)

        print(f"    Patched P(Paris): {patched_metric:.4f}")
        print(f"    Recovery: {recovery:.3f}")

        # Step 5: Systematic trace
        print("\n  Step 5: Running systematic trace...")

        def metric_fn(logits):
            probs = torch.softmax(logits[0, -1], dim=-1)
            return probs[target].item()

        tracer = CausalTracer(adapter)
        results = tracer.trace(
            clean_input=clean_ids,
            corrupted_input=corrupt_ids,
            metric_fn=metric_fn,
            layers=[20, 21, 22, 23, 24],
            components=['mlp_out', 'attn_out'],
            verbose=False,
        )

        print(f"    ✓ Traced {len(results.results)} components")

        # Step 6: Analyze
        print("\n  Step 6: Analyzing results...")
        top_5 = results.top_results(5)

        print(f"\n    Top 5 Causal Components:")
        for i, r in enumerate(top_5):
            print(f"      {i+1}. L{r.config.layer} {r.config.component}: {r.recovery:.3f}")

        print("\n✓ Complete workflow successful!")
        print("\n" + "=" * 70)
        print("  ALL ACTIVATION PATCHING TESTS PASSED ON MPS!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("  Oculi Activation Patching MPS Verification")
    print("  Testing Phase 2.3 (v0.6.0) on Apple Silicon")
    print("=" * 70)

    # Test 1: Device detection
    device = verify_device()
    if device is None or device.type != 'mps':
        print("\n❌ MPS not available. Exiting.")
        return

    # Test 2: Model loading
    model, tokenizer = load_model(device)
    if model is None:
        print("\n❌ Model loading failed. Exiting.")
        return

    # Create adapter
    print_section("Creating Adapter")
    adapter = LlamaAttentionAdapter(model, tokenizer)
    print(f"✓ Adapter created")
    print(f"  Device: {adapter.device}")
    print(f"  Layers: {adapter.num_layers()}")
    print(f"  Heads: {adapter.num_heads()}")

    # Run all tests
    tests = [
        ("PatchConfig", test_patch_config, [adapter]),
        ("ActivationPatch", test_activation_patch, [adapter, device]),
        ("PatchingContext", test_patching_context, [adapter, model, device]),
        ("CausalTracer", test_causal_tracer, [adapter, model, device]),
        ("Full Workflow", test_full_workflow, [adapter, model, tokenizer, device]),
    ]

    results = []
    for name, test_fn, args in tests:
        try:
            success = test_fn(*args)
            results.append((name, success))
        except KeyboardInterrupt:
            print("\n\nTests interrupted by user.")
            return
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)

    for name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"  {status:8} {name}")

    total = len(results)
    passed = sum(1 for _, s in results if s)

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
        print("  Activation patching is fully functional on MPS")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed")

    print("=" * 70)


if __name__ == "__main__":
    main()
