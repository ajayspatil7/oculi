#!/usr/bin/env python3
"""
Quick Test Script for Oculi with Mock LLaMA
=============================================

Run this to verify your Oculi installation works on your MacBook.

Usage:
    python tests/quick_test.py
"""

import sys
sys.path.insert(0, '/Users/ajaysp/oculi')

import torch
from tests.mocks.mock_llama import MockLlamaAdapter, create_mock_adapter

def test_basic_capture():
    """Test basic attention capture."""
    print("=" * 60)
    print("TEST 1: Basic Attention Capture")
    print("=" * 60)
    
    adapter = MockLlamaAdapter()
    
    print(f"\n📊 Model Configuration:")
    print(f"   • Layers: {adapter.num_layers()}")
    print(f"   • Query Heads: {adapter.num_heads()}")
    print(f"   • KV Heads: {adapter.num_kv_heads()}")
    print(f"   • Head Dim: {adapter.head_dim()}")
    print(f"   • Attention Type: {adapter.attention_structure().attention_type}")
    
    # Test capture
    text = "The quick brown fox jumps over the lazy dog."
    input_ids = adapter.tokenize(text)
    print(f"\n🔤 Input: '{text}'")
    print(f"   Tokens: {input_ids.shape[1]}")
    
    capture = adapter.capture(input_ids)
    
    print(f"\n📦 Captured Tensors:")
    print(f"   • Queries: {capture.queries.shape}")
    print(f"   • Keys: {capture.keys.shape}")
    print(f"   • Values: {capture.values.shape}")
    print(f"   • Patterns: {capture.patterns.shape}")
    
    # Validate shapes
    L, H, T, D = capture.queries.shape
    assert L == adapter.num_layers(), f"Expected {adapter.num_layers()} layers"
    assert H == adapter.num_heads(), f"Expected {adapter.num_heads()} heads"
    assert D == adapter.head_dim(), f"Expected {adapter.head_dim()} head_dim"
    
    print("\n✅ Basic capture test PASSED!")
    return capture


def test_selective_capture():
    """Test capturing specific layers/components."""
    print("\n" + "=" * 60)
    print("TEST 2: Selective Layer Capture")
    print("=" * 60)
    
    from oculi.capture.structures import CaptureConfig
    
    adapter = MockLlamaAdapter()
    input_ids = adapter.tokenize("Testing selective capture")
    
    # Capture only layers 1 and 2, only queries
    config = CaptureConfig(
        layers=[1, 2],
        capture_queries=True,
        capture_keys=False,
        capture_values=False,
        capture_patterns=True
    )
    
    capture = adapter.capture(input_ids, config=config)
    
    print(f"\n📦 Selective Capture (layers [1, 2], queries + patterns only):")
    print(f"   • Queries: {capture.queries.shape}")
    print(f"   • Keys: {capture.keys}")
    print(f"   • Values: {capture.values}")
    print(f"   • Patterns: {capture.patterns.shape}")
    print(f"   • Captured layers: {capture.captured_layers}")
    
    assert capture.n_layers == 2, "Should have captured 2 layers"
    assert capture.keys is None, "Keys should not be captured"
    assert capture.values is None, "Values should not be captured"
    
    print("\n✅ Selective capture test PASSED!")


def test_attention_structure():
    """Test GQA structure inspection."""
    print("\n" + "=" * 60)
    print("TEST 3: GQA Structure Inspection")
    print("=" * 60)
    
    adapter = MockLlamaAdapter()
    structure = adapter.attention_structure()
    
    print(f"\n🔍 Attention Structure:")
    print(f"   • Query Heads: {structure.n_query_heads}")
    print(f"   • KV Heads: {structure.n_kv_heads}")
    print(f"   • Head Dim: {structure.head_dim}")
    print(f"   • Attention Type: {structure.attention_type}")
    print(f"   • GQA Ratio: {structure.gqa_ratio}:1")
    
    assert structure.attention_type == "GQA", "Should be GQA"
    assert structure.gqa_ratio == 2, "Should be 2:1 ratio"
    
    print("\n✅ GQA structure test PASSED!")


def test_hooks():
    """Test hook registration and removal."""
    print("\n" + "=" * 60)
    print("TEST 4: Hook Management")
    print("=" * 60)
    
    adapter = MockLlamaAdapter()
    input_ids = adapter.tokenize("Testing hooks")
    
    captured_data = []
    
    def my_hook(module, input, output):
        captured_data.append(output.detach().clone())
    
    # Add hook
    handle_id = adapter.add_hook(my_hook, layer=1, component='q')
    print(f"\n🔗 Added hook: {handle_id}")
    
    # Forward pass
    with torch.no_grad():
        adapter._model(input_ids)
    
    print(f"   Captured tensors: {len(captured_data)}")
    assert len(captured_data) == 1, "Should have captured 1 tensor"
    
    # Remove hook
    adapter.remove_hook(handle_id)
    print(f"   Removed hook: {handle_id}")
    
    # Forward again - should not capture
    captured_data.clear()
    with torch.no_grad():
        adapter._model(input_ids)
    
    assert len(captured_data) == 0, "Should not capture after removal"
    
    print("\n✅ Hook management test PASSED!")


def test_custom_dimensions():
    """Test creating adapters with custom dimensions."""
    print("\n" + "=" * 60)
    print("TEST 5: Custom Model Dimensions")
    print("=" * 60)
    
    # Create larger mock
    adapter = create_mock_adapter(
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        hidden_size=512
    )
    
    print(f"\n📊 Custom Model (8L/8H/4KV/512D):")
    print(f"   • Layers: {adapter.num_layers()}")
    print(f"   • Query Heads: {adapter.num_heads()}")
    print(f"   • KV Heads: {adapter.num_kv_heads()}")
    print(f"   • Head Dim: {adapter.head_dim()}")
    
    assert adapter.num_layers() == 8
    assert adapter.num_heads() == 8
    assert adapter.num_kv_heads() == 4
    
    capture = adapter.capture(adapter.tokenize("Custom dimensions test"))
    print(f"\n📦 Capture shapes:")
    print(f"   • Queries: {capture.queries.shape}")
    
    print("\n✅ Custom dimensions test PASSED!")


def test_generation():
    """Test text generation."""
    print("\n" + "=" * 60)
    print("TEST 6: Text Generation")
    print("=" * 60)
    
    adapter = MockLlamaAdapter()
    
    prompt = "Hello world"
    output = adapter.generate(prompt, max_new_tokens=10, temperature=1.0)
    
    print(f"\n🎯 Generation:")
    print(f"   • Prompt: '{prompt}'")
    print(f"   • Output: '{output}'")
    
    print("\n✅ Generation test PASSED!")


def main():
    print("\n" + "🚀 OCULI MOCK LLAMA TEST SUITE " + "🚀")
    print("=" * 60)
    
    try:
        capture = test_basic_capture()
        test_selective_capture()
        test_attention_structure()
        test_hooks()
        test_custom_dimensions()
        test_generation()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("\nThe mock LLaMA model is working correctly.")
        print("You can now use it for testing Oculi on your MacBook!")
        print("\nExample usage:")
        print("  from tests.mocks import MockLlamaAdapter")
        print("  adapter = MockLlamaAdapter()")
        print("  capture = adapter.capture(adapter.tokenize('your text'))")
        print()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
