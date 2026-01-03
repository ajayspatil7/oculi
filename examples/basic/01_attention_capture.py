"""
Basic Attention Capture Example
================================

This example demonstrates how to capture attention data from a transformer model.

What you'll learn:
- Loading a model and creating an adapter
- Basic attention capture
- Understanding capture shapes
- Working with Q/K/V vectors and attention patterns
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from oculi.models.llama import LlamaAttentionAdapter

def main():
    print("=" * 60)
    print("Oculi Example: Basic Attention Capture")
    print("=" * 60)

    # For testing on CPU, use mock model
    # Uncomment for real model (requires GPU):
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

    # Using mock model for CPU testing
    print("\n📦 Loading mock model for testing...")
    from tests.mocks import MockLlamaAdapter
    adapter = MockLlamaAdapter()

    # Prepare input
    text = "The quick brown fox jumps over the lazy dog"
    print(f"\n📝 Input text: '{text}'")

    input_ids = adapter.tokenize(text)
    print(f"   Tokenized: {input_ids.shape}")

    # Capture attention
    print("\n🔍 Capturing attention...")
    capture = adapter.capture(input_ids)

    # Inspect captured data
    print(f"\n✅ Capture successful!")
    print(f"\n📊 Captured Shapes:")
    print(f"   Queries:  {capture.queries.shape}  # [layers, heads, tokens, head_dim]")
    print(f"   Keys:     {capture.keys.shape}  # [layers, kv_heads, tokens, head_dim]")
    print(f"   Values:   {capture.values.shape}  # [layers, kv_heads, tokens, head_dim]")
    print(f"   Patterns: {capture.patterns.shape}  # [layers, heads, tokens, tokens]")

    # Model metadata
    print(f"\n🏗️  Model Architecture:")
    print(f"   Layers: {capture.n_layers}")
    print(f"   Heads: {capture.n_heads}")
    print(f"   KV Heads: {capture.n_kv_heads}")
    print(f"   Head Dim: {capture.head_dim}")
    print(f"   GQA: {capture.is_gqa} (ratio: {capture.gqa_ratio}:1)")

    # Examine attention patterns
    print(f"\n👀 Attention Pattern at Layer 0, Head 0:")
    layer, head = 0, 0
    pattern = capture.patterns[layer, head]

    print(f"   Shape: {pattern.shape}  # [tokens, tokens]")
    print(f"   Sum per row: {pattern.sum(dim=-1)[:5]}")  # Should be ~1.0
    print(f"   (Each row sums to 1.0 due to softmax)")

    # Causal masking check
    print(f"\n🔒 Causal Masking Check:")
    upper_triangle = pattern.triu(diagonal=1)  # Above diagonal
    print(f"   Upper triangle sum: {upper_triangle.sum():.6f}")
    print(f"   (Should be ~0.0 for causal attention)")

    print(f"\n✅ Example complete!")
    print(f"\n💡 Next steps:")
    print(f"   - Try with real model (uncomment lines above)")
    print(f"   - Explore selective capture (CaptureConfig)")
    print(f"   - Check out entropy analysis example")

if __name__ == "__main__":
    main()
