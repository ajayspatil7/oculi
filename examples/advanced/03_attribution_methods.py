"""
Attribution Methods Example
============================

This example demonstrates how to use attribution methods to understand:
- How information flows through the transformer
- Which layers contribute to predictions
- How to decompose contributions into attention vs MLP

Phase 2 Feature ✨
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from oculi.models.llama import LlamaAttentionAdapter
from oculi.analysis import AttributionMethods
import torch

def main():
    print("=" * 70)
    print("Oculi Example: Attribution Methods (Phase 2)")
    print("=" * 70)

    # Using mock model for testing
    print("\n📦 Loading mock model...")
    from tests.mocks import MockLlamaAdapter
    adapter = MockLlamaAdapter()

    # Prepare input
    text = "The cat sat on the mat"
    print(f"\n📝 Input text: '{text}'")
    input_ids = adapter.tokenize(text)

    # Capture full model state
    print("\n🔍 Capturing full model state...")
    full = adapter.capture_full(input_ids)

    print(f"   ✅ Attention: {full.attention is not None}")
    print(f"   ✅ Residual: {full.residual is not None}")
    print(f"   ✅ MLP: {full.mlp is not None}")
    print(f"   ✅ Logits: {full.logits is not None}")

    # === 1. Attention Flow ===
    print("\n" + "=" * 70)
    print("1️⃣  Attention Flow - Track information propagation")
    print("=" * 70)

    flow = AttributionMethods.attention_flow(full.attention, normalize=True)
    print(f"\n📊 Flow shape: {flow.values.shape}  # [L, H, T, T]")

    # Example: How does the first token (BOS) contribute?
    layer, head = 1, 0
    token_from = 0  # BOS token
    token_to = -1   # Last token

    contribution = flow.values[layer, head, token_to, token_from]
    print(f"\n🔗 BOS contribution to final token:")
    print(f"   Layer {layer}, Head {head}: {contribution:.4f}")

    # === 2. Value-Weighted Attention ===
    print("\n" + "=" * 70)
    print("2️⃣  Value-Weighted Attention - Account for value magnitudes")
    print("=" * 70)

    weighted = AttributionMethods.value_weighted_attention(
        full.attention,
        norm_type="l2"
    )
    print(f"\n📊 Weighted attention shape: {weighted.values.shape}  # [L, H, T, T]")

    # Compare raw vs weighted
    layer, head, token = 1, 0, -1
    raw_pattern = full.attention.patterns[layer, head, token, :]
    weighted_pattern = weighted.values[layer, head, token, :]

    diff = (weighted_pattern - raw_pattern).abs().sum()
    print(f"\n⚖️  Difference from raw attention:")
    print(f"   L1 difference: {diff:.4f}")
    print(f"   (How much value magnitudes change the effective attention)")

    # === 3. Direct Logit Attribution ===
    print("\n" + "=" * 70)
    print("3️⃣  Direct Logit Attribution - Which layers matter most?")
    print("=" * 70)

    # Get mock unembed matrix
    vocab_size = 1000
    hidden_dim = full.residual.hidden_dim
    unembed = torch.randn(vocab_size, hidden_dim)

    target_token_id = 100  # Arbitrary target

    dla = AttributionMethods.direct_logit_attribution(
        full.residual,
        unembed,
        target_token_id,
        position=-1
    )

    print(f"\n📊 Attribution shape: {dla.values.shape}  # [L]")

    # Find most important layers
    top_3 = dla.values.abs().topk(3)
    print(f"\n🏆 Top 3 most important layers:")
    for i, (score, layer) in enumerate(zip(top_3.values, top_3.indices)):
        sign = "+" if score > 0 else "-"
        print(f"   {i+1}. Layer {layer.item():2d}: {sign}{abs(score.item()):.4f}")

    # === 4. Component Attribution ===
    print("\n" + "=" * 70)
    print("4️⃣  Component Attribution - Attention vs MLP")
    print("=" * 70)

    component_attr = AttributionMethods.component_attribution(
        full.residual,
        full.mlp,
        unembed,
        target_token_id,
        position=-1
    )

    print(f"\n📊 Component attribution shape: {component_attr.values.shape}  # [L, 2]")
    print(f"   Dimension 0: Attention contributions")
    print(f"   Dimension 1: MLP contributions")

    attn_contrib = component_attr.values[:, 0]
    mlp_contrib = component_attr.values[:, 1]

    print(f"\n🔍 Layer-wise breakdown:")
    for layer in range(min(5, len(attn_contrib))):  # First 5 layers
        a = attn_contrib[layer].item()
        m = mlp_contrib[layer].item()
        dominant = "🎯 Attn" if abs(a) > abs(m) else "🧠 MLP"
        print(f"   Layer {layer}: Attn={a:+.4f}, MLP={m:+.4f}  {dominant}")

    # === 5. Top Attributions Helper ===
    print("\n" + "=" * 70)
    print("5️⃣  Top Attributions - Extract most important contributors")
    print("=" * 70)

    top_10 = AttributionMethods.top_attributions(dla, k=5)
    print(f"\n🏆 Top 5 layer attributions:")
    for i, (indices, value) in enumerate(top_10):
        layer = indices[0] if isinstance(indices, tuple) else indices
        print(f"   {i+1}. Layer {layer}: {value:+.4f}")

    # === Summary ===
    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    print("\n✅ Attribution methods revealed:")
    print(f"   • How attention flows through {full.attention.n_layers} layers")
    print(f"   • Impact of value magnitudes on effective attention")
    print(f"   • Layer-wise contributions to target logit")
    print(f"   • Attention vs MLP importance per layer")

    print(f"\n💡 Next steps:")
    print(f"   - Try with real model and analyze specific predictions")
    print(f"   - Combine with composition analysis")
    print(f"   - Visualize attribution patterns")
    print(f"   - Use for circuit discovery")

if __name__ == "__main__":
    main()
