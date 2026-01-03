# Attribution Methods

Attribution methods help you understand **how information flows** through the transformer and **which components contribute** to specific outputs.

## Overview

Oculi provides five attribution methods:

1. **Attention Flow** - Track information propagation through attention layers
2. **Value-Weighted Attention** - Account for value vector magnitudes
3. **Direct Logit Attribution** - Measure layer contributions to target logits
4. **Component Attribution** - Decompose into attention vs MLP contributions
5. **Head Attribution** - Per-head contribution to target logits

## Prerequisites

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from oculi.models.llama import LlamaAttentionAdapter
from oculi.analysis import AttributionMethods

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
adapter = LlamaAttentionAdapter(model, tokenizer)

# Capture full model state
text = "The cat sat on the mat"
input_ids = tokenizer.encode(text, return_tensors="pt")
full = adapter.capture_full(input_ids)
```

## Attention Flow

**Purpose:** Track how information from source positions propagates to target positions through attention across layers.

**Method:** Compose attention patterns across layers via matrix multiplication.

```python
flow = AttributionMethods.attention_flow(full.attention, normalize=True)

print(f"Flow shape: {flow.values.shape}")  # [L, H, T, T]

# flow.values[l, h, i, j] = cumulative attention from position j to i at layer l
# This tells you: "How much does token j contribute to token i's representation at layer l?"

# Example: Track how token 0 (BOS) contributes across layers
layer = 20
head = 5
token_from = 0
token_to = -1  # Last token

contribution = flow.values[layer, head, token_to, token_from]
print(f"BOS contribution to final token at L{layer}H{head}: {contribution:.4f}")
```

**Interpretation:**
- High flow value → Strong cumulative attention path
- Low flow value → Weak or indirect attention path
- Diagonal values → Self-attention strength

## Value-Weighted Attention

**Purpose:** Correct attention patterns by accounting for value vector magnitudes. High attention doesn't always mean high contribution if values are small.

```python
weighted_attn = AttributionMethods.value_weighted_attention(
    full.attention,
    norm_type="l2"  # Options: "l2", "l1", "linf"
)

print(f"Weighted attention: {weighted_attn.values.shape}")  # [L, H, T, T]

# Compare raw vs value-weighted attention
layer, head, token = 20, 5, -1
raw_attn = full.attention.patterns[layer, head, token, :]
weighted = weighted_attn.values[layer, head, token, :]

print(f"Raw attention sum: {raw_attn.sum():.4f}")  # Should be ~1.0
print(f"Weighted attention sum: {weighted.sum():.4f}")  # Should be ~1.0
print(f"Difference: {(weighted - raw_attn).abs().sum():.4f}")
```

**Use Cases:**
- Identifying truly important tokens (high attention × high value magnitude)
- Detecting "attention sinks" (high attention but low contribution)
- Value norm analysis by attention head

## Direct Logit Attribution

**Purpose:** Measure how much each layer's residual stream contribution affects a specific target token's logit.

```python
# Which layers are most important for predicting "mat"?
target_token_id = tokenizer.encode("mat")[0]
unembed = model.lm_head.weight

dla = AttributionMethods.direct_logit_attribution(
    full.residual,
    unembed,
    target_token_id,
    position=-1  # Last position
)

print(f"Layer attributions: {dla.values.shape}")  # [L]

# Find most important layers
top_layers = dla.values.abs().topk(5)
for i, (score, layer) in enumerate(zip(top_layers.values, top_layers.indices)):
    print(f"{i+1}. Layer {layer.item()}: {score.item():.4f}")
```

**Interpretation:**
- Positive values → Layer increases target logit
- Negative values → Layer decreases target logit
- Large magnitude → Strong causal effect
- Small magnitude → Weak or negligible effect

**Typical Patterns:**
- Early layers often have small contributions
- Middle layers build up features
- Late layers make final adjustments

## Component Attribution

**Purpose:** Decompose each layer's contribution into attention vs MLP components.

```python
component_attr = AttributionMethods.component_attribution(
    full.residual,
    full.mlp,
    unembed,
    target_token_id,
    position=-1
)

print(f"Component attribution: {component_attr.values.shape}")  # [L, 2]

# Extract attention and MLP contributions
attn_contrib = component_attr.values[:, 0]
mlp_contrib = component_attr.values[:, 1]

# Analyze contribution patterns
for layer in range(len(attn_contrib)):
    a = attn_contrib[layer].item()
    m = mlp_contrib[layer].item()
    dominant = "Attention" if abs(a) > abs(m) else "MLP"
    print(f"Layer {layer:2d}: Attn={a:+.3f}, MLP={m:+.3f} ({dominant})")
```

**Use Cases:**
- Understanding attention vs MLP roles per layer
- Identifying "attention-heavy" vs "MLP-heavy" layers
- Debugging why certain layers matter
- Circuit analysis (which components are involved)

## Head Attribution

**Purpose:** Granular per-head attribution to target logits.

```python
# Need output projection weights
output_weights = model.model.layers[0].self_attn.o_proj.weight

head_attr = AttributionMethods.head_attribution(
    full.attention,
    output_weights,
    unembed,
    target_token_id,
    position=-1
)

print(f"Head attribution: {head_attr.values.shape}")  # [L, H]

# Get top contributing heads
top_heads = AttributionMethods.top_attributions(head_attr, k=10)
for (layer, head), score in top_heads:
    print(f"L{layer}H{head}: {score:.4f}")
```

**Note:** Head attribution requires model's output projection weights. Implementation may vary by model architecture.

## Combining Methods

Use multiple attribution methods together for comprehensive analysis:

```python
# 1. Find important layers
dla = AttributionMethods.direct_logit_attribution(...)
important_layers = dla.values.abs().topk(3).indices

# 2. Decompose those layers into components
for layer in important_layers:
    component_attr = AttributionMethods.component_attribution(...)
    attn_score = component_attr.values[layer, 0]
    mlp_score = component_attr.values[layer, 1]
    print(f"Layer {layer}: Attn={attn_score:.3f}, MLP={mlp_score:.3f}")

# 3. If attention dominates, look at head-level
if attn_score > mlp_score:
    head_attr = AttributionMethods.head_attribution(...)
    layer_heads = head_attr.values[layer, :]
    top_head = layer_heads.argmax()
    print(f"  → Dominated by Head {top_head}")
```

## Top Attributions Helper

Extract top-k attributions with indices:

```python
result = AttributionMethods.direct_logit_attribution(...)

# Get top 10 by absolute value
top_10 = AttributionMethods.top_attributions(result, k=10)

for indices, value in top_10:
    print(f"Index {indices}: {value:.4f}")
```

Works with any `AttributionResult` regardless of shape.

## Mathematical Background

### Attention Flow
```
flow[0] = attn[0]  # Layer 0: direct attention
flow[l] = attn[l] @ flow[l-1]  # Subsequent layers: composition
```

### Value-Weighted Attention
```
weighted[t_q, t_k] = attn[t_q, t_k] * ||V[t_k]||
```

### Direct Logit Attribution
```
delta[L] = residual[L] - residual[L-1]
attribution[L] = delta[L] @ unembed[target_token]
```

### Component Attribution
```
attn_delta[L] = post_attn[L] - pre_attn[L]
mlp_delta[L] = post_mlp[L] - pre_mlp[L]
attribution = {attn_delta, mlp_delta} @ unembed[target]
```

## Best Practices

1. **Always capture full state** for attribution:
   ```python
   full = adapter.capture_full(input_ids)
   ```

2. **Use absolute values** for importance ranking:
   ```python
   important = attribution.values.abs().topk(k)
   ```

3. **Normalize for comparison**:
   ```python
   normalized = attribution.values / attribution.values.abs().sum()
   ```

4. **Combine multiple methods** for complete picture

5. **Visualize results** for interpretation

## Next Steps

- [Composition Analysis](composition-analysis.md) - Analyze head interactions
- [Logit Lens](logit-lens.md) - Track predictions across layers
- [Circuit Detection](circuit-detection.md) - Identify canonical circuits
- [API Reference](../api-reference/analysis.md) - Complete API documentation
