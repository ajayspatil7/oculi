# 📘 Oculi: The Definitive Guide to Mechanistic Interpretability

> _Understanding the Inner Workings of Large Language Models_

---

**Author:** Ajay S Patil
**Version:** 0.6.0
**Model:** `meta-llama/Llama-3.2-3B-Instruct`

---

## Table of Contents

1. [Preface: Why Mechanistic Interpretability?](#preface-why-mechanistic-interpretability)
2. [Chapter 1: Getting Started](#chapter-1-getting-started)
3. [Chapter 2: Attention — The Heart of Transformers](#chapter-2-attention)
4. [Chapter 3: The Residual Stream](#chapter-3-the-residual-stream)
5. [Chapter 4: MLP Internals](#chapter-4-mlp-internals)
6. [Chapter 5: Logit Lens](#chapter-5-logit-lens)
7. [Chapter 6: Circuit Detection](#chapter-6-circuit-detection)
8. [Chapter 7: Attribution Methods](#chapter-7-attribution-methods)
9. [Chapter 8: Head Composition](#chapter-8-head-composition)
10. [Chapter 9: Interventions](#chapter-9-interventions)
11. [Chapter 10: Activation Patching](#chapter-10-activation-patching)
12. [Appendix A: Tensor Shapes](#appendix-a-tensor-shapes)
13. [Appendix B: Troubleshooting](#appendix-b-troubleshooting)

---

# Preface: Why Mechanistic Interpretability?

## The Black Box Problem

Large Language Models are remarkable—they write code, answer questions, generate human-like text. But here's the unsettling truth: **we don't fully understand how they work.**

When a model solves a coding problem, _what computations_ led to that answer? When it hallucinates, _which circuits_ failed? These questions matter for:

- **Safety**: Ensuring models behave as intended
- **Debugging**: Understanding why models fail
- **Capability Elicitation**: Unlocking hidden abilities
- **Trust**: Building confidence in AI systems

## What is Mechanistic Interpretability?

Mechanistic interpretability (mech interp) is **reverse engineering neural networks**. Just as a software engineer decompiles a binary to understand its logic, we dissect neural networks to understand their algorithms.

Key questions:

- What features do individual neurons represent?
- How do attention heads route information?
- What circuits implement specific behaviors?
- How do predictions form layer by layer?

## Why Oculi?

Oculi is a **surgical toolkit** for transformer interpretability:

- **Direct Access**: Raw Q/K/V vectors, attention patterns, activations
- **Minimal Overhead**: Efficient hooks, selective capture
- **Explicit Control**: No magic—you decide what to capture
- **Research-Ready**: Built for serious mechanistic analysis

---

# Chapter 1: Getting Started

## What You'll Learn

- Install and configure Oculi
- Load models correctly for interpretability
- Perform your first capture
- Understand core data structures

## Why This Matters

Before analyzing a model, you need to **instrument** it. Oculi's adapter pattern wraps your model with hooks that capture internal states without modifying the model.

## Installation

```bash
git clone https://github.com/ajayspatil7/oculi.git
cd oculi
pip install -e ".[all]"
```

## 🔬 Hands-On: Your First Capture

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from oculi.models.llama import LlamaAttentionAdapter
from oculi.utils import get_default_device, get_device_info

# Step 1: Check available device (MPS/CUDA/CPU)
device = get_default_device()
print(get_device_info())  # Shows device capabilities

# Step 2: Load model with memory-efficient settings
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    low_cpu_mem_usage=True,
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# Step 3: Create adapter
adapter = LlamaAttentionAdapter(model, tokenizer)

# Step 4: Prepare input
text = "The capital of France is"
input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

# Step 5: Capture attention internals
capture = adapter.capture(input_ids)

# Step 6: Inspect shapes
print(f"Queries: {capture.queries.shape}")   # [L, H, T, D]
print(f"Keys: {capture.keys.shape}")         # [L, H_kv, T, D]
print(f"Values: {capture.values.shape}")     # [L, H_kv, T, D]
print(f"Patterns: {capture.patterns.shape}") # [L, H, T, T]
```

**Expected Output:**

```
Queries: torch.Size([28, 24, 6, 128])   # 28 layers, 24 heads, 6 tokens
Keys: torch.Size([28, 8, 6, 128])       # 8 KV heads (GQA)
Values: torch.Size([28, 8, 6, 128])
Patterns: torch.Size([28, 24, 6, 6])    # Attention matrix per head
```

## Tensor Shape Reference

| Tensor     | Shape             | Description                          |
| ---------- | ----------------- | ------------------------------------ |
| `queries`  | `[L, H, T, D]`    | Query vectors per layer, head, token |
| `keys`     | `[L, H_kv, T, D]` | Key vectors (fewer heads in GQA)     |
| `values`   | `[L, H_kv, T, D]` | Value vectors                        |
| `patterns` | `[L, H, T, T]`    | Attention weights (rows sum to 1)    |

---

# Chapter 2: Attention

## What You'll Learn

- How attention works mathematically
- Interpreting attention patterns
- Analyzing query and key vectors
- Finding what tokens attend to

## Why Attention Matters

Attention is the **information routing mechanism**. It determines:

- Which tokens influence which outputs
- How context is aggregated
- What the model "looks at" when predicting

Understanding attention reveals **what the model considers important**.

## The Math

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I contain?"
- **V (Value)**: "What information do I pass along?"
- **Pattern**: The softmax result—probability distribution over tokens

## 🔬 Hands-On: Visualizing Attention

```python
import matplotlib.pyplot as plt

# Get attention pattern for layer 15, head 0
layer, head = 15, 0
pattern = capture.patterns[layer, head].cpu().numpy()

# Get tokens for labels
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(pattern, cmap='Blues')
plt.colorbar(label='Attention Weight')
plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
plt.yticks(range(len(tokens)), tokens)
plt.xlabel('Key (attending to)')
plt.ylabel('Query (attending from)')
plt.title(f'Attention Pattern: Layer {layer}, Head {head}')
plt.tight_layout()
plt.show()
```

## 🔬 Hands-On: Finding High-Attention Pairs

```python
# Which token pairs have strongest attention?
pattern = capture.patterns[15, 0]

# Find max attention for each query position
max_attn, max_idx = pattern.max(dim=-1)

for q_pos in range(len(tokens)):
    k_pos = max_idx[q_pos].item()
    weight = max_attn[q_pos].item()
    print(f"'{tokens[q_pos]}' → '{tokens[k_pos]}' (weight: {weight:.3f})")
```

## 🔬 Hands-On: Attention Entropy

Low entropy = focused attention (few tokens)  
High entropy = diffuse attention (many tokens)

```python
from oculi.analysis import EntropyAnalysis

entropy = EntropyAnalysis.token_entropy(capture)
print(f"Entropy shape: {entropy.shape}")  # [L, H, T]

# Find most focused heads
mean_entropy = entropy.mean(dim=-1)  # [L, H]
flat = mean_entropy.flatten()
top_k = torch.topk(flat, k=5, largest=False)

print("\nMost Focused Heads:")
for idx, val in zip(top_k.indices, top_k.values):
    layer = idx // capture.n_heads
    head = idx % capture.n_heads
    print(f"  L{layer}H{head}: entropy = {val:.3f}")
```

---

# Chapter 3: The Residual Stream

## What You'll Learn

- Understanding residual stream architecture
- Capturing activations at intervention points
- Analyzing how information accumulates

## Why It Matters

The residual stream is the **highway** of information. Each layer reads from and writes to this shared stream:

```
x₀ → [+Attn₁] → [+MLP₁] → [+Attn₂] → [+MLP₂] → ... → xₙ
```

**Intervention Points:**

- `pre_attn`: Before attention (layer input)
- `post_attn`: After attention, before MLP
- `pre_mlp`: Before MLP
- `post_mlp`: After MLP (layer output)

## 🔬 Hands-On: Capturing Residual Stream

```python
from oculi import ResidualConfig

config = ResidualConfig(
    layers=[10, 15, 20],
    capture_pre_attn=True,
    capture_post_attn=True,
    capture_pre_mlp=True,
    capture_post_mlp=True,
)

residual = adapter.capture_residual(input_ids, config=config)

print(f"Pre-attention: {residual.pre_attn.shape}")   # [3, T, H]
print(f"Post-attention: {residual.post_attn.shape}") # [3, T, H]
print(f"Post-MLP: {residual.post_mlp.shape}")        # [3, T, H]
```

## 🔬 Hands-On: Measuring Layer Contributions

```python
# How much does each layer change the residual stream?
layer_deltas = []

for i in range(len(config.layers)):
    pre = residual.pre_attn[i]
    post = residual.post_mlp[i]
    delta = (post - pre).norm(dim=-1).mean()
    layer_deltas.append(delta.item())

print("Layer contribution (L2 norm):")
for layer, delta in zip(config.layers, layer_deltas):
    print(f"  Layer {layer}: {delta:.2f}")
```

---

# Chapter 4: MLP Internals

## What You'll Learn

- Understanding MLP architecture
- Capturing gate, up, and down projections
- Analyzing neuron activations

## Why MLP Matters

MLPs are the **memory** of transformers. While attention routes information, MLPs:

- Store factual knowledge
- Perform non-linear transformations
- Implement learned "lookup tables"

Research shows individual neurons encode specific concepts!

## LLaMA MLP Architecture

```
           ┌─→ gate_proj ─→ SiLU ─┐
hidden ────┤                      ├─→ multiply ─→ down_proj ─→ output
           └─→ up_proj ───────────┘
```

## 🔬 Hands-On: Capturing MLP Activations

```python
from oculi import MLPConfig

config = MLPConfig(
    layers=[15, 20, 25],
    capture_gate=True,
    capture_up=True,
    capture_post_activation=True,
    capture_output=True,
)

mlp = adapter.capture_mlp(input_ids, config=config)

print(f"Gate projection: {mlp.gate.shape}")
print(f"Post-activation: {mlp.post_activation.shape}")
print(f"Output: {mlp.output.shape}")
```

## 🔬 Hands-On: Finding Active Neurons

```python
# Which neurons fire most strongly?
layer_idx = 1  # Second captured layer
token_idx = -1  # Last token

activations = mlp.post_activation[layer_idx, token_idx]

# Top-k most active neurons
top_k = torch.topk(activations.abs(), k=10)

print(f"Top 10 active neurons:")
for idx, val in zip(top_k.indices, top_k.values):
    print(f"  Neuron {idx.item()}: {val.item():.3f}")
```

---

# Chapter 5: Logit Lens

## What You'll Learn

- The logit lens technique
- Tracking predictions across layers
- When does the model "know" the answer?

## Why It Matters

Logit lens answers: **"If we stopped at layer L, what would the model predict?"**

This reveals:

- When factual knowledge is retrieved
- How predictions refine through layers
- Which layers matter for specific predictions

## 🔬 Hands-On: Layer-by-Layer Predictions

```python
from oculi import LogitConfig
from oculi.analysis import LogitLensAnalysis

config = LogitConfig(layers=None, top_k=10)
logits = adapter.capture_logits(input_ids, config=config)

lens = LogitLensAnalysis(tokenizer)
predictions = lens.layer_predictions(logits, token_position=-1, top_k=3)

print("Predictions at each layer:\n")
for pred in predictions[::4]:  # Every 4th layer
    layer = pred['layer']
    top = pred['predictions'][:3]
    print(f"Layer {layer:2d}: {top}")
```

## 🔬 Hands-On: Token Probability Trajectory

```python
# When does "Paris" become the prediction?
target = tokenizer.encode("Paris", add_special_tokens=False)[0]
trajectory = lens.token_probability_trajectory(logits, target)

import matplotlib.pyplot as plt
plt.plot(trajectory.cpu().numpy())
plt.xlabel('Layer')
plt.ylabel('P("Paris")')
plt.title('When Does the Model Know?')
plt.show()
```

---

# Chapter 6: Circuit Detection

## What You'll Learn

- Canonical circuit types
- Automatic detection methods
- Validating hypotheses

## Why It Matters

Transformers implement **algorithms** via circuits. Famous circuits:

- **Induction Heads**: Copy patterns (A B ... A → B)
- **Previous Token Heads**: Attend to t-1
- **Positional Heads**: Attend to BOS, recent tokens

## 🔬 Hands-On: Detecting Induction Heads

```python
from oculi.analysis import CircuitDetection

induction = CircuitDetection.detect_induction_heads(capture, threshold=0.3)
mask = induction > 0.3

print(f"Found {mask.sum().item()} induction heads:\n")
for layer in range(capture.n_layers):
    for head in range(capture.n_heads):
        if mask[layer, head]:
            print(f"  L{layer}H{head}: {induction[layer, head]:.3f}")
```

## 🔬 Hands-On: Previous Token Heads

```python
prev_token = CircuitDetection.detect_previous_token_heads(capture, threshold=0.5)
mask = prev_token > 0.5

print(f"Previous token heads: {mask.sum().item()}")
```

## 🔬 Hands-On: Classify All Heads

```python
for layer in [0, 10, 20]:
    for head in range(4):
        result = CircuitDetection.classify_attention_head(capture, layer, head)
        if result['pattern'] != 'unknown':
            print(f"L{layer}H{head}: {result['pattern']} ({result['score']:.2f})")
```

---

# Chapter 7: Attribution Methods

## What You'll Learn

- Direct logit attribution
- Component attribution
- Finding important heads/layers

## Why It Matters

Attribution answers: **"Which components caused this output?"**

Essential for:

- Debugging model behavior
- Understanding factual recall
- Identifying important components

## 🔬 Hands-On: Direct Logit Attribution

```python
from oculi.analysis import AttributionMethods

full = adapter.capture_full(input_ids)

target_id = tokenizer.encode("Paris", add_special_tokens=False)[0]
unembed = model.lm_head.weight

dla = AttributionMethods.direct_logit_attribution(
    full.residual, unembed, target_id, position=-1
)

print("Layer contributions to 'Paris':")
for layer, contrib in enumerate(dla.values):
    bar = "█" * int(abs(contrib) * 20)
    print(f"L{layer:2d}: {contrib:+.4f} {bar}")
```

## 🔬 Hands-On: Component Attribution

```python
comp = AttributionMethods.component_attribution(
    full.residual, full.mlp, unembed, target_id
)

print("\nAttention vs MLP:")
for layer in range(min(10, len(comp.values))):
    attn = comp.values[layer, 0]
    mlp = comp.values[layer, 1]
    print(f"L{layer:2d}: Attn={attn:+.3f}, MLP={mlp:+.3f}")
```

---

# Chapter 8: Head Composition

## What You'll Learn

- QK and OV composition
- Virtual attention through paths
- Finding composed head pairs

## Why It Matters

Heads don't work alone—they **compose**. Understanding composition reveals:

- Multi-hop reasoning paths
- Information flow
- Why certain heads matter together

## 🔬 Hands-On: QK Composition

```python
from oculi.analysis import CompositionAnalysis

qk = CompositionAnalysis.qk_composition(
    capture,
    source=(10, 5),
    target=(20, 3)
)
print(f"QK composition: {qk.values.mean():.4f}")
```

## 🔬 Hands-On: Induction Circuit Detection

```python
circuits = CompositionAnalysis.detect_induction_circuit(capture, threshold=0.3)

print(f"Found {len(circuits.metadata['circuits'])} circuits:\n")
for c in circuits.metadata['circuits'][:5]:
    prev = c['previous_token_head']
    ind = c['induction_head']
    score = c['composition_score']
    print(f"  L{prev[0]}H{prev[1]} → L{ind[0]}H{ind[1]} ({score:.3f})")
```

---

# Chapter 9: Interventions

## What You'll Learn

- Q/K scaling (Spectra method)
- Head ablation
- Interpreting results

## Why It Matters

Correlation ≠ causation. Interventions answer: **"What happens if we modify this?"**

## 🔬 Hands-On: Sharpening Attention

```python
from oculi.intervention import SpectraScaler, InterventionContext

scaler = SpectraScaler(layer=23, head=5, alpha=1.5)  # α>1 sharpens

prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

with InterventionContext(adapter, [scaler]):
    output = model.generate(input_ids, max_new_tokens=10)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 🔬 Hands-On: Head Ablation

```python
from oculi.intervention import HeadAblation

ablation = HeadAblation(layer=20, head=3)

with InterventionContext(adapter, [ablation]):
    output = model.generate(input_ids, max_new_tokens=10)

print(f"Without L20H3: {tokenizer.decode(output[0])}")
```

---

# Chapter 10: Activation Patching

## What You'll Learn

- What is activation patching and why it matters
- Causal intervention methodology
- Manual patching with PatchingContext
- Systematic tracing with CausalTracer
- Recovery score interpretation

## Why It Matters

**The Gold Standard for Causality**

Correlation ≠ Causation. Just because a component activates doesn't mean it's *necessary* for the output.

Activation patching answers: **"Is this component causally responsible for this behavior?"**

**The Method:**
1. **Clean run**: Model on correct input → save activations
2. **Corrupted run**: Model on incorrect input → wrong output
3. **Patch**: Replace corrupted activations with clean ones
4. **Measure**: Does output recover to clean behavior?

If patching component X recovers the correct output, X is **causally necessary**.

## Real-World Applications

- **Circuit Discovery**: Find minimal circuits for behaviors
- **Debugging**: Pinpoint where models fail
- **Safety**: Identify components responsible for harmful outputs
- **Factual Recall**: Track where facts are stored (ROME, Meng et al. 2022)

---

## The Paradigm

```
Clean Input:    "The capital of France is" → "Paris" ✓
Corrupted:      "The capital of Poland is" → "Warsaw"

Question: Which layer/component causes the difference?

Method: Patch clean activations → corrupted run

Layer 5 MLP  → Recovery: 0.1  (not important)
Layer 15 MLP → Recovery: 0.8  (important!)
Layer 25 MLP → Recovery: 0.3  (somewhat important)
```

**Recovery Score:**
```
recovery = (metric_patched - metric_corrupted) / (metric_clean - metric_corrupted)
```

- **1.0** = Full recovery (component fully explains behavior)
- **0.0** = No recovery (component doesn't matter)
- **>1.0** = Over-recovery (patching overcompensated)
- **<0.0** = Negative (patching made it worse)

---

## 🔬 Hands-On: Manual Patching

```python
from oculi.intervention import PatchConfig, ActivationPatch, PatchingContext

# Step 1: Define your inputs
clean_text = "The Eiffel Tower is in Paris"
corrupt_text = "The Eiffel Tower is in London"

clean_ids = tokenizer.encode(clean_text, return_tensors="pt").to(device)
corrupt_ids = tokenizer.encode(corrupt_text, return_tensors="pt").to(device)

# Step 2: Capture clean activations
clean_capture = adapter.capture_full(clean_ids)

# Step 3: Create a patch
patch = ActivationPatch(
    config=PatchConfig(layer=20, component='mlp_out'),
    source_activation=clean_capture.mlp.output[20]  # From clean run
)

# Step 4: Apply patch during corrupted run
with PatchingContext(adapter, [patch]):
    patched_output = model(corrupt_ids)
    patched_logits = patched_output.logits

# Step 5: Compare
clean_logits = model(clean_ids).logits
corrupt_logits = model(corrupt_ids).logits

target_token = tokenizer.encode("Paris", add_special_tokens=False)[0]

clean_prob = torch.softmax(clean_logits[0, -1], dim=-1)[target_token].item()
corrupt_prob = torch.softmax(corrupt_logits[0, -1], dim=-1)[target_token].item()
patched_prob = torch.softmax(patched_logits[0, -1], dim=-1)[target_token].item()

print(f"P('Paris' | clean):    {clean_prob:.3f}")
print(f"P('Paris' | corrupt):  {corrupt_prob:.3f}")
print(f"P('Paris' | patched):  {patched_prob:.3f}")

recovery = (patched_prob - corrupt_prob) / (clean_prob - corrupt_prob)
print(f"\nRecovery: {recovery:.2f}")
```

---

## 🔬 Hands-On: Systematic Causal Tracing

Instead of manually testing each component, use `CausalTracer` to sweep all layers:

```python
from oculi.intervention import CausalTracer

tracer = CausalTracer(adapter)

# Define metric function
target_token = tokenizer.encode("Paris", add_special_tokens=False)[0]

def metric_fn(logits):
    """Get probability of target token."""
    probs = torch.softmax(logits[0, -1], dim=-1)
    return probs[target_token].item()

# Run systematic sweep
results = tracer.trace(
    clean_input=clean_ids,
    corrupted_input=corrupt_ids,
    metric_fn=metric_fn,
    layers=range(20, 28),         # Sweep layers 20-27
    components=['mlp_out', 'attn_out'],
    verbose=True,
)

print(results.summary())
```

**Output:**
```
Clean metric: 0.9234
Corrupted metric: 0.0123
Difference: 0.9111

Patching L20 mlp_out... recovery=0.12
Patching L20 attn_out... recovery=0.05
Patching L21 mlp_out... recovery=0.34
Patching L21 attn_out... recovery=0.18
Patching L22 mlp_out... recovery=0.87  ← Important!
...
```

---

## 🔬 Hands-On: Analyzing Results

```python
# Get top 5 most important components
top_5 = results.top_results(5)

for result in top_5:
    print(f"L{result.config.layer} {result.config.component}: "
          f"recovery={result.recovery:.2f}")
```

**Output:**
```
L22 mlp_out: recovery=0.87
L23 attn_out: recovery=0.72
L24 mlp_out: recovery=0.65
L21 mlp_out: recovery=0.34
L25 mlp_out: recovery=0.28
```

**Interpretation**: Layer 22 MLP is **causally critical** for this fact!

---

## 🔬 Hands-On: Recovery Heatmap

```python
import matplotlib.pyplot as plt

# Get recovery matrix [layers × components]
matrix = results.recovery_matrix()

plt.figure(figsize=(10, 6))
plt.imshow(matrix.numpy(), cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(label='Recovery Score')
plt.xlabel('Component')
plt.ylabel('Layer')
plt.xticks(range(len(results.components)), results.components, rotation=45)
plt.yticks(range(len(results.layers)), results.layers)
plt.title('Causal Importance Heatmap')
plt.tight_layout()
plt.show()
```

This shows which layers/components are **causally responsible**.

---

## 🔬 Hands-On: Component Types

Oculi supports patching 6 different intervention points:

```python
# Residual stream points
PatchConfig(layer=20, component='residual_pre_attn')   # Layer input
PatchConfig(layer=20, component='residual_post_attn')  # After attention
PatchConfig(layer=20, component='residual_post_mlp')   # After MLP

# Component outputs
PatchConfig(layer=20, component='attn_out')  # Attention output
PatchConfig(layer=20, component='mlp_out')   # MLP output

# Single head
PatchConfig(layer=20, component='head', head=5)  # Specific attention head
```

---

## 🔬 Hands-On: Token-Level Patching

Patch only specific token positions:

```python
# Only patch positions 5, 6, 7
patch = ActivationPatch(
    config=PatchConfig(
        layer=20,
        component='mlp_out',
        tokens=[5, 6, 7]  # Only these positions
    ),
    source_activation=clean_capture.mlp.output[20]
)

# Useful for finding which tokens are causally important
results = tracer.trace(
    clean_input=clean_ids,
    corrupted_input=corrupt_ids,
    metric_fn=metric_fn,
    layers=[22],
    components=['mlp_out'],
    tokens=[5],  # Only patch subject token
)
```

---

## 🔬 Hands-On: Multiple Patches

Apply multiple patches simultaneously:

```python
patches = [
    ActivationPatch(
        PatchConfig(layer=20, component='mlp_out'),
        clean_capture.mlp.output[20]
    ),
    ActivationPatch(
        PatchConfig(layer=21, component='attn_out'),
        clean_capture.attention.attn_out[21]  # If captured
    ),
]

with PatchingContext(adapter, patches):
    output = model(corrupt_ids)

# Test if combination of components is necessary
```

---

## Advanced: Finding Circuits

Combine with attribution to find **minimal circuits**:

1. **Attribution** → Find components that *correlate* with output
2. **Patching** → Verify which are *causally necessary*
3. **Composition** → Find connections between components

```python
# Step 1: Attribution
from oculi.analysis import AttributionMethods

dla = AttributionMethods.direct_logit_attribution(...)
top_layers = dla.top_k(5)  # Most activated layers

# Step 2: Causal Verification
results = tracer.trace(
    clean_input=clean_ids,
    corrupted_input=corrupt_ids,
    metric_fn=metric_fn,
    layers=top_layers,  # Only test high-attribution layers
    components=['mlp_out', 'attn_out'],
)

# Step 3: Find causally necessary components
causal_components = [r for r in results.results if r.recovery > 0.7]

print("Minimal Circuit Components:")
for comp in causal_components:
    print(f"  L{comp.config.layer} {comp.config.component}")
```

---

## Key Insights

1. **Localization**: Facts are often stored in specific MLP layers
2. **Attention Routes**: Attention heads route information, MLPs store it
3. **Late Layers**: Factual recall often happens in later layers (20-28 for LLaMA-3)
4. **Redundancy**: Multiple components can contribute (recovery can sum > 1.0)

---

## Common Pitfalls

**1. Metric Selection**

Bad: `lambda logits: logits[0, -1, 0].item()`  (arbitrary token)
Good: `lambda logits: probs[target_token].item()` (meaningful token)

**2. Input Selection**

- Clean and corrupted should differ in **only one fact**
- Too different → recovery scores uninterpretable

**3. Device Compatibility**

```python
# Ensure activations on same device as model
clean_capture = adapter.capture_full(clean_ids.to(device))
```

---

## Published Results Replicated

**ROME (Rank-One Model Editing)**
- Factual associations stored in mid-late MLP layers
- Editing layer 20-24 MLPs changes facts

**IOI (Indirect Object Identification)**
- Specific attention heads implement name mover circuit
- Patching heads 9.9, 10.0 disrupts behavior

**Induction Circuits**
- Layers 0-2: Previous token heads
- Layers 5-8: Induction heads
- Patching confirms causal role

---

## Summary

Activation patching is **the gold standard** for:
- Proving causality (not just correlation)
- Finding minimal circuits
- Debugging model failures
- Validating mechanistic hypotheses

**Key API:**
- `PatchConfig` → What to patch
- `ActivationPatch` → Patch specification
- `PatchingContext` → Apply patches
- `CausalTracer` → Systematic experiments

**Next**: Appendix A shows all tensor shapes for patching.

---

# Appendix A: Tensor Shapes

| Structure        | Field           | Shape           | Description        |
| ---------------- | --------------- | --------------- | ------------------ |
| AttentionCapture | queries         | [L, H, T, D]    | Query vectors      |
|                  | keys            | [L, H_kv, T, D] | Key vectors        |
|                  | patterns        | [L, H, T, T]    | Attention weights  |
| ResidualCapture  | pre_attn        | [L, T, H]       | Before attention   |
|                  | post_mlp        | [L, T, H]       | After MLP          |
| MLPCapture       | post_activation | [L, T, I]       | Neuron activations |
| LogitCapture     | top_k_indices   | [L, T, K]       | Top-K tokens       |

**L**=layers, **H**=heads, **T**=tokens, **D**=head dim, **I**=intermediate

---

# Appendix B: Troubleshooting

## CUDA Out of Memory

```python
# Capture fewer layers
config = CaptureConfig(layers=[20, 21, 22])

# Skip values
config = CaptureConfig(capture_values=False)

# Use top-k for logits
config = LogitConfig(top_k=10)
```

## MPS (Apple Silicon) Notes

```python
from oculi.utils import is_mps_available, get_default_device

# Check MPS availability
if is_mps_available():
    print("Running on Apple Silicon GPU!")
    device = get_default_device()  # Returns mps
else:
    device = get_default_device()  # Returns cuda or cpu
```

**MPS Limitations:**

- Some operations fall back to CPU
- Use `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` for memory issues
- FP16 works well on M1/M2/M3/M4

## GQA Confusion

LLaMA uses Grouped Query Attention:

- 24 query heads, 8 KV heads (3:1)
- Each KV head serves 3 query heads
- Oculi handles expansion automatically

---

_Happy Interpreting! 🔬_
