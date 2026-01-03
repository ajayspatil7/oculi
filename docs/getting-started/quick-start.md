# Quick Start

Get started with Oculi in 5 minutes.

## Installation

```bash
git clone https://github.com/ajayspatil7/oculi.git
cd oculi
pip install -e .
```

## Basic Usage

### 1. Load a Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from oculi.models.llama import LlamaAttentionAdapter

# Load model (works with any LLaMA 2/3 model)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Create Oculi adapter
adapter = LlamaAttentionAdapter(model, tokenizer)
```

### 2. Capture Attention

```python
# Prepare input
text = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer.encode(text, return_tensors="pt")

# Capture attention data
capture = adapter.capture(input_ids)

print(f"Queries: {capture.queries.shape}")   # [L, H, T, D]
print(f"Keys: {capture.keys.shape}")         # [L, H_kv, T, D]
print(f"Values: {capture.values.shape}")     # [L, H_kv, T, D]
print(f"Patterns: {capture.patterns.shape}") # [L, H, T, T]
```

**Output:**
```
Queries: torch.Size([32, 32, 10, 128])
Keys: torch.Size([32, 8, 10, 128])
Values: torch.Size([32, 8, 10, 128])
Patterns: torch.Size([32, 32, 10, 10])
```

### 3. Analyze Attention

```python
from oculi.analysis import EntropyAnalysis, CircuitDetection

# Compute entropy (how diffuse is attention?)
entropy = EntropyAnalysis.token_entropy(capture)
print(f"Entropy shape: {entropy.shape}")  # [L, H, T]

# Detect induction heads
induction_heads = CircuitDetection.detect_induction_heads(capture)
print(f"Induction heads detected: {induction_heads.sum()}")
```

### 4. Full Capture (Everything)

```python
# Capture attention + residual + MLP + logits in one pass
full = adapter.capture_full(input_ids)

print(f"Attention: {full.attention is not None}")  # True
print(f"Residual: {full.residual is not None}")    # True
print(f"MLP: {full.mlp is not None}")              # True
print(f"Logits: {full.logits is not None}")        # True
```

## Phase 2 Features (NEW)

### Attribution Analysis

```python
from oculi.analysis import AttributionMethods

# Which layers contribute most to the output?
target_token_id = tokenizer.encode("dog")[0]
unembed = model.lm_head.weight

attribution = AttributionMethods.direct_logit_attribution(
    full.residual, unembed, target_token_id
)
print(f"Most important layer: {attribution.values.argmax().item()}")
```

### Composition Analysis

```python
from oculi.analysis import CompositionAnalysis

# How do heads compose?
qk_comp = CompositionAnalysis.qk_composition(
    full.attention,
    source=(10, 5),  # Layer 10, Head 5
    target=(20, 3)   # Layer 20, Head 3
)
print(f"Composition score: {qk_comp.values.mean():.4f}")

# Detect induction circuits
circuits = CompositionAnalysis.detect_induction_circuit(full.attention)
print(f"Found {len(circuits.metadata['circuits'])} induction circuits")
```

## Common Patterns

### Selective Capture (Save Memory)

```python
from oculi import CaptureConfig

# Only capture specific layers and components
config = CaptureConfig(
    layers=[20, 21, 22],       # Last 3 layers only
    capture_queries=True,
    capture_keys=False,        # Don't need keys
    capture_values=False,      # Don't need values
    capture_patterns=True
)

capture = adapter.capture(input_ids, config=config)
```

### Intervention Example

```python
from oculi.intervention import SpectraScaler, InterventionContext

# Sharpen attention at layer 23, head 5
scaler = SpectraScaler(layer=23, head=5, alpha=1.5)

with InterventionContext(adapter, [scaler]):
    output = adapter.generate(
        "The capital of France is",
        max_new_tokens=10
    )
print(output)
```

## Testing Without GPU

Use mock models for quick testing:

```python
from tests.mocks import MockLlamaAdapter

# Tiny model for CPU testing
adapter = MockLlamaAdapter()
input_ids = adapter.tokenize("Test input")

# All features work
capture = adapter.capture(input_ids)
full = adapter.capture_full(input_ids)
```

## Next Steps

<div class="grid cards" markdown>

-   __Learn Core Concepts__

    Understand Oculi's design philosophy

    [:octicons-arrow-right-24: Core Concepts](core-concepts.md)

-   __Explore Guides__

    In-depth documentation for each feature

    [:octicons-arrow-right-24: User Guides](../guides/attention-capture.md)

-   __Try Tutorials__

    Step-by-step examples with explanations

    [:octicons-arrow-right-24: Tutorials](../tutorials/01-basic-attention.md)

-   __API Reference__

    Complete API documentation

    [:octicons-arrow-right-24: API Docs](../api-reference/capture.md)

</div>

## Help & Support

- **Issues:** [GitHub Issues](https://github.com/ajayspatil7/oculi/issues)
- **Discussions:** [GitHub Discussions](https://github.com/ajayspatil7/oculi/discussions)
- **Contributing:** [Contributing Guide](../CONTRIBUTING.md)
