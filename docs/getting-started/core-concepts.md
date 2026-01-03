# Core Concepts

Understanding Oculi's design philosophy and key abstractions.

## Design Philosophy

Oculi is built on four core principles:

### 1. Learning-First Design

**Adapters are executable documentation.**

Instead of hiding model internals behind abstractions, Oculi makes them explicit. Each adapter is a learning resource that shows you exactly where every component lives in the model.

```python
# Traditional approach (hidden)
capture = magic_capture(model, input)  # How does this work? 🤷

# Oculi approach (explicit)
from oculi.models.llama import LlamaAttentionAdapter
adapter = LlamaAttentionAdapter(model, tokenizer)
# See exactly what's happening in adapter.py
```

See: [`oculi/models/llama/anatomy.py`](https://github.com/ajayspatil7/oculi/blob/main/oculi/models/llama/anatomy.py) - Full documentation of model structure

### 2. Explicit Control

**No magic. You choose what to capture.**

Oculi never auto-detects or makes assumptions. You explicitly specify what you want:

```python
from oculi import CaptureConfig

# Explicit configuration
config = CaptureConfig(
    layers=[20, 21, 22],       # Exactly these layers
    capture_queries=True,       # Yes to queries
    capture_values=False,       # No to values
    qk_stage='post_rope'        # After RoPE
)

capture = adapter.capture(input_ids, config=config)
```

### 3. Pure Functional Analysis

**Stateless, deterministic, testable.**

All analysis functions are pure:

```python
def analysis_function(capture: AttentionCapture, **params) -> torch.Tensor:
    """
    Pure function: AttentionCapture → Tensor

    - No side effects
    - No model access
    - No plotting
    - Deterministic output
    """
```

Benefits:
- Same inputs → Same outputs
- Easy to test
- Easy to understand
- Easy to compose

### 4. Memory-Conscious

**Selective capture, efficient storage.**

Transformers are memory-intensive. Oculi gives you control:

```python
# Memory strategies
config1 = CaptureConfig(layers=[20, 21, 22])  # Only specific layers
config2 = CaptureConfig(capture_values=False)  # Skip values
config3 = LogitConfig(top_k=10)  # Top-k only for logits

# Choose what you need
capture = adapter.capture(input_ids, config1)
```

---

## Key Abstractions

### AttentionCapture

The core data structure for captured attention:

```python
@dataclass(frozen=True)
class AttentionCapture:
    queries: torch.Tensor   # [L, H, T, D]
    keys: torch.Tensor      # [L, H_kv, T, D]
    values: torch.Tensor    # [L, H_kv, T, D]
    patterns: torch.Tensor  # [L, H, T, T]

    n_layers: int
    n_heads: int
    n_kv_heads: int
    n_tokens: int
    head_dim: int
```

**Invariants:**
- All tensors on CPU, detached
- `patterns[l, h, i, j]` = attention from token i to token j
- Patterns sum to 1.0 (softmax)
- Causal masking enforced

### ResidualCapture

Residual stream at four intervention points:

```python
@dataclass(frozen=True)
class ResidualCapture:
    pre_attn: Optional[torch.Tensor]   # [L, T, H] - Before attention
    post_attn: Optional[torch.Tensor]  # [L, T, H] - After attention
    pre_mlp: Optional[torch.Tensor]    # [L, T, H] - Before MLP
    post_mlp: Optional[torch.Tensor]   # [L, T, H] - After MLP
```

**Usage:** Understanding information flow through residual stream

### Model Adapters

Bridge between Oculi's API and specific model architectures:

```python
class ModelAdapter(ABC):
    @abstractmethod
    def capture(self, input_ids, config) -> AttentionCapture:
        """Run forward pass and capture attention."""

    @abstractmethod
    def attention_structure(self, layer) -> AttentionStructure:
        """Describe attention architecture."""
```

**Current Adapters:**
- `LlamaAttentionAdapter` - LLaMA 2/3 (GQA support)
- More coming soon (Mistral, Qwen)

---

## Analysis Workflow

### 1. Capture

```python
# Basic capture
capture = adapter.capture(input_ids)

# Full capture (everything)
full = adapter.capture_full(input_ids)
```

### 2. Analyze

```python
from oculi.analysis import EntropyAnalysis, AttributionMethods

# Compute metrics
entropy = EntropyAnalysis.token_entropy(capture)

# Attribution
attribution = AttributionMethods.direct_logit_attribution(
    full.residual, unembed, target_token
)
```

### 3. Interpret

```python
# Find important layers
important_layers = attribution.values.abs().topk(5)

# Examine circuits
circuits = CompositionAnalysis.detect_induction_circuit(capture)
```

### 4. Intervene (Optional)

```python
from oculi.intervention import SpectraScaler, InterventionContext

scaler = SpectraScaler(layer=23, head=5, alpha=1.5)

with InterventionContext(adapter, [scaler]):
    output = adapter.generate(prompt, max_new_tokens=10)
```

---

## Understanding Shapes

### Attention Shapes

```python
L  = n_layers       # Number of transformer layers
H  = n_heads        # Number of query heads
H_kv = n_kv_heads   # Number of key/value heads (GQA)
T  = n_tokens       # Sequence length
D  = head_dim       # Dimension per head
```

**Common Shapes:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `queries` | `[L, H, T, D]` | Query vectors per layer/head/position |
| `keys` | `[L, H_kv, T, D]` | Key vectors (note H_kv for GQA) |
| `values` | `[L, H_kv, T, D]` | Value vectors |
| `patterns` | `[L, H, T, T]` | Attention weights (i→j) |

### Analysis Shapes

| Method | Output Shape | Description |
|--------|--------------|-------------|
| `token_entropy()` | `[L, H, T]` | Entropy per position |
| `q_norms()` | `[L, H, T]` | Query norms |
| `direct_logit_attribution()` | `[L]` | Layer contributions |
| `component_attribution()` | `[L, 2]` | Attention vs MLP |

---

## Grouped Query Attention (GQA)

Modern LLMs use GQA where multiple query heads share key/value heads:

```python
# LLaMA-3-8B example
n_heads = 32      # Query heads
n_kv_heads = 8    # KV heads
gqa_ratio = 4     # 4 query heads per KV head

# Shapes reflect this
queries.shape  # [32, 32, T, 128]
keys.shape     # [32, 8, T, 128]  # Note: 8 not 32!
values.shape   # [32, 8, T, 128]
```

**Detection:**

```python
if capture.is_gqa:
    ratio = capture.gqa_ratio
    print(f"GQA model: {ratio}:1 ratio")
```

---

## Intervention Points

Oculi captures at strategic intervention points:

```
Input
  ↓
[Layer 0]
  pre_attn  ←─ Capture point 1
  ↓
  Attention
  ↓
  post_attn ←─ Capture point 2
  ↓
  pre_mlp   ←─ Capture point 3
  ↓
  MLP
  ↓
  post_mlp  ←─ Capture point 4
  ↓
[Layer 1]
  ...
```

Each point reveals different aspects of computation.

---

## Configuration System

### CaptureConfig

```python
config = CaptureConfig(
    layers=[20, 21, 22],        # Which layers
    capture_queries=True,        # What components
    capture_keys=True,
    capture_values=True,
    capture_patterns=True,
    qk_stage='pre_rope'         # When (pre/post RoPE)
)
```

### ResidualConfig

```python
config = ResidualConfig(
    layers=[20, 21, 22],
    capture_pre_attn=True,
    capture_post_attn=True,
    capture_pre_mlp=True,
    capture_post_mlp=True
)
```

### LogitConfig

```python
config = LogitConfig(
    layers=None,  # All layers
    top_k=10      # Memory-efficient: top-10 only
)
```

---

## Testing Strategy

### Three Testing Tiers

**Contract Tests** - Shape and semantic contracts:
```python
def test_entropy_shape():
    entropy = EntropyAnalysis.token_entropy(capture)
    assert entropy.shape == (L, H, T)
```

**Integration Tests** - End-to-end with mock models:
```python
def test_full_capture():
    adapter = MockLlamaAdapter()
    full = adapter.capture_full(input_ids)
    assert full.attention is not None
```

**Performance Tests** - Real models (GPU):
```python
def test_memory_efficiency():
    # Only run on GPU machines
    ...
```

### Mock Models for CPU

```python
from tests.mocks import MockLlamaAdapter

# Tiny model for testing
adapter = MockLlamaAdapter()
input_ids = adapter.tokenize("Test")
capture = adapter.capture(input_ids)
```

---

## Best Practices

### 1. Always Read Files First

```python
# Before using an adapter
from oculi.models.llama import LlamaAttentionAdapter

# Check out the source code at:
# oculi/models/llama/adapter.py
# oculi/models/llama/anatomy.py
```

### 2. Use Type Hints

```python
from oculi.capture.structures import AttentionCapture

def my_analysis(capture: AttentionCapture) -> torch.Tensor:
    ...
```

### 3. Validate Configs

```python
config = CaptureConfig(layers=[100])  # Invalid!
config.validate(adapter)  # Raises ValueError
```

### 4. Handle GQA Properly

```python
# Account for GQA in analysis
if capture.is_gqa:
    # Keys/values have H_kv heads, not H heads
    n_kv_heads = capture.n_kv_heads
```

### 5. Memory Management

```python
# For large models
config = CaptureConfig(
    layers=[20, 21, 22],     # Subset only
    capture_values=False     # Skip if not needed
)
```

---

## Next Steps

<div class="grid cards" markdown>

-   __Quick Start__

    Get up and running

    [:octicons-arrow-right-24: Quick Start](quick-start.md)

-   __User Guides__

    In-depth feature docs

    [:octicons-arrow-right-24: Guides](../guides/attention-capture.md)

-   __API Reference__

    Complete API docs

    [:octicons-arrow-right-24: API](../api-reference/capture.md)

-   __Examples__

    Working code samples

    [:octicons-arrow-right-24: Examples](../../examples/)

</div>
