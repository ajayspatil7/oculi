# Oculi

> **A low-level, surgical instrumentation layer for LLMs**

[![Version](https://img.shields.io/badge/version-0.1.0-blue)]()
[![Python](https://img.shields.io/badge/python-3.10+-green)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## What is Oculi?

Oculi is a **research-first** mechanistic interpretability toolkit for transformer language models. It provides:

- **Token-level QKV capture** with pre/post-RoPE options
- **Attention entropy computation** with causal masking
- **Surgical interventions** via Q/K scaling (the Spectra method)
- **Model-agnostic design** (LLaMA, Mistral, Qwen, Falcon)

---

## Installation

```bash
# From source
git clone https://github.com/ajayspatil7/oculi.git
cd oculi
pip install -e .

# With visualization support
pip install -e ".[viz]"

# With dev tools
pip install -e ".[all]"
```

---

## Quick Start

```python
import oculi

# Load model (auto-detects architecture)
model = oculi.load("meta-llama/Meta-Llama-3-8B")

# Capture attention data
capture = model.capture(input_ids)

# Compute metrics
entropy = oculi.analysis.EntropyAnalysis.token_entropy(capture)
q_norms = oculi.analysis.NormAnalysis.q_norms(capture)

print(f"Entropy shape: {entropy.shape}")  # [L, H, T]
print(f"Q norms shape: {q_norms.shape}")  # [L, H, T]
```

### Intervention Example

```python
from oculi.intervention import SpectraScaler, InterventionContext

# Define intervention: sharpen attention at layer 23, head 5
scaler = SpectraScaler(layer=23, head=5, alpha=1.5)

# Apply and generate
with InterventionContext(model, [scaler]):
    output = model.generate("The answer is")
```

---

## Core Concepts

### Capture

```python
from oculi import CaptureConfig

# Capture specific layers and components
config = CaptureConfig(
    layers=[20, 21, 22],       # Only capture these layers
    capture_patterns=True,      # Attention patterns
    capture_queries=True,       # Q vectors
    qk_stage='pre_rope'         # Before positional encoding
)

capture = model.capture(input_ids, config=config)
```

### Analysis

All analysis functions are **pure**: `AttentionCapture → Tensor`

```python
from oculi.analysis import (
    NormAnalysis,      # q_norms, k_norms, v_norms
    EntropyAnalysis,   # token_entropy, delta_entropy
    AttentionAnalysis, # max_weight, effective_span
    CorrelationAnalysis # pearson, norm_entropy_correlation
)
```

### Intervention

```python
from oculi.intervention import (
    QScaler,        # Scale Q by α
    KScaler,        # Scale K by α
    SpectraScaler,  # Scale both Q,K by √α (net effect: α on logits)
    HeadAblation,   # Zero out head
)
```

---

## Supported Models

| Model      | Adapter          | Attention | Status |
| ---------- | ---------------- | --------- | ------ |
| LLaMA 2/3  | `LlamaAdapter`   | GQA       | ✅     |
| Mistral    | `MistralAdapter` | GQA       | 🔄     |
| Qwen 2/2.5 | `QwenAdapter`    | GQA       | 🔄     |
| Falcon     | `FalconAdapter`  | MQA       | 🔄     |

---

## Documentation

- [API Contract](docs/API_CONTRACT.md) — Tensor shapes, math definitions, guarantees

---

## Architecture

```
oculi/
├── capture/        # Core data structures & model interface
├── analysis/       # Pure analysis functions
├── intervention/   # Intervention definitions
├── visualize/      # Research-quality plots
├── _private/       # Implementation details (adapters, hooks)
└── __init__.py     # Public API exports
```

**Design Principles:**

1. **Public/Private Separation** — Public API is versioned, private can change
2. **No PyTorch in Public** — Researchers cite semantics, not hooks
3. **Pure Functional Analysis** — Stateless, deterministic, testable

---

## License

MIT

## Author

**Ajay S Patil**
