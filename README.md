# Oculi

> **A low-level, surgical instrumentation layer for LLMs**

[![Version](https://img.shields.io/badge/version-0.2.0-blue)]()
[![Python](https://img.shields.io/badge/python-3.10+-green)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## What is Oculi?

Oculi is a **research-first** mechanistic interpretability toolkit for transformer language models. It provides:

- **Token-level QKV capture** with pre/post-RoPE options
- **Attention entropy computation** with causal masking
- **Surgical interventions** via Q/K scaling (the Spectra method)
- **Learning-first design** — adapters are _executable documentation_ of model internals

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
from transformers import AutoModelForCausalLM, AutoTokenizer
from oculi.models.llama import LlamaAttentionAdapter

# Load model explicitly (no magic)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Create adapter
adapter = LlamaAttentionAdapter(model, tokenizer)

# Capture attention data
input_ids = tokenizer.encode("Hello world", return_tensors="pt")
capture = adapter.capture(input_ids)

print(f"Queries: {capture.queries.shape}")   # [L, H, T, D]
print(f"Patterns: {capture.patterns.shape}") # [L, H, T, T]
```

### Analysis Example

```python
from oculi.analysis import EntropyAnalysis, NormAnalysis

# Compute metrics
entropy_analyzer = EntropyAnalysis(capture)
entropy = entropy_analyzer.compute()
print(f"Entropy: {entropy.results.shape}")  # [L, H, T]

# Query norms
norm_analyzer = NormAnalysis(capture)
q_norms = norm_analyzer.compute_query_norms()
print(f"Q norms: {q_norms.shape}")  # [L, H, T]
```

### Testing Without GPU

```python
# Use mock model for CPU testing
from tests.mocks import MockLlamaAdapter

adapter = MockLlamaAdapter()  # Tiny LLaMA-like model
capture = adapter.capture(adapter.tokenize("Test input"))
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

capture = adapter.capture(input_ids, config=config)
```

### Analysis

All analysis functions are **pure**: `AttentionCapture → Tensor`

```python
from oculi.analysis import (
    NormAnalysis,       # q_norms, k_norms, v_norms
    EntropyAnalysis,    # token_entropy, delta_entropy
    CorrelationAnalysis # pearson, spearman with p-values
)
```

### Intervention

```python
from oculi.intervention import (
    QScaler,        # Scale Q by α
    KScaler,        # Scale K by α
    HeadAblation,   # Zero out head
)
```

---

## Supported Models

| Model      | Adapter                 | Attention | Status |
| ---------- | ----------------------- | --------- | ------ |
| LLaMA 2/3  | `LlamaAttentionAdapter` | GQA       | ✅     |
| Mistral    | Coming soon             | GQA       | 🔄     |
| Qwen 2/2.5 | Coming soon             | GQA       | 🔄     |

---

## Architecture

```
oculi/
├── models/          # 🔥 PUBLIC model adapters
│   ├── base.py      # AttentionAdapter contract
│   └── llama/       # LLaMA family
│       ├── adapter.py   # LlamaAttentionAdapter
│       ├── attention.py # Q/K/V extraction, GQA, RoPE
│       └── notes.md     # Architecture documentation
│
├── capture/         # Capture utilities & data structures
├── analysis/        # Pure analysis functions
├── intervention/    # Intervention definitions
└── visualize/       # Research-quality plots
```

**Design Principles:**

1. **Learning-First** — Adapters are _executable documentation_, not hidden glue
2. **Explicit Imports** — No magic auto-detection, you choose the model
3. **Public Model Anatomy** — See exactly where Q/K/V live in `attention.py`
4. **Pure Functional Analysis** — Stateless, deterministic, testable

---

## Documentation

- [API Contract](docs/API_CONTRACT.md) — Tensor shapes, math definitions, guarantees
- [LLaMA Notes](oculi/models/llama/notes.md) — GQA, RoPE, architecture details

---

## License

MIT

## Author

**Ajay S Patil**
