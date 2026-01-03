# Oculi Documentation

<div align="center">
  <h3>Comprehensive Mechanistic Interpretability Toolkit for Transformer LLMs</h3>

  [![Version](https://img.shields.io/badge/version-0.3.0--dev-blue)]()
  [![Python](https://img.shields.io/badge/python-3.10+-green)]()
  [![License](https://img.shields.io/badge/license-MIT-lightgrey)]()
  [![Tests](https://img.shields.io/badge/tests-85%20passing-brightgreen)]()
</div>

---

## Welcome to Oculi

Oculi is a **research-first** mechanistic interpretability toolkit for transformer language models. It provides surgical instrumentation for understanding how transformers work internally.

!!! tip "What makes Oculi different?"
    - **Learning-First Design** — Adapters are executable documentation
    - **Explicit Control** — No magic, you choose what to capture
    - **Pure Functional** — Stateless, deterministic analysis
    - **Memory-Conscious** — Selective capture, efficient storage

## Features

### ✅ Comprehensive Capture System

- **Attention Internals** — Q/K/V vectors, attention patterns with pre/post-RoPE options
- **Residual Stream** — Activations at all intervention points (pre/post attention, pre/post MLP)
- **MLP Internals** — Gate, up projections, activations, and outputs
- **Layer-wise Logits** — Logit lens analysis with memory-efficient top-k

### 🔍 Analysis Tools

- **Circuit Detection** — Automatic detection of induction heads, previous token heads
- **Logit Lens** — Track prediction formation across layers
- **Attribution Methods** — Understand information flow and component contributions ✨ **NEW**
- **Composition Analysis** — Analyze how attention heads interact ✨ **NEW**
- **Entropy & Norms** — Attention focus metrics, vector magnitudes
- **Correlation Analysis** — Statistical relationships with p-values

### 🎯 Surgical Interventions

- **Q/K Scaling** — The Spectra method for attention sharpening/flattening
- **Head Ablation** — Zero out specific attention heads
- **Activation Patching** — (Coming in v0.6.0)

## Quick Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from oculi.models.llama import LlamaAttentionAdapter
from oculi.analysis import AttributionMethods, CompositionAnalysis

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Create adapter
adapter = LlamaAttentionAdapter(model, tokenizer)

# Capture everything
input_ids = tokenizer.encode("The cat sat on the mat", return_tensors="pt")
full = adapter.capture_full(input_ids)

# Analyze attribution
target_token = tokenizer.encode("mat")[0]
attribution = AttributionMethods.direct_logit_attribution(
    full.residual, model.lm_head.weight, target_token
)
print(f"Most important layer: {attribution.values.argmax()}")

# Detect induction circuits
circuits = CompositionAnalysis.detect_induction_circuit(full.attention)
print(f"Found {len(circuits.metadata['circuits'])} circuits")
```

## Getting Started

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Quick Start__

    ---

    Get up and running in 5 minutes

    [:octicons-arrow-right-24: Quick Start](getting-started/quick-start.md)

-   :material-book-open-variant:{ .lg .middle } __User Guide__

    ---

    In-depth guides for each feature

    [:octicons-arrow-right-24: Guides](guides/attention-capture.md)

-   :material-school:{ .lg .middle } __Tutorials__

    ---

    Step-by-step tutorials with examples

    [:octicons-arrow-right-24: Tutorials](tutorials/01-basic-attention.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Complete API documentation

    [:octicons-arrow-right-24: Reference](api-reference/capture.md)

</div>

## Latest Updates

### Phase 2 (v0.5.0-dev) - Current

??? success "✅ Attribution Methods"
    - Attention flow tracking across layers
    - Value-weighted attention patterns
    - Direct logit attribution
    - Component attribution (attention vs MLP)
    - Head-level attribution
    - Top-k attribution extraction

??? success "✅ Head Composition Analysis"
    - QK composition between head pairs
    - OV composition for value flow
    - Virtual attention through multi-head paths
    - Path patching importance scores
    - Full composition matrices
    - Automatic induction circuit detection

??? info "🔄 In Progress"
    - Activation patching for causal interventions
    - SAE integration
    - Probing & steering vectors

### Phase 1 (v0.3.0-v0.4.0) - Complete

??? check "Completed Features"
    - ✅ Residual stream capture at all intervention points
    - ✅ MLP internals capture
    - ✅ Logit lens analysis
    - ✅ Circuit detection primitives
    - ✅ Unified full capture

## Supported Models

| Model Family | Adapter                 | Attention Type | Status |
|--------------|-------------------------|----------------|--------|
| LLaMA 2/3    | `LlamaAttentionAdapter` | GQA            | ✅     |
| Mistral      | Coming soon             | GQA            | 🔄     |
| Qwen 2/2.5   | Coming soon             | GQA            | 🔄     |

## Installation

```bash
# Basic installation
pip install oculi

# From source (for development)
git clone https://github.com/ajayspatil7/oculi.git
cd oculi
pip install -e .

# With visualization support
pip install -e ".[viz]"

# With documentation tools
pip install -e ".[docs]"

# Everything
pip install -e ".[all]"
```

**Requirements:** Python 3.10+, PyTorch 2.0.0+, Transformers 4.30.0+

## Community & Support

- **GitHub Issues** — [Report bugs or request features](https://github.com/ajayspatil7/oculi/issues)
- **Discussions** — [Ask questions and share ideas](https://github.com/ajayspatil7/oculi/discussions)
- **Contributing** — [Read our contributing guide](CONTRIBUTING.md)

## Citation

If you use Oculi in your research, please cite:

```bibtex
@software{oculi2024,
  author = {Patil, Ajay S},
  title = {Oculi: Mechanistic Interpretability Toolkit for Transformers},
  year = {2024},
  url = {https://github.com/ajayspatil7/oculi}
}
```

## License

MIT License - see [LICENSE](https://github.com/ajayspatil7/oculi/blob/main/LICENSE) for details.

---

<div align="center">
  <sub>Built with ❤️ for the mechanistic interpretability community</sub>
</div>
