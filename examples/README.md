# Oculi Examples

This directory contains working examples demonstrating Oculi's features.

## Directory Structure

```
examples/
├── basic/          # Beginner-friendly examples
├── advanced/       # Advanced features (Phase 2+)
└── notebooks/      # Jupyter notebooks (coming soon)
```

## Running Examples

### Prerequisites

```bash
# Install Oculi
cd /path/to/oculi
pip install -e .

# For visualization examples
pip install -e ".[viz]"
```

### Basic Examples

Start here if you're new to Oculi:

```bash
# 1. Basic attention capture
python examples/basic/01_attention_capture.py

# 2. Residual stream capture
python examples/basic/02_residual_stream.py

# 3. MLP internals
python examples/basic/03_mlp_internals.py
```

### Advanced Examples

Explore Phase 2 features:

```bash
# 1. Circuit detection
python examples/advanced/01_circuit_detection.py

# 2. Logit lens analysis
python examples/advanced/02_logit_lens.py

# 3. Attribution methods ✨ NEW
python examples/advanced/03_attribution_methods.py

# 4. Composition analysis ✨ NEW
python examples/advanced/04_composition_analysis.py
```

## Using Mock Models (CPU)

All examples work with mock models for CPU testing:

```python
from tests.mocks import MockLlamaAdapter

# Tiny model, fast on CPU
adapter = MockLlamaAdapter()
```

## Using Real Models (GPU)

For real experiments, uncomment the model loading:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from oculi.models.llama import LlamaAttentionAdapter

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

adapter = LlamaAttentionAdapter(model, tokenizer)
```

## Example Index

### Basic Examples

| File | Description | Key Features |
|------|-------------|--------------|
| `01_attention_capture.py` | Basic attention capture | Q/K/V vectors, patterns, GQA |
| `02_residual_stream.py` | Residual stream analysis | 4 intervention points |
| `03_mlp_internals.py` | MLP activation capture | Gate, up, activations |

### Advanced Examples

| File | Description | Key Features |
|------|-------------|--------------|
| `01_circuit_detection.py` | Detect attention circuits | Induction heads, prev-token heads |
| `02_logit_lens.py` | Track predictions across layers | Layer-wise logits, convergence |
| `03_attribution_methods.py` | Attribution analysis ✨ | Flow, DLA, component attribution |
| `04_composition_analysis.py` | Head composition ✨ | QK/OV composition, circuits |

## Common Patterns

### Selective Capture (Memory Optimization)

```python
from oculi import CaptureConfig

config = CaptureConfig(
    layers=[20, 21, 22],     # Only last 3 layers
    capture_values=False     # Skip values
)

capture = adapter.capture(input_ids, config=config)
```

### Full Capture (Everything)

```python
# Capture attention + residual + MLP + logits
full = adapter.capture_full(input_ids)

# Access components
attention = full.attention
residual = full.residual
mlp = full.mlp
logits = full.logits
```

### Intervention Example

```python
from oculi.intervention import SpectraScaler, InterventionContext

scaler = SpectraScaler(layer=23, head=5, alpha=1.5)

with InterventionContext(adapter, [scaler]):
    output = adapter.generate(prompt, max_new_tokens=10)
```

## Notebooks (Coming Soon)

Interactive Jupyter notebooks will be added for:

- 📓 Attention pattern visualization
- 📓 Induction head analysis
- 📓 Attribution case studies
- 📓 Circuit discovery workflows

## Troubleshooting

### ImportError

```bash
pip install -e .  # Install in editable mode
```

### CUDA Out of Memory

Use selective capture or mock models:

```python
# Option 1: Selective capture
config = CaptureConfig(layers=[20, 21, 22])

# Option 2: Mock model
from tests.mocks import MockLlamaAdapter
adapter = MockLlamaAdapter()
```

### ModuleNotFoundError: No module named 'tests'

Run from the repository root:

```bash
cd /path/to/oculi
python examples/basic/01_attention_capture.py
```

## Contributing Examples

Want to add an example? Please:

1. Follow existing style and structure
2. Include clear docstrings
3. Work with both mock and real models
4. Add to this README's index
5. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

## Help & Support

- **Documentation:** [User Guides](../docs/guides/)
- **Issues:** [GitHub Issues](https://github.com/ajayspatil7/oculi/issues)
- **Discussions:** [GitHub Discussions](https://github.com/ajayspatil7/oculi/discussions)

---

Happy exploring! 🔬
