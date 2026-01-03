# Oculi

> **A comprehensive mechanistic interpretability toolkit for transformer LLMs**

[![Version](https://img.shields.io/badge/version-0.3.0--dev-blue)]()
[![Python](https://img.shields.io/badge/python-3.10+-green)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## What is Oculi?

Oculi is a **research-first** mechanistic interpretability toolkit for transformer language models. It provides surgical instrumentation for understanding how transformers work internally.

### Core Capabilities

**Comprehensive Capture System:**
- ✅ **Attention Internals** — Q/K/V vectors, attention patterns with pre/post-RoPE options
- ✅ **Residual Stream** — Activations at all intervention points (pre/post attention, pre/post MLP)
- ✅ **MLP Internals** — Gate, up projections, activations, and outputs
- ✅ **Layer-wise Logits** — Logit lens analysis with memory-efficient top-k

**Analysis Tools:**
- 🔍 **Circuit Detection** — Automatic detection of induction heads, previous token heads, positional patterns
- 📊 **Logit Lens** — Track prediction formation across layers
- 📈 **Entropy & Norms** — Attention focus metrics, vector magnitudes
- 🔗 **Correlation Analysis** — Statistical relationships with p-values

**Surgical Interventions:**
- 🎯 **Q/K Scaling** — The Spectra method for attention sharpening/flattening
- ❌ **Head Ablation** — Zero out specific attention heads
- 🔄 **Activation Patching** — (Coming in v0.6.0)

**Design Philosophy:**
- **Learning-First** — Adapters are _executable documentation_ of model internals
- **Pure Functional** — Stateless, deterministic analysis functions
- **Explicit Control** — No magic, you choose what to capture
- **Memory-Conscious** — Selective capture, top-k optimization

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

**Requirements:**
- Python 3.10+
- PyTorch 2.0.0+
- Transformers 4.30.0+

---

## Quick Start

### Basic Attention Capture

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from oculi.models.llama import LlamaAttentionAdapter

# Load model explicitly (no magic auto-detection)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Create adapter
adapter = LlamaAttentionAdapter(model, tokenizer)

# Capture attention data
input_ids = tokenizer.encode("The quick brown fox jumps over the lazy dog", return_tensors="pt")
capture = adapter.capture(input_ids)

print(f"Queries: {capture.queries.shape}")   # [L, H, T, D] - [32, 32, 10, 128]
print(f"Keys: {capture.keys.shape}")         # [L, H_kv, T, D] - [32, 8, 10, 128] (GQA)
print(f"Patterns: {capture.patterns.shape}") # [L, H, T, T] - [32, 32, 10, 10]
```

---

## 🆕 Phase 1 Features (v0.3.0-dev)

### Residual Stream Capture

Capture activations at all key intervention points in the transformer:

```python
from oculi import ResidualConfig

# Configure what to capture
config = ResidualConfig(
    layers=[20, 21, 22],  # Specific layers
    capture_pre_attn=True,   # Before attention
    capture_post_attn=True,  # After attention, before MLP
    capture_pre_mlp=True,    # Before MLP
    capture_post_mlp=True    # After MLP (residual stream output)
)

# Capture residual stream
residual = adapter.capture_residual(input_ids, config=config)

print(f"Pre-attention: {residual.pre_attn.shape}")   # [L, T, H] - [3, 10, 4096]
print(f"Post-attention: {residual.post_attn.shape}") # [L, T, H]
print(f"Pre-MLP: {residual.pre_mlp.shape}")          # [L, T, H]
print(f"Post-MLP: {residual.post_mlp.shape}")        # [L, T, H]
```

### MLP Internals Capture

Examine MLP activations and neuron-level behavior:

```python
from oculi import MLPConfig

# Capture MLP activations
config = MLPConfig(
    layers=[20, 21, 22],
    capture_gate=True,           # Gate projection
    capture_up=True,             # Up projection
    capture_post_activation=True, # After SiLU activation
    capture_output=True          # MLP output
)

mlp = adapter.capture_mlp(input_ids, config=config)

print(f"Gate projection: {mlp.gate.shape}")           # [L, T, intermediate_dim]
print(f"Up projection: {mlp.up.shape}")               # [L, T, intermediate_dim]
print(f"Post-activation: {mlp.post_activation.shape}") # [L, T, intermediate_dim]
print(f"Output: {mlp.output.shape}")                  # [L, T, hidden_dim]
```

### Logit Lens Analysis

Track how predictions evolve across layers:

```python
from oculi import LogitConfig
from oculi.analysis import LogitLensAnalysis

# Capture layer-wise logits
config = LogitConfig(
    layers=None,  # All layers
    top_k=10      # Memory-efficient: only store top-10 per position
)

logits = adapter.capture_logits(input_ids, config=config)

# Analyze predictions
lens = LogitLensAnalysis(tokenizer)

# Get top predictions at each layer
predictions = lens.layer_predictions(logits, token_position=-1, top_k=5)
for pred in predictions[:5]:  # First 5 layers
    print(f"Layer {pred['layer']}: {pred['predictions'][:3]}")

# Measure prediction convergence
convergence = lens.prediction_convergence(logits)
print(f"Convergence (KL divergence): {convergence.shape}")  # [L]

# Track specific token probability across layers
token_id = tokenizer.encode("dog")[0]
trajectory = lens.token_probability_trajectory(logits, token_id)
print(f"Token trajectory: {trajectory.shape}")  # [L, T]
```

### Circuit Detection

Automatically detect canonical transformer circuits:

```python
from oculi.analysis import CircuitDetection

# Detect induction heads (A B ... A -> B pattern)
induction_scores = CircuitDetection.detect_induction_heads(capture, threshold=0.5)
print(f"Induction heads: {induction_scores.shape}")  # [L, H] - scores per head
print(f"Found {(induction_scores > 0.5).sum()} induction heads")

# Detect previous token heads (attend to t-1)
prev_token_scores = CircuitDetection.detect_previous_token_heads(capture, threshold=0.8)
print(f"Previous token heads: {(prev_token_scores > 0.8).sum()}")

# Detect positional heads (BOS, recent tokens, etc.)
positional = CircuitDetection.detect_positional_heads(capture)
print(f"BOS-attending heads: {positional['bos'].sum()}")
print(f"Recent-attending heads: {positional['recent'].sum()}")

# Classify all heads
for layer in range(capture.n_layers):
    for head in range(capture.n_heads):
        classification = CircuitDetection.classify_attention_head(
            capture, layer, head
        )
        if classification['pattern'] != 'unknown':
            print(f"L{layer}H{head}: {classification['pattern']} "
                  f"(score: {classification['score']:.2f})")
```

### Unified Full Capture

Capture everything in a single forward pass:

```python
from oculi import FullCapture

# Single forward pass captures:
# - Attention (Q/K/V/patterns)
# - Residual stream (all 4 points)
# - MLP internals
# - Logits (with top-k)
full = adapter.capture_full(input_ids)

# Access components
print(f"Attention: {full.attention is not None}")   # True
print(f"Residual: {full.residual is not None}")     # True
print(f"MLP: {full.mlp is not None}")               # True
print(f"Logits: {full.logits is not None}")         # True

# Use individual captures
entropy = EntropyAnalysis.token_entropy(full.attention)
lens_analysis = LogitLensAnalysis(tokenizer).layer_predictions(full.logits, -1)
```

---

## Analysis Examples

### Entropy Analysis

```python
from oculi.analysis import EntropyAnalysis

# Token-level entropy (how diffuse is attention?)
entropy = EntropyAnalysis.token_entropy(capture)
print(f"Entropy shape: {entropy.shape}")  # [L, H, T]

# Effective rank (how many tokens effectively attended to?)
eff_rank = EntropyAnalysis.effective_rank(capture)
print(f"Effective rank: {eff_rank.shape}")  # [L, H, T]

# Compare two conditions
entropy_baseline = EntropyAnalysis.token_entropy(capture_baseline)
entropy_intervention = EntropyAnalysis.token_entropy(capture_intervention)
delta = EntropyAnalysis.delta_entropy(capture_intervention, capture_baseline)
print(f"Entropy change: {delta.shape}")  # [L, H]
```

### Norm Analysis

```python
from oculi.analysis import NormAnalysis

# Query/key/value vector norms
q_norms = NormAnalysis.q_norms(capture)
k_norms = NormAnalysis.k_norms(capture)
v_norms = NormAnalysis.v_norms(capture)

print(f"Q norms: {q_norms.shape}")  # [L, H, T]
print(f"K norms: {k_norms.shape}")  # [L, H_kv, T]
print(f"V norms: {v_norms.shape}")  # [L, H_kv, T]
```

### Correlation Analysis

```python
from oculi.analysis import CorrelationAnalysis

# Correlate entropy with Q norms
correlation = CorrelationAnalysis.norm_entropy_correlation(
    capture, ignore_first=2
)
print(f"Norm-entropy correlation: {correlation.shape}")  # [L, H]

# Custom correlation with p-values
x = q_norms.flatten()
y = entropy.flatten()
corr, pval = CorrelationAnalysis.pearson_with_pvalue(x, y)
print(f"Correlation: {corr:.3f}, p-value: {pval:.3e}")
```

---

## Intervention Examples

### Q/K Scaling (Spectra Method)

```python
from oculi.intervention import SpectraScaler, InterventionContext

# Sharpen attention at layer 23, head 5
scaler = SpectraScaler(layer=23, head=5, alpha=1.5)  # α > 1 sharpens

# Apply during generation
with InterventionContext(adapter, [scaler]):
    output = adapter.generate(
        "The capital of France is",
        max_new_tokens=10
    )
print(output)

# Flatten attention (α < 1)
flattener = SpectraScaler(layer=23, head=5, alpha=0.5)
with InterventionContext(adapter, [flattener]):
    output = adapter.generate("The capital of France is", max_new_tokens=10)
print(output)
```

### Head Ablation

```python
from oculi.intervention import HeadAblation

# Zero out specific head
ablation = HeadAblation(layer=20, head=3)

with InterventionContext(adapter, [ablation]):
    output = adapter.generate("Test prompt", max_new_tokens=10)
print(output)

# Ablate multiple heads
ablations = [
    HeadAblation(layer=20, head=3),
    HeadAblation(layer=21, head=5),
    HeadAblation(layer=22, head=7),
]

with InterventionContext(adapter, ablations):
    output = adapter.generate("Test prompt", max_new_tokens=10)
```

---

## Advanced Usage

### Selective Capture (Memory Optimization)

```python
from oculi import CaptureConfig

# Only capture what you need
config = CaptureConfig(
    layers=[20, 21, 22],       # Only last few layers
    capture_queries=True,       # Need queries
    capture_keys=False,         # Don't need keys
    capture_values=False,       # Don't need values
    capture_patterns=True,      # Need patterns
    qk_stage='post_rope'        # After position encoding
)

capture = adapter.capture(input_ids, config=config)
# Memory usage: ~40% of full capture
```

### Stratified Analysis

```python
from oculi.analysis import StratifiedView, find_extreme_heads

# Find heads with highest entropy
high_entropy_heads = find_extreme_heads(
    entropy,
    k=10,
    mode='max',
    layer_range=(20, 32)  # Only later layers
)

print("Top 10 highest entropy heads:")
for layer, head, score in high_entropy_heads:
    print(f"  Layer {layer}, Head {head}: {score:.3f}")

# Slice by specific dimensions
view = StratifiedView.by_layer(entropy, layer=25)
print(f"Layer 25 entropy: {view.shape}")  # [H, T]

view = StratifiedView.by_head(entropy, layer=25, head=10)
print(f"Layer 25, Head 10 entropy: {view.shape}")  # [T]
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
│       ├── anatomy.py   # Full model anatomy documentation
│       └── notes.md     # Architecture details
│
├── capture/         # Capture utilities & data structures
│   ├── structures.py    # AttentionCapture, ResidualCapture, MLPCapture, etc.
│   └── hooks.py         # Hook management
│
├── analysis/        # Pure analysis functions
│   ├── entropy.py       # Entropy metrics
│   ├── norms.py         # Vector norms
│   ├── circuits.py      # Circuit detection ✨ NEW
│   ├── logit_lens.py    # Logit lens analysis ✨ NEW
│   ├── correlation.py   # Statistical analysis
│   └── stratified.py    # Slicing helpers
│
├── intervention/    # Intervention definitions
│   ├── scalers.py       # Q/K/Spectra scaling
│   ├── ablation.py      # Head ablation
│   └── context.py       # Context manager
│
├── visualize/       # Research-quality plots
│   ├── entropy.py
│   ├── correlation.py
│   └── intervention.py
│
└── _private/        # Private implementation
    └── hooks/           # Hook implementations
```

**Design Principles:**

1. **Learning-First** — Adapters are _executable documentation_, not hidden glue
2. **Explicit Imports** — No magic auto-detection, you choose the model
3. **Public Model Anatomy** — See exactly where every component lives in `anatomy.py`
4. **Pure Functional Analysis** — Stateless, deterministic, testable
5. **Parallel Captures** — Independent structures (Attention, Residual, MLP, Logit) for memory control

---

## Testing Without GPU

```python
# Use mock model for CPU testing
from tests.mocks import MockLlamaAdapter

adapter = MockLlamaAdapter()  # Tiny LLaMA-like model
input_ids = adapter.tokenize("Test input")

# All features work with mock
capture = adapter.capture(input_ids)
residual = adapter.capture_residual(input_ids)
mlp = adapter.capture_mlp(input_ids)
logits = adapter.capture_logits(input_ids)
full = adapter.capture_full(input_ids)

# Circuit detection on mock
circuits = CircuitDetection.detect_induction_heads(capture)
```

---

## Roadmap

### ✅ Phase 1 (v0.3.0 - v0.4.0) - Complete
- ✅ Residual stream capture
- ✅ MLP internals capture
- ✅ Logit lens analysis
- ✅ Circuit detection primitives
- ✅ Unified full capture

### 🔄 Phase 2 (v0.5.0 - v0.6.0) - In Progress
- [ ] Attribution methods (attention flow, direct logit attribution)
- [ ] Head composition analysis (OV/QK circuits)
- [ ] Activation patching (causal interventions)
- [ ] Steering vectors

### ⏳ Phase 3 (v0.7.0 - v0.8.0) - Planned
- [ ] Caching system
- [ ] Memory optimization (FP16, lazy materialization)
- [ ] Export formats (HDF5, JSON, NumPy)
- [ ] TransformerLens compatibility

### 🎯 Phase 4 (v1.0.0) - Future
- [ ] API freeze
- [ ] Complete documentation
- [ ] Benchmark suite
- [ ] Production-ready release

---

## Documentation

- [API Contract](docs/API_CONTRACT.md) — Tensor shapes, math definitions, guarantees
- [LLaMA Anatomy](oculi/models/llama/anatomy.py) — Hook points, module paths, tensor shapes
- [LLaMA Notes](oculi/models/llama/notes.md) — GQA, RoPE, architecture details

---

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

---

## Contributing

Contributions welcome! Please see the [implementation plan](.claude/plans/inherited-fluttering-owl.md) for current priorities.

---

## License

MIT License - see [LICENSE](LICENSE) for details

## Author

**Ajay S Patil**
- GitHub: [@ajayspatil7](https://github.com/ajayspatil7)
- Email: ajayspatil7@gmail.com
