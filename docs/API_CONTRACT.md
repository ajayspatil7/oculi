# Oculi API Contract Specification

> **Version:** 0.1.0-draft  
> **Status:** Pre-release (API may change before 1.0)  
> **Last Updated:** 2026-01-01

---

## 1. Design Philosophy

### 1.1 Abstraction Firewall

Oculi maintains a strict separation between:

| Layer       | Purpose               | Location            | Stability            |
| ----------- | --------------------- | ------------------- | -------------------- |
| **Public**  | Research semantics    | `oculi/`          | Versioned, stable    |
| **Private** | Engineering mechanics | `oculi/_private/` | Internal, may change |

**Guarantees:**

- Public API changes trigger semantic versioning bumps
- Private implementation can be refactored without public API changes
- No PyTorch hooks, model-specific code, or implementation details in public layer

### 1.2 Guiding Principles

1. **Research Semantics First** — APIs express what researchers reason about
2. **Deterministic by Default** — Same inputs → same outputs (within floating-point limits)
3. **Inference Only** — No training, no gradient flows through captures
4. **Pure Functional Analysis** — Analysis functions are stateless transformations

---

## 2. Core Data Structures

### 2.1 AttentionCapture

The primary output of the capture system. Immutable container for captured attention data.

```python
@dataclass(frozen=True)
class AttentionCapture:
    """
    Immutable container for captured attention data from a forward pass.

    All tensors are detached from computation graph and on CPU.
    """

    # Query vectors: [n_layers, n_heads, n_tokens, head_dim]
    queries: torch.Tensor

    # Key vectors: [n_layers, n_kv_heads, n_tokens, head_dim]
    keys: torch.Tensor

    # Value vectors: [n_layers, n_kv_heads, n_tokens, head_dim]
    values: torch.Tensor

    # Attention patterns: [n_layers, n_heads, n_tokens, n_tokens]
    # patterns[l, h, i, j] = attention from token i to token j at layer l, head h
    patterns: torch.Tensor

    # Metadata
    n_layers: int
    n_heads: int
    n_kv_heads: int
    n_tokens: int
    head_dim: int
    model_name: str

    @property
    def is_gqa(self) -> bool:
        """True if model uses Grouped Query Attention."""
        return self.n_heads != self.n_kv_heads

    @property
    def gqa_ratio(self) -> int:
        """Number of query heads per KV head."""
        return self.n_heads // self.n_kv_heads
```

**Shape Contracts:**

| Tensor     | Shape             | Dtype   | Description             |
| ---------- | ----------------- | ------- | ----------------------- |
| `queries`  | `[L, H_q, T, D]`  | float32 | Query vectors           |
| `keys`     | `[L, H_kv, T, D]` | float32 | Key vectors             |
| `values`   | `[L, H_kv, T, D]` | float32 | Value vectors           |
| `patterns` | `[L, H_q, T, T]`  | float32 | Attention probabilities |

Where:

- `L` = number of layers
- `H_q` = number of query heads
- `H_kv` = number of key/value heads (may differ from H_q in GQA)
- `T` = number of tokens (sequence length)
- `D` = head dimension

**Invariants:**

- All tensors are on CPU
- All tensors are detached (no gradients)
- `patterns` sums to 1.0 along last dimension (softmax output)
- `patterns[l, h, i, j] = 0` for `j > i` (causal masking)

---

### 2.2 AttentionStructure

Semantic description of a model's attention architecture.

```python
@dataclass(frozen=True)
class AttentionStructure:
    """
    Describes the attention structure at a given layer.

    Abstracts away model-specific details into semantic categories.
    """
    n_query_heads: int
    n_kv_heads: int
    head_dim: int

    @property
    def attention_type(self) -> str:
        """Returns 'MHA', 'GQA', or 'MQA'."""
        if self.n_query_heads == self.n_kv_heads:
            return "MHA"  # Multi-Head Attention
        elif self.n_kv_heads == 1:
            return "MQA"  # Multi-Query Attention
        else:
            return "GQA"  # Grouped Query Attention

    @property
    def gqa_ratio(self) -> int:
        """Query heads per KV head."""
        return self.n_query_heads // self.n_kv_heads
```

---

### 2.3 CaptureConfig

Configuration for what to capture during a forward pass.

```python
@dataclass
class CaptureConfig:
    """
    Specifies what attention data to capture.

    Use to control memory usage by capturing only needed components.
    """

    # Which layers to capture (None = all)
    layers: Optional[List[int]] = None

    # Which components to capture
    capture_queries: bool = True
    capture_keys: bool = True
    capture_values: bool = True
    capture_patterns: bool = True

    # Capture stage for Q/K vectors
    # 'pre_rope': Before rotary position embedding (position-agnostic)
    # 'post_rope': After rotary position embedding (position-aware)
    qk_stage: Literal['pre_rope', 'post_rope'] = 'pre_rope'

    def validate(self, model_adapter: 'ModelAdapter') -> None:
        """
        Validate config against model.

        Raises:
            ValueError: If layers out of range or invalid combination
        """
        ...
```

**Validation Rules:**

- `layers` must be in range `[0, model.num_layers())`
- At least one capture flag must be True
- Raises `ValueError` with descriptive message on failure

---

## 3. Model Adapter Interface

### 3.1 ModelAdapter (Abstract)

The public interface for model interaction. Delegates to private implementations.

```python
class ModelAdapter(ABC):
    """
    Model-agnostic interface for transformer instrumentation.

    Public methods define WHAT operations are available.
    Private adapters implement HOW for specific model families.
    """

    @abstractmethod
    def num_layers(self) -> int:
        """Total number of transformer layers."""
        ...

    @abstractmethod
    def num_heads(self, layer: int = 0) -> int:
        """Number of query attention heads at given layer."""
        ...

    @abstractmethod
    def attention_structure(self, layer: int = 0) -> AttentionStructure:
        """
        Returns semantic description of attention at given layer.

        Use this to determine GQA/MQA structure without model-specific logic.
        """
        ...

    @abstractmethod
    def capture(
        self,
        input_ids: torch.Tensor,
        config: CaptureConfig = None
    ) -> AttentionCapture:
        """
        Run forward pass and capture attention data.

        Args:
            input_ids: Token IDs [batch=1, seq_len]
            config: What to capture (default: all components, all layers)

        Returns:
            AttentionCapture with requested data

        Raises:
            ValueError: If config validation fails
            RuntimeError: If capture hooks fail
        """
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """Generate text completion."""
        ...
```

**Guarantees:**

- `capture()` is deterministic given same inputs
- `capture()` does not modify model weights
- All returned tensors are detached and on CPU

---

## 4. Analysis Layer

### 4.1 Design Contract

All analysis functions follow this pattern:

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

---

### 4.2 NormAnalysis

Compute vector norms from captured data.

```python
class NormAnalysis:
    """Query, Key, Value norm computations."""

    @staticmethod
    def q_norms(capture: AttentionCapture) -> torch.Tensor:
        """
        L2 norms of query vectors.

        Args:
            capture: AttentionCapture from model

        Returns:
            Tensor of shape [n_layers, n_heads, n_tokens]

        Formula:
            ||Q||₂ = √(Σᵢ Qᵢ²)
        """
        ...

    @staticmethod
    def k_norms(capture: AttentionCapture) -> torch.Tensor:
        """
        L2 norms of key vectors.

        Returns:
            Tensor of shape [n_layers, n_kv_heads, n_tokens]
        """
        ...

    @staticmethod
    def v_norms(capture: AttentionCapture) -> torch.Tensor:
        """
        L2 norms of value vectors.

        Returns:
            Tensor of shape [n_layers, n_kv_heads, n_tokens]
        """
        ...
```

**Shape Contracts:**

| Method      | Output Shape   | Description                      |
| ----------- | -------------- | -------------------------------- |
| `q_norms()` | `[L, H_q, T]`  | Query norms per layer/head/token |
| `k_norms()` | `[L, H_kv, T]` | Key norms per layer/head/token   |
| `v_norms()` | `[L, H_kv, T]` | Value norms per layer/head/token |

---

### 4.3 EntropyAnalysis

Compute entropy metrics from attention patterns.

```python
class EntropyAnalysis:
    """Attention entropy computations."""

    @staticmethod
    def token_entropy(
        capture: AttentionCapture,
        ignore_first: int = 2
    ) -> torch.Tensor:
        """
        Shannon entropy of attention distribution per token.

        Args:
            capture: AttentionCapture from model
            ignore_first: Set first N tokens to NaN (insufficient context)

        Returns:
            Tensor of shape [n_layers, n_heads, n_tokens]

        Formula:
            H(t) = -Σⱼ₌₀ᵗ p(j|t) · log(p(j|t))

            Where p(j|t) = attention weight from token t to token j
            Only sums over j ≤ t (causal masking)

        Interpretation:
            - High entropy → diffuse attention (attending to many tokens)
            - Low entropy → focused attention (attending to few tokens)
            - H = 0 → deterministic attention (100% on one token)
            - H = log(t) → uniform attention over t tokens
        """
        ...

    @staticmethod
    def effective_rank(capture: AttentionCapture) -> torch.Tensor:
        """
        Effective number of attended tokens.

        Returns:
            Tensor of shape [n_layers, n_heads, n_tokens]

        Formula:
            eff_rank = exp(H)

            Where H is Shannon entropy.

        Interpretation:
            If eff_rank = 3.5, attention is spread as if uniformly
            attending to ~3.5 tokens.
        """
        ...

    @staticmethod
    def delta_entropy(
        capture_treatment: AttentionCapture,
        capture_control: AttentionCapture,
        method: Literal['mean', 'final'] = 'mean'
    ) -> torch.Tensor:
        """
        Difference in entropy between two conditions.

        Args:
            capture_treatment: Capture under treatment condition
            capture_control: Capture under control condition
            method:
                'mean': Compare mean entropy across positions
                'final': Compare entropy at final token only

        Returns:
            Tensor of shape [n_layers, n_heads]

        Formula:
            ΔH = mean(H_treatment) - mean(H_control)

        Interpretation:
            - ΔH > 0 → Treatment causes more diffuse attention
            - ΔH < 0 → Treatment causes more focused attention
        """
        ...
```

**Shape Contracts:**

| Method             | Output Shape  | Description               |
| ------------------ | ------------- | ------------------------- |
| `token_entropy()`  | `[L, H_q, T]` | Entropy per position      |
| `effective_rank()` | `[L, H_q, T]` | Effective attended tokens |
| `delta_entropy()`  | `[L, H_q]`    | Delta between conditions  |

---

### 4.4 AttentionAnalysis

Compute attention pattern metrics.

```python
class AttentionAnalysis:
    """Attention pattern analysis."""

    @staticmethod
    def max_weight(capture: AttentionCapture) -> torch.Tensor:
        """
        Maximum attention weight per token.

        Returns:
            Tensor of shape [n_layers, n_heads, n_tokens]

        Formula:
            max_attn(t) = maxⱼ≤ₜ p(j|t)
        """
        ...

    @staticmethod
    def effective_span(
        capture: AttentionCapture,
        threshold: float = 0.9
    ) -> torch.Tensor:
        """
        Minimum tokens needed to capture threshold of attention mass.

        Args:
            capture: AttentionCapture from model
            threshold: Cumulative attention threshold (default 0.9 = 90%)

        Returns:
            Tensor of shape [n_layers, n_heads, n_tokens]
            Values are integers (as float32 for consistency)

        Formula:
            k_eff(t) = argmin_k { Σᵢ₌₁ᵏ sorted_weights[i] ≥ threshold }

            Where sorted_weights are attention weights sorted descending.

        Interpretation:
            If k_eff = 5, the top 5 tokens account for ≥90% of attention.
        """
        ...

    @staticmethod
    def attention_to_position(
        capture: AttentionCapture,
        target_positions: List[int]
    ) -> torch.Tensor:
        """
        Attention weight to specific token positions.

        Args:
            capture: AttentionCapture from model
            target_positions: List of position indices to measure

        Returns:
            Tensor of shape [n_layers, n_heads, n_tokens, len(target_positions)]
        """
        ...
```

**Shape Contracts:**

| Method                    | Output Shape     | Description                     |
| ------------------------- | ---------------- | ------------------------------- |
| `max_weight()`            | `[L, H_q, T]`    | Peak attention per token        |
| `effective_span()`        | `[L, H_q, T]`    | k_eff metric                    |
| `attention_to_position()` | `[L, H_q, T, P]` | Attention to P target positions |

---

### 4.5 CorrelationAnalysis

Statistical correlation between metrics.

```python
class CorrelationAnalysis:
    """Cross-metric correlation analysis."""

    @staticmethod
    def pearson(
        x: torch.Tensor,
        y: torch.Tensor,
        dim: int = -1
    ) -> torch.Tensor:
        """
        Pearson correlation coefficient.

        Args:
            x, y: Tensors of same shape
            dim: Dimension to correlate along

        Returns:
            Correlation tensor with dim reduced

        Formula:
            r = Σ(x - μₓ)(y - μᵧ) / (σₓ · σᵧ · n)
        """
        ...

    @staticmethod
    def spearman(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Spearman rank correlation."""
        ...

    @staticmethod
    def norm_entropy_correlation(capture: AttentionCapture) -> torch.Tensor:
        """
        Correlation between query norm and attention entropy.

        Returns:
            Tensor of shape [n_layers, n_heads]

        This is the core metric for the Oculi hypothesis:
        Strong negative correlation suggests Q magnitude controls attention focus.
        """
        ...
```

---

## 5. Intervention Layer

### 5.1 Intervention Semantics

**CRITICAL DEFINITION:**

When we "scale Q by α", we mean:

```python
Q_scaled = α · Q
```

This is **raw magnitude scaling**, NOT direction-preserving normalization.

**Mathematical Consequence:**

Scaling Q by α affects attention logits as:

```
logits_scaled = (αQ) · Kᵀ / √d = α · (Q · Kᵀ / √d) = α · logits_original
```

After softmax, this sharpens (α > 1) or flattens (α < 1) attention.

The **Oculi Method** scales both Q and K by √α to achieve equivalent effect:

```
logits = (√α · Q) · (√α · K)ᵀ / √d = α · logits_original
```

---

### 5.2 QScaler

```python
@dataclass
class QScaler:
    """
    Intervention that scales query vectors at a specific head.

    Semantic Definition:
        Q_new = alpha · Q_original

    Effect on Attention:
        alpha > 1.0 → Sharpen attention (increase focus)
        alpha < 1.0 → Flatten attention (decrease focus)
        alpha = 1.0 → No change (identity)

    This is RAW magnitude scaling, not direction-preserving.
    The attention logits scale linearly with alpha.
    """
    layer: int
    head: int
    alpha: float

    def validate(self, adapter: ModelAdapter) -> None:
        """
        Validate intervention parameters.

        Raises:
            ValueError: If layer or head out of range
            ValueError: If alpha <= 0
        """
        if not 0 <= self.layer < adapter.num_layers():
            raise ValueError(f"Layer {self.layer} out of range [0, {adapter.num_layers()})")
        if not 0 <= self.head < adapter.num_heads(self.layer):
            raise ValueError(f"Head {self.head} out of range [0, {adapter.num_heads(self.layer)})")
        if self.alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {self.alpha}")
```

---

### 5.3 KScaler

```python
@dataclass
class KScaler:
    """
    Intervention that scales key vectors at a specific KV-head.

    For GQA models: Affects all query heads sharing this KV-head.

    Semantic Definition:
        K_new = alpha · K_original
    """
    layer: int
    kv_head: int  # Note: KV-head index, not query head
    alpha: float
```

---

### 5.4 OculiScaler

```python
@dataclass
class OculiScaler:
    """
    The Oculi intervention: scale both Q and K by √α.

    Semantic Definition:
        Q_new = √α · Q_original
        K_new = √α · K_original

    Net Effect:
        logits_new = α · logits_original

    This achieves the same attention sharpening as scaling logits by α,
    but intervenes at the representation level rather than post-computation.

    Mathematical Justification:
        softmax(α · QKᵀ/√d) ≈ softmax((√α·Q)(√α·K)ᵀ/√d)
    """
    layer: int
    head: int
    alpha: float  # Net scaling factor (internal uses √α on each)
```

---

### 5.5 HeadAblation

```python
@dataclass
class HeadAblation:
    """
    Intervention that zeros out a head's output.

    Semantic Definition:
        head_output_new = 0

    Use for: Measuring causal importance of specific heads.
    """
    layer: int
    head: int
```

---

### 5.6 InterventionContext

```python
class InterventionContext:
    """
    Context manager for applying interventions during generation.

    Usage:
        scaler = OculiScaler(layer=23, head=5, alpha=1.5)

        with InterventionContext(adapter, [scaler]):
            output = adapter.generate(prompt)

    Interventions are automatically removed on context exit.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        interventions: List[Union[QScaler, KScaler, OculiScaler, HeadAblation]]
    ):
        ...

    def __enter__(self) -> 'InterventionContext':
        """Apply all interventions."""
        ...

    def __exit__(self, *args) -> None:
        """Remove all interventions."""
        ...
```

---

## 6. Visualization Layer

### 6.1 Design Contract

Visualization functions return `matplotlib.Figure` objects. User controls saving.

```python
def plot_function(data: torch.Tensor, **params) -> matplotlib.Figure:
    """
    Pure function: Tensor → Figure

    - No side effects (no plt.show(), no saving)
    - Returns Figure for user to display/save
    - Uses consistent Oculi color scheme
    """
```

---

### 6.2 Available Plots

```python
class EntropyPlots:
    """Entropy visualization."""

    @staticmethod
    def heatmap(
        entropy: torch.Tensor,
        title: str = "Attention Entropy"
    ) -> Figure:
        """
        Entropy heatmap across layers and heads.

        Args:
            entropy: Shape [n_layers, n_heads] (mean across positions)

        Returns:
            Figure with heatmap, layer on Y-axis, head on X-axis
        """
        ...

    @staticmethod
    def distribution(
        entropy: torch.Tensor,
        title: str = "Entropy Distribution"
    ) -> Figure:
        """
        Histogram of entropy values across all positions.
        """
        ...


class InterventionPlots:
    """Intervention effect visualization."""

    @staticmethod
    def alpha_curve(
        alphas: List[float],
        metric_values: List[float],
        metric_name: str = "Accuracy",
        title: str = "Intervention Effect"
    ) -> Figure:
        """
        Plot metric vs alpha (the "Goldilocks curve").

        Annotates peak value and location.
        """
        ...

    @staticmethod
    def delta_entropy_heatmap(
        delta_entropy: torch.Tensor,
        title: str = "ΔEntropy (Treatment - Control)"
    ) -> Figure:
        """
        Heatmap of entropy change per layer/head.

        Diverging colormap: red = increase, blue = decrease.
        """
        ...


class CorrelationPlots:
    """Correlation visualization."""

    @staticmethod
    def scatter(
        x: torch.Tensor,
        y: torch.Tensor,
        x_label: str,
        y_label: str,
        title: str = "Correlation"
    ) -> Figure:
        """
        Scatter plot with regression line and correlation coefficient.
        """
        ...
```

---

## 7. Failure Modes

### 7.1 Expected Errors

| Condition          | Exception               | Message                                      |
| ------------------ | ----------------------- | -------------------------------------------- |
| Layer out of range | `ValueError`            | "Layer {n} out of range [0, {max})"          |
| Head out of range  | `ValueError`            | "Head {n} out of range [0, {max})"           |
| Alpha ≤ 0          | `ValueError`            | "Alpha must be positive, got {alpha}"        |
| Invalid QK stage   | `ValueError`            | "qk_stage must be 'pre_rope' or 'post_rope'" |
| Unsupported model  | `UnsupportedModelError` | "No adapter for model: {name}"               |
| Hook failure       | `RuntimeError`          | "Capture failed at layer {n}: {details}"     |

### 7.2 Edge Cases

| Input                    | Behavior                              |
| ------------------------ | ------------------------------------- |
| Sequence length = 1      | `entropy[0] = NaN` (no prior context) |
| Empty layers list        | Capture all layers                    |
| `ignore_first > seq_len` | All entropy values = NaN              |
| GQA head mapping         | K/V operations use KV-head index      |

---

## 8. Versioning Policy

### 8.1 Semantic Versioning

Format: `MAJOR.MINOR.PATCH`

**MAJOR (1.0 → 2.0):**

- Changing tensor output shapes
- Renaming public classes/methods
- Changing mathematical definitions
- Breaking `AttentionCapture` structure

**MINOR (1.0 → 1.1):**

- Adding new analysis methods
- Adding new model adapters
- New optional parameters with defaults
- New visualization functions

**PATCH (1.0.0 → 1.0.1):**

- Bug fixes in private layer
- Performance improvements
- Documentation fixes
- Test additions

### 8.2 Pre-1.0 Warning

> [!CAUTION]
> Versions `0.x.y` are pre-release. Public API may change between minor versions.
> Production use not recommended until `1.0.0`.

---

## 9. Private Layer Scope

### 9.1 Directory Structure

```
oculi/_private/
├── adapters/           # Model-specific implementations
│   ├── llama.py        # LlamaForCausalLM adapter
│   ├── mistral.py      # Mistral adapter
│   ├── qwen.py         # Qwen2/2.5 adapter
│   └── falcon.py       # Falcon adapter
│
├── hooks/              # PyTorch hook machinery
│   ├── capture.py      # Forward hook registration
│   ├── intervention.py # Intervention hook implementation
│   └── utils.py        # Hook utilities
│
├── cache/              # Memory optimization
│   └── activation.py   # Activation caching
│
└── validation/         # Sanity checks
    └── checks.py       # Internal validation
```

### 9.2 Private Layer Rules

1. **PRs Accepted For:** New model adapters, bug fixes, performance improvements
2. **PRs NOT Accepted For:** Changes that affect public API semantics
3. **Internal Refactoring:** Allowed freely without version bumps
4. **No Public Imports:** Nothing from `_private/` appears in public `__init__.py`

---

## 10. Explicit Non-Guarantees

**We do NOT guarantee:**

1. **Performance** — Capture may be memory-intensive; optimize in private layer
2. **Internal Tensor Layouts** — Private tensors may have different shapes
3. **Hook Ordering** — Order of hook execution is implementation detail
4. **GPU Memory** — Large sequences may require chunking (user responsibility)
5. **Thread Safety** — Single-threaded capture assumed
6. **Gradient Flow** — All captures are inference-only (no backprop)

---

## 11. Contract Test Examples

These tests **define the contract** independent of implementation.

```python
# tests/contract_tests/test_shapes.py

def test_q_norms_shape():
    """Verify q_norms returns documented shape."""
    capture = mock_capture(n_layers=4, n_heads=8, n_tokens=16, head_dim=64)
    norms = NormAnalysis.q_norms(capture)
    assert norms.shape == (4, 8, 16)  # [L, H, T]


def test_token_entropy_shape():
    """Verify entropy returns documented shape."""
    capture = mock_capture(n_layers=4, n_heads=8, n_tokens=16, head_dim=64)
    entropy = EntropyAnalysis.token_entropy(capture)
    assert entropy.shape == (4, 8, 16)  # [L, H, T]


def test_entropy_causal_masking():
    """Verify entropy respects causal structure."""
    # Token 0 attends only to itself → zero entropy
    # Token 1 attends uniformly to 0,1 → H = log(2)
    patterns = torch.tensor([[
        [[1.0, 0.0, 0.0],
         [0.5, 0.5, 0.0],
         [0.33, 0.33, 0.34]]
    ]])  # [1, 1, 3, 3]

    capture = AttentionCapture(
        patterns=patterns,
        # ... other fields
    )

    entropy = EntropyAnalysis.token_entropy(capture, ignore_first=0)

    assert torch.isclose(entropy[0, 0, 0], torch.tensor(0.0), atol=1e-3)
    assert torch.isclose(entropy[0, 0, 1], torch.tensor(0.693), atol=1e-2)  # log(2)


def test_oculi_scaler_symmetry():
    """Verify OculiScaler applies √α to both Q and K."""
    # This is a semantic test: we verify the EFFECT, not implementation
    # With α=4, Q and K should each be scaled by 2
    # Net effect on logits = 4x
    pass  # Implementation in integration tests
```

---

## Appendix A: Mathematical Reference

### Shannon Entropy

```
H(t) = -Σⱼ₌₀ᵗ p(j|t) · log(p(j|t))

Where:
- t = query token position
- j = key token position (j ≤ t due to causal masking)
- p(j|t) = attention probability from t to j
```

### Effective Rank

```
eff_rank = exp(H)

Interpretation: Number of tokens if attention were uniform
```

### Effective Span (k_eff)

```
k_eff(t) = min{k : Σᵢ₌₁ᵏ sorted_p[i] ≥ 0.9}

Where sorted_p is attention weights sorted in descending order
```

### L2 Norm

```
||Q||₂ = √(Σᵢ Qᵢ²)
```

### Pearson Correlation

```
r = Σᵢ(xᵢ - μₓ)(yᵢ - μᵧ) / (n · σₓ · σᵧ)
```

---

## Appendix B: Supported Models

| Model Family | Adapter          | Attention Type   | Notes               |
| ------------ | ---------------- | ---------------- | ------------------- |
| LLaMA 2/3    | `LlamaAdapter`   | GQA (4:1 or 8:1) | Primary support     |
| Mistral      | `MistralAdapter` | GQA (4:1)        | Sliding window      |
| Qwen 2/2.5   | `QwenAdapter`    | GQA (7:1)        | Via TransformerLens |
| Falcon       | `FalconAdapter`  | MQA (71:1)       | Multi-query         |

---

_End of API Contract Specification_
