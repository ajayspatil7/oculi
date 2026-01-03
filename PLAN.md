Oculi Extension: Comprehensive Mechanistic Interpretability Features

Executive Summary

Goal: Extend Oculi from attention-focused toolkit to full mechanistic interpretability platform

Test: DO NOT TEST ANYTHING LOCALLY, TEST ON MOCK MODEL I am on MacBook Pro M4 Pro

Current State: v0.2.0 - Mature attention capture (Q/K/V/patterns), clean architecture, GQA-aware
Target State: v1.0.0 - Full forward pass instrumentation, circuit discovery, causal interventions, production-ready

Timeline: 4 phases over ~15 months (Q1 2026 - Q1 2027)

Strategy: Maintain backward compatibility through v0.x, lock API at v1.0

---

Architectural Decisions

1.  Data Structure Strategy: Parallel Capture Structures (Not Extension)

Decision: Keep AttentionCapture unchanged, create companion structures

# Current (unchanged)

AttentionCapture # [L, H, T, D] for Q/K/V/patterns - stays through v1.0

# New parallel structures (Phase 1)

ResidualCapture # Residual stream at intervention points
MLPCapture # MLP internals (pre/post activation, neuron-level)
LogitCapture # Layer-wise logits for logit lens

# Unified accessor (Phase 2)

FullCapture # Optional container holding all capture types

Rationale: Memory control, backward compatibility, clear semantics, gradual migration path

2.  Module Organization: Semantic Grouping

oculi/
├── capture/ # EXTEND: Add ResidualCapture, MLPCapture, LogitCapture
├── analysis/ # EXTEND: Add circuits.py, attribution.py, logit_lens.py, composition.py
├── intervention/ # EXTEND: Add patching.py, steering.py, residual.py
├── models/llama/ # EXTEND: Add anatomy.py, extend adapter.py
├── cache/ # NEW: Phase 3 - caching system
├── export/ # NEW: Phase 3 - export formats
└── compat/ # NEW: Phase 3 - tool integrations

3.  Version Strategy

v0.2.0 (current) → Attention-focused
v0.3.0 (Phase 1) → +Residual/MLP capture
v0.4.0 (Phase 1) → +Circuit detection, +Logit capture
v0.5.0 (Phase 2) → +Attribution methods
v0.6.0 (Phase 2) → +Activation patching
v0.7.0 (Phase 3) → +Caching, optimization
v0.8.0 (Phase 3) → Stabilization
v1.0.0 (Phase 4) → API lock, production-ready

Breaking changes allowed in v0.x with migration guides. Semantic versioning strict at v1.0+

---

Phase 1: Forward Pass Instrumentation (v0.3.0 - v0.4.0)

Duration: 6-8 weeks | Priority: HIGH | Dependencies: None

Objectives

1.  Capture residual stream activations
2.  Capture MLP internals
3.  Implement circuit detection primitives
4.  Enable logit lens analysis

1.1 Residual Stream Capture (v0.3.0)

New Data Structure:
@dataclass(frozen=True)
class ResidualCapture:
"""Residual stream at key intervention points."""
pre_attn: Optional[torch.Tensor] # [L, T, H] - Before attention
post_attn: Optional[torch.Tensor] # [L, T, H] - After attn, before MLP
pre_mlp: Optional[torch.Tensor] # [L, T, H] - Before MLP
post_mlp: Optional[torch.Tensor] # [L, T, H] - After MLP (= pre_attn[L+1])

     n_layers: int
     n_tokens: int
     hidden_dim: int
     model_name: str
     captured_layers: tuple

Hook Locations (LLaMA):

- model.model.layers[i] - Hook on full transformer block (input/output)
- Pre-attention: Block input
- Post-attention: After self_attn but before add & norm
- Pre-MLP: After first layer norm
- Post-MLP: After MLP output before final add & norm

Files to Create/Modify:

- oculi/capture/structures.py - Add ResidualCapture, ResidualConfig
- oculi/models/llama/adapter.py - Add capture_residual() method
- oculi/models/llama/anatomy.py - NEW: Document residual hook points
- oculi/\_private/hooks/capture.py - Add residual hook creators

  1.2 MLP Capture (v0.3.0)

New Data Structure:
@dataclass(frozen=True)
class MLPCapture:
"""MLP internals per layer.""" # LLaMA: hidden → 4*hidden (gate + up) → hidden (down)
pre_activation: Optional[torch.Tensor] # [L, T, 4*H] - Before SiLU
post_activation: Optional[torch.Tensor] # [L, T, 4*H] - After SiLU
neuron_activations: Optional[torch.Tensor] # = post_activation
mlp_output: Optional[torch.Tensor] # [L, T, H] - Final output

     n_layers: int
     n_tokens: int
     hidden_dim: int
     intermediate_dim: int  # 4 * hidden_dim for LLaMA

Hook Locations (LLaMA):

- model.model.layers[i].mlp.gate_proj - Gate projection
- model.model.layers[i].mlp.up_proj - Up projection
- model.model.layers[i].mlp.act_fn - Activation function (SiLU)
- model.model.layers[i].mlp.down_proj - Down projection

Files to Create/Modify:

- oculi/capture/structures.py - Add MLPCapture, MLPConfig
- oculi/models/llama/adapter.py - Add capture_mlp() method
- oculi/models/llama/anatomy.py - Document MLP structure

  1.3 Circuit Detection Primitives (v0.4.0)

New Analysis Module:

# oculi/analysis/circuits.py

class CircuitDetection:
"""Detect canonical circuit patterns in transformers."""

     @staticmethod
     def detect_induction_heads(capture: AttentionCapture, threshold: float = 0.5) -> torch.Tensor:
         """Detect [A][B]...[A] → attend to [B] pattern. Returns [L, H] boolean."""

     @staticmethod
     def detect_previous_token_heads(capture: AttentionCapture, threshold: float = 0.8) -> torch.Tensor:
         """Detect heads attending primarily to previous token. Returns [L, H] boolean."""

     @staticmethod
     def detect_copy_heads(capture: AttentionCapture, threshold: float = 0.7) -> torch.Tensor:
         """Detect heads that copy information (high attn + value preservation)."""

     @staticmethod
     def detect_positional_heads(capture: AttentionCapture) -> Dict[str, torch.Tensor]:
         """Detect heads attending to specific positions (BOS, EOS, first, last)."""

Validation: Test against known induction head datasets (toy models with known circuits)

Files to Create:

- oculi/analysis/circuits.py - NEW: Circuit detection methods

  1.4 Logit Capture (v0.4.0)

New Data Structure:
@dataclass(frozen=True)
class LogitCapture:
"""Layer-wise logits for logit lens."""
logits: torch.Tensor # [L, T, V] - Logits after each layer

     # Optional: memory-efficient top-k
     top_k_logits: Optional[torch.Tensor]   # [L, T, K]
     top_k_tokens: Optional[torch.Tensor]   # [L, T, K]

     n_layers: int
     n_tokens: int
     vocab_size: int

Implementation: Apply unembed matrix to residual stream at each layer

# residual: [L, T, H]

# unembed: [V, H]

logits[layer] = residual[layer] @ unembed.T # [T, V]

New Analysis Module:

# oculi/analysis/logit_lens.py

class LogitLensAnalysis:
"""Logit lens and tuned lens analysis."""

     @staticmethod
     def layer_predictions(logit_capture: LogitCapture, top_k: int = 10) -> List[List[str]]:
         """Top-k predictions at each layer."""

     @staticmethod
     def prediction_convergence(logit_capture: LogitCapture) -> torch.Tensor:
         """Measure prediction stability across layers. Returns [L] divergence."""

     @staticmethod
     def token_probability_trajectory(logit_capture: LogitCapture, token_id: int) -> torch.Tensor:
         """Track specific token's probability across layers. Returns [L, T]."""

Files to Create:

- oculi/capture/structures.py - Add LogitCapture
- oculi/models/llama/adapter.py - Add capture_logits() method
- oculi/analysis/logit_lens.py - NEW: Logit lens analysis

---

Phase 2: Attribution & Advanced Analysis (v0.5.0 - v0.6.0)

Duration: 8-10 weeks | Priority: HIGH | Dependencies: Phase 1

Objectives

1.  Implement attribution methods
2.  Enable activation patching
3.  Analyze head composition
4.  Understand information flow

2.1 Attribution Methods (v0.5.0)

New Analysis Module:

# oculi/analysis/attribution.py

class AttributionMethods:
"""Causal attribution for transformer internals."""

     @staticmethod
     def attention_flow(capture: AttentionCapture, residual: ResidualCapture) -> torch.Tensor:
         """Track information flow through attention. Returns [L, L, T] flow matrix."""

     @staticmethod
     def value_weighted_attention(capture: AttentionCapture) -> torch.Tensor:
         """Attention weighted by value contribution (not just patterns). Returns [L, H, T, T]."""

     @staticmethod
     def direct_logit_attribution(residual: ResidualCapture, logit_capture: LogitCapture,
                                   target_token_id: int) -> torch.Tensor:
         """Direct effect of each layer on target logit. Returns [L] contribution."""

     @staticmethod
     def integrated_gradients(adapter: AttentionAdapter, input_ids: torch.Tensor,
                             target_token: int, n_steps: int = 50) -> torch.Tensor:
         """Integrated gradients attribution (requires gradient mode). Returns [T, H]."""

Files to Create:

- oculi/analysis/attribution.py - NEW: Attribution methods

  2.2 Head Composition Analysis (v0.5.0)

New Analysis Module:

# oculi/analysis/composition.py

class CompositionAnalysis:
"""Analyze how heads compose via OV/QK circuits."""

     @staticmethod
     def ov_circuit_matrix(adapter: AttentionAdapter, layer: int, head: int) -> torch.Tensor:
         """W_V @ W_O for a head. Returns [head_dim, hidden_dim] - what head writes."""

     @staticmethod
     def qk_circuit_matrix(adapter: AttentionAdapter, layer: int, head: int) -> torch.Tensor:
         """W_Q @ W_K^T for a head. Returns [hidden_dim, hidden_dim] - what patterns head detects."""

     @staticmethod
     def composition_score(ov_upstream: torch.Tensor, qk_downstream: torch.Tensor) -> float:
         """Measure how upstream head's output feeds downstream input. Returns similarity score."""

     @staticmethod
     def find_composed_heads(adapter: AttentionAdapter, threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
         """Find (layer1, head1, layer2, head2) pairs with strong composition."""

Files to Create:

- oculi/analysis/composition.py - NEW: Composition analysis

  2.3 Activation Patching (v0.6.0)

New Intervention System:

# oculi/intervention/patching.py

@dataclass
class ActivationPatch:
"""Replace activations from source run."""
layer: int
component: str # 'attn_out', 'mlp_out', 'residual_post_attn', etc.
source_activation: torch.Tensor # From corrupted/clean run
patch_tokens: Optional[List[int]] = None # None = all tokens

     def validate(self, adapter):
         """Validate layer, component, shape."""

class PatchingContext:
"""Context manager for activation patching (extends InterventionContext pattern)."""

     def __init__(self, adapter: AttentionAdapter, patches: List[ActivationPatch]):
         self.adapter = adapter
         self.patches = patches

     def __enter__(self):
         """Register patching hooks."""

     def __exit__(self, exc_type, exc_val, exc_tb):
         """Remove hooks."""

class PatchingExperiment:
"""Systematic patching experiments."""

     def run_causal_tracing(self, clean_input: torch.Tensor, corrupted_input: torch.Tensor,
                           metric_fn: Callable, components: List[str]) -> Dict[str, float]:
         """Systematic patching to find causal components. Returns {component: metric}."""

Usage Pattern:

# Clean and corrupted runs

clean_residual = adapter.capture_residual(clean_input)
corrupted_residual = adapter.capture_residual(corrupted_input)

# Patch layer 20 from corrupted into clean

patch = ActivationPatch(
layer=20,
component='residual_post_attn',
source_activation=corrupted_residual.post_attn[20]
)

with PatchingContext(adapter, [patch]):
patched_output = adapter.generate(clean_input)

Files to Create/Modify:

- oculi/intervention/patching.py - NEW: Patching interventions
- oculi/intervention/context.py - MODIFY: Extend to support patching
- oculi/capture/structures.py - Add PatchConfig

---

Phase 3: Optimization & Production (v0.7.0 - v0.8.0)

Duration: 6-8 weeks | Priority: MEDIUM | Dependencies: Phase 1, 2

Objectives

1.  Implement caching for expensive captures
2.  Optimize memory usage
3.  Add export formats
4.  Integrate with existing tools

3.1 Caching System (v0.7.0)

New Module:

# oculi/cache/manager.py

class CaptureCache:
"""Disk-backed cache with LRU eviction."""

     def __init__(self, cache_dir: Path, max_size_gb: float = 10.0):
         self.cache_dir = cache_dir
         self.max_size = max_size_gb

     def get(self, input_ids: torch.Tensor, config: CaptureConfig) -> Optional[AttentionCapture]:
         """Retrieve cached capture (SHA256-based lookup)."""

     def put(self, input_ids: torch.Tensor, config: CaptureConfig, capture: AttentionCapture):
         """Store capture to disk (HDF5 format)."""

     @contextmanager
     def cached_capture(self, adapter, input_ids, config):
         """Auto-caching context manager."""

Files to Create:

- oculi/cache/manager.py - NEW: Caching implementation
- oculi/cache/**init**.py - NEW

  3.2 Memory Optimization (v0.7.0)

Strategy:

1.  Selective materialization (GPU→CPU only when accessed)
2.  Chunked processing for long sequences
3.  FP16 storage option

Config Extension:
@dataclass
class CaptureConfig: # Existing fields...

     # NEW: Memory optimization
     materialize_immediately: bool = True  # False = lazy GPU→CPU
     storage_dtype: torch.dtype = torch.float32  # Or float16
     chunk_size: Optional[int] = None  # For sequences > chunk_size

Files to Modify:

- oculi/capture/structures.py - Extend configs with memory options

  3.3 Export Formats (v0.7.0)

New Module:

# oculi/export/**init**.py

class CaptureExporter:
@staticmethod
def to_hdf5(capture: AttentionCapture, path: Path):
"""Export to HDF5 (efficient for large tensors)."""

     @staticmethod
     def to_json(capture: AttentionCapture, path: Path):
         """Export metadata + tensor references."""

     @staticmethod
     def to_numpy(capture: AttentionCapture, dir: Path):
         """Export as .npz (numpy-compatible)."""

Files to Create:

- oculi/export/**init**.py - NEW: Export utilities

  3.4 Tool Integration (v0.8.0)

TransformerLens Compatibility:

# oculi/compat/transformer_lens.py

def to_transformer_lens(capture: AttentionCapture) -> ActivationCache:
"""Convert Oculi → TransformerLens format."""

def from_transformer_lens(cache: ActivationCache) -> AttentionCapture:
"""Import TransformerLens → Oculi."""

Circuitsvis Integration:

# oculi/visualize/circuits.py

def plot_attention_circuitsvis(capture: AttentionCapture, layer: int) -> HTML:
"""Render with circuitsvis library."""

Files to Create:

- oculi/compat/transformer_lens.py - NEW
- oculi/compat/**init**.py - NEW
- oculi/visualize/circuits.py - NEW

---

Phase 4: Stabilization & v1.0 (v0.8.0 → v1.0.0)

Duration: 4-6 weeks | Priority: MEDIUM | Dependencies: All phases

Objectives

1.  API audit and freeze
2.  Comprehensive documentation
3.  Benchmark suite
4.  Contract test expansion

4.1 API Finalization

- Review all public APIs for consistency
- Document all shape contracts
- Finalize deprecations from v0.x
- Ensure error messages are helpful

  4.2 Documentation Overhaul

- Complete API reference (every method)
- Tutorial notebooks for each major feature
- Migration guide v0.x → v1.0
- Architecture deep dive

  4.3 Benchmark Suite

# tests/benchmarks/

def benchmark_capture_speed() # Overhead vs vanilla forward pass
def benchmark_memory_usage() # Memory profiling
def benchmark_intervention_overhead() # Generation slowdown

4.4 Contract Test Expansion

- Test every shape contract
- Test GQA edge cases (different ratios)
- Test long sequence handling
- Test error messages

---

Implementation Priority Order

Tier 1 (Absolutely Essential - Phase 1):

1.  ResidualCapture (foundation for everything)
2.  MLPCapture (see full model)
3.  LogitCapture (understand layer-by-layer)
4.  CircuitDetection (validate capture system)

Tier 2 (Highly Valuable - Phase 2):

5.  AttributionMethods (understand causality)
6.  CompositionAnalysis (OV/QK circuits)
7.  ActivationPatching (causal interventions)

Tier 3 (Production Polish - Phase 3):

8.  CaptureCache (performance)
9.  MemoryOptimization (scalability)
10. ExportFormats (reproducibility)
11. ToolIntegration (ecosystem)

---

Critical Files Reference

Files to Create (New):

oculi/models/llama/anatomy.py # Full model anatomy documentation
oculi/analysis/circuits.py # Circuit detection
oculi/analysis/attribution.py # Attribution methods
oculi/analysis/logit_lens.py # Logit lens analysis
oculi/analysis/composition.py # Head composition
oculi/intervention/patching.py # Activation patching
oculi/intervention/steering.py # Steering vectors (future)
oculi/intervention/residual.py # Residual interventions (future)
oculi/cache/manager.py # Caching system
oculi/cache/**init**.py
oculi/export/**init**.py # Export utilities
oculi/compat/transformer_lens.py # TransformerLens integration
oculi/compat/**init**.py
oculi/visualize/circuits.py # Circuit visualization
tests/contract_tests/test_shapes.py # Shape contract tests
tests/integration/test_circuits.py # Circuit detection tests
tests/integration/test_patching.py # Patching tests
tests/benchmarks/benchmark_memory.py # Memory benchmarks
tests/benchmarks/benchmark_speed.py # Speed benchmarks

Files to Modify (Extend):

oculi/capture/structures.py # Add ResidualCapture, MLPCapture, LogitCapture
oculi/models/llama/adapter.py # Add capture_residual(), capture_mlp(), capture_logits()
oculi/models/base.py # Extend interface for new capture methods
oculi/intervention/context.py # Support patching interventions
oculi/**init**.py # Export new structures
oculi/analysis/**init**.py # Export new analysis classes
oculi/intervention/**init**.py # Export new intervention types

---

Testing Strategy

Three-Tier Testing Pyramid:

Tier 1: Contract Tests (fast, comprehensive)

- Shape contracts: Every function's output shape verified
- Invariants: Mathematical properties (entropy sums, attention sums to 1, etc.)
- Semantics: Behavior contracts (causal masking, GQA expansion, etc.)
- Uses tiny mock models (deterministic, fast)

Tier 2: Integration Tests (medium speed)

- Real capture workflows with mock LLaMA
- Circuit detection on known circuits
- Patching experiments with ground truth
- Mix of mock + small real models (LLaMA-160M if available)

Tier 3: Performance Tests (slow, CI optional)

- Memory profiling with real models (LLaMA-3-8B)
- Speed benchmarks
- Long sequence handling
- Only runs on GPU machines

Validation Strategy:

- Circuit detection: Test against known induction head datasets
- Activation patching: Replicate published results (causal tracing papers)
- Attribution: Compare against established methods (TransformerLens)

---

Memory Management Strategy

Problem: Full forward pass capture is massive (LLaMA-3-8B, 512 tokens ≈ 50GB)

Solution: Multi-Level Approach

1.  Selective Capture (user controls what to capture)
    config = FullCaptureConfig(
    attention=True,
    residual=['post_attn', 'post_mlp'], # Only key stages
    mlp=False, # Skip if not needed
    layers=[20, 21, 22] # Subset of layers
    )
2.  Lazy Materialization (GPU→CPU on access)
    capture.queries # Property that transfers on first access, not during forward
3.  Chunked Processing (for long sequences)
    for chunk in long_sequence.chunks(512):
    chunk_capture = adapter.capture(chunk) # Process and discard
4.  FP16 Storage (half memory, minimal precision loss)
    config = CaptureConfig(storage_dtype=torch.float16)

---

API Evolution Example

Current (v0.2.0):
from oculi.models.llama import LlamaAttentionAdapter
adapter = LlamaAttentionAdapter(model, tokenizer)
capture = adapter.capture(input_ids)

Phase 1 (v0.3-0.4) - Backward compatible:

# Old API still works

capture = adapter.capture(input_ids)

# New capabilities

residual = adapter.capture_residual(input_ids)
mlp = adapter.capture_mlp(input_ids)
logits = adapter.capture_logits(input_ids)

# Circuit detection

from oculi.analysis import CircuitDetection
induction_heads = CircuitDetection.detect_induction_heads(capture)

Phase 2 (v0.5-0.6) - Advanced features:

# Attribution

from oculi.analysis import AttributionMethods
flow = AttributionMethods.attention_flow(capture, residual)

# Patching

from oculi.intervention import ActivationPatch, PatchingContext
patch = ActivationPatch(layer=20, component='mlp_out', source=corrupted.mlp_output[20])
with PatchingContext(adapter, [patch]):
output = adapter.generate(clean_input)

Phase 3 (v0.7) - Production features:
from oculi import CaptureCache

cache = CaptureCache(cache_dir=".cache")
with cache.cached_capture(adapter, input_ids, config) as capture:
circuits = CircuitDetection.analyze_all(capture)

v1.0 - Stable API:

# All v0.x APIs still work (deprecated warnings removed)

# API locked, semantic versioning enforced

---

Risk Mitigation

Technical Risks:

| Risk                      | Impact | Mitigation                                               |
| ------------------------- | ------ | -------------------------------------------------------- |
| Memory explosion          | HIGH   | Selective capture, chunking, FP16, lazy materialization  |
| Patching breaks models    | HIGH   | Extensive validation, restore guarantees, contract tests |
| API churn alienates users | MEDIUM | Deprecation ladder, migration guides, v0.x flexibility   |
| Performance degradation   | MEDIUM | Benchmarks in CI, dedicated optimization phase           |

Design Risks:

| Risk                           | Impact | Mitigation                                               |
| ------------------------------ | ------ | -------------------------------------------------------- |
| Over-abstraction loses clarity | MEDIUM | Keep anatomy.py files explicit and educational           |
| Feature bloat                  | LOW    | Strict tier prioritization, defer non-essential features |
| Future model incompatibility   | MEDIUM | Adapter pattern designed for extensibility               |

---

Success Metrics

Phase 1:

- ResidualCapture correctly captures all 4 stream stages
- MLPCapture works on LLaMA-2/3
- Circuit detection finds known induction heads (validation dataset)
- Logit lens shows layer-by-layer prediction improvement
- Memory usage < 2x attention-only capture
- All contract tests pass

Phase 2:

- Attribution methods replicate published results
- Activation patching recovers known circuits
- Head composition identifies known composition patterns
- Performance <20% slower than Phase 1

Phase 3:

- Cache hit rate >80% in typical workflow
- Memory optimization enables 2x longer sequences
- Export/import preserves data (bit-exact round-trip)
- TransformerLens compatibility verified

v1.0:

- Zero breaking changes v0.8 → v1.0
- 100% API documentation coverage
- Tutorial for every major feature
- 3+ external projects using Oculi
- Migration guide tested

---

Timeline

2026 Q1 (Jan-Mar): Phase 1 Planning & Implementation
Week 1-2: Architecture finalization, API design
Week 3-6: ResidualCapture + MLPCapture
Week 7-10: LogitCapture + CircuitDetection
Week 11-12: Testing, v0.3.0-0.4.0 releases

2026 Q2 (Apr-Jun): Phase 1 Completion + Phase 2 Start
Week 13-14: Circuit validation
Week 15-18: Attribution methods
Week 19-22: Head composition
Week 23-24: v0.5.0 release

2026 Q3 (Jul-Sep): Phase 2 Completion
Week 25-30: Activation patching (complex!)
Week 31-34: Steering + residual interventions
Week 35-36: v0.6.0 release

2026 Q4 (Oct-Dec): Phase 3 Optimization
Week 37-40: Caching system
Week 41-44: Memory optimization + export
Week 45-48: Tool integration, v0.7.0 release

2027 Q1 (Jan-Mar): Stabilization & v1.0
Week 49-52: API audit
Week 53-56: Documentation overhaul
Week 57-60: Benchmarks + contract tests
Week 61-64: Beta testing, v1.0.0 release

---

Next Steps (When Implementation Begins)

1.  Week 1-2: Finalize architecture decisions, review this plan with stakeholders
2.  Week 3: Create oculi/models/llama/anatomy.py (model documentation)
3.  Week 4-5: Implement ResidualCapture + tests
4.  Week 6-7: Implement MLPCapture + tests
5.  Week 8-9: Implement LogitCapture + logit lens analysis
6.  Week 10-11: Implement circuit detection primitives
7.  Week 12: Integration testing, v0.3.0-0.4.0 releases

---

Conclusion

This plan extends Oculi from an attention visualization toolkit to a comprehensive mechanistic interpretability platform while maintaining:

- Backward compatibility through v0.x
- Architectural consistency with learning-first design
- Incremental delivery via 4 phases
- Production readiness by v1.0

The phased approach allows for learning from usage patterns and adjusting priorities based on community feedback before the v1.0 API freeze.
