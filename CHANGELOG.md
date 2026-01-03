# Changelog

All notable changes to Oculi will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MkDocs documentation site with Material theme
- Comprehensive user guides and tutorials
- CONTRIBUTING.md with development guidelines
- Examples directory with working code samples
- Enhanced README with Phase 2 features

## [0.5.0-dev] - 2025-01-03

### Added
- **Attribution Methods** (`oculi/analysis/attribution.py`):
  - `attention_flow()` - Track information flow through attention layers
  - `value_weighted_attention()` - Weight attention by value magnitude
  - `direct_logit_attribution()` - Layer contribution to target logit
  - `component_attribution()` - Decompose into attention vs MLP contributions
  - `head_attribution()` - Per-head logit contribution
  - `top_attributions()` - Get top-k attributions with indices
  - 18 unit tests covering all attribution methods

- **Head Composition Analysis** (`oculi/analysis/composition.py`):
  - `qk_composition()` - Measure Q-K composition between head pairs
  - `ov_composition()` - Measure O-V composition for value flow
  - `virtual_attention()` - Compute effective attention through multi-head paths
  - `path_patching_score()` - Estimate path importance for output
  - `composition_matrix()` - Full head-to-head composition matrix
  - `detect_induction_circuit()` - Automatic detection of induction head circuits
  - 16 unit tests covering all composition methods

### Changed
- Updated `oculi/analysis/__init__.py` to export new attribution and composition classes
- Enhanced test suite to 85 total tests (all passing)

### Documentation
- Added comprehensive docstrings for all new methods
- Updated roadmap to reflect Phase 2 progress

## [0.4.0] - 2024-12-XX

### Added
- **Logit Capture** (`oculi/capture/structures.py`):
  - `LogitCapture` data structure for layer-wise logits
  - `LogitConfig` for configuring logit capture
  - Memory-efficient top-k storage option

- **Logit Lens Analysis** (`oculi/analysis/logit_lens.py`):
  - `layer_predictions()` - Top-k predictions at each layer
  - `prediction_convergence()` - Measure prediction stability across layers
  - `token_probability_trajectory()` - Track specific token probability

- **Full Capture** (`oculi/capture/structures.py`):
  - `FullCapture` unified container for all capture types
  - `capture_full()` method for single-pass capture

### Changed
- Extended `AttentionAdapter` interface with `capture_logits()` method
- Improved memory efficiency for long sequences

## [0.3.0] - 2024-11-XX

### Added
- **Residual Stream Capture** (`oculi/capture/structures.py`):
  - `ResidualCapture` data structure
  - `ResidualConfig` for selective capture
  - Four intervention points: pre_attn, post_attn, pre_mlp, post_mlp
  - `capture_residual()` method in adapters

- **MLP Internals Capture** (`oculi/capture/structures.py`):
  - `MLPCapture` data structure for MLP activations
  - `MLPConfig` for selective MLP capture
  - Gate, up, post-activation, and output capture
  - `capture_mlp()` method in adapters

- **Circuit Detection** (`oculi/analysis/circuits.py`):
  - `detect_induction_heads()` - Find A-B-A→B patterns
  - `detect_previous_token_heads()` - Find t→t-1 patterns
  - `detect_positional_heads()` - Find BOS/recent token patterns
  - `classify_attention_head()` - Classify individual heads
  - `summarize_circuits()` - High-level circuit summary

### Changed
- Extended `LlamaAttentionAdapter` with Phase 1 methods
- Improved hook management for multiple capture types
- Enhanced test coverage (51 total tests)

### Documentation
- Added `oculi/models/llama/anatomy.py` with detailed model documentation
- Updated API contract specification
- Added Phase 1 examples to README

## [0.2.0] - 2024-10-XX

### Added
- **Correlation Analysis** (`oculi/analysis/correlation.py`):
  - Pearson and Spearman correlation
  - `norm_entropy_correlation()` for analyzing attention focus
  - P-value computation support

- **Stratified Analysis** (`oculi/analysis/stratified.py`):
  - `StratifiedView` for layer/head/token slicing
  - `find_extreme_heads()` for identifying outlier heads

- **Interventions**:
  - `QScaler`, `KScaler`, `SpectraScaler` for attention scaling
  - `HeadAblation` for zeroing head outputs
  - `InterventionContext` manager for applying interventions

### Changed
- Improved GQA (Grouped Query Attention) handling
- Better error messages for invalid configurations

## [0.1.0] - 2024-09-XX

### Added
- Initial public release
- **Attention Capture**:
  - `AttentionCapture` data structure
  - `LlamaAttentionAdapter` for LLaMA 2/3 models
  - Q/K/V vectors and attention pattern capture
  - Pre/post RoPE capture options

- **Basic Analysis**:
  - `NormAnalysis` for Q/K/V vector norms
  - `EntropyAnalysis` for attention entropy
  - `AttentionAnalysis` for pattern metrics

- **Visualization**:
  - Entropy heatmaps
  - Attention pattern plots
  - Correlation visualizations

- **Testing**:
  - Contract test framework
  - Mock LLaMA adapter for CPU testing
  - 20 initial tests

### Documentation
- Initial README with examples
- API contract specification
- Basic usage documentation

---

## Version History

- **0.5.0-dev** (Current) - Phase 2.1 & 2.2: Attribution & Composition
- **0.4.0** - Phase 1 Complete: Logit Lens & Full Capture
- **0.3.0** - Phase 1: Residual Stream, MLP Capture, Circuit Detection
- **0.2.0** - Interventions & Advanced Analysis
- **0.1.0** - Initial Release: Attention Capture & Basic Analysis

## Roadmap

### Phase 2 (v0.5.0 - v0.6.0) - In Progress
- ✅ Attribution methods
- ✅ Head composition analysis
- ⏳ Activation patching
- ⏳ SAE integration
- ⏳ Probing & steering vectors

### Phase 3 (v0.7.0 - v0.8.0) - Planned
- Caching system
- Memory optimization (FP16, lazy materialization)
- Export formats (HDF5, JSON, NumPy)
- TransformerLens compatibility

### Phase 4 (v1.0.0) - Future
- API freeze
- Complete documentation
- Benchmark suite
- Production-ready release

---

## Links

- **Repository:** https://github.com/ajayspatil7/oculi
- **Documentation:** https://github.com/ajayspatil7/oculi#readme
- **Issues:** https://github.com/ajayspatil7/oculi/issues
- **Discussions:** https://github.com/ajayspatil7/oculi/discussions

---

[Unreleased]: https://github.com/ajayspatil7/oculi/compare/v0.4.0...HEAD
[0.5.0-dev]: https://github.com/ajayspatil7/oculi/compare/v0.4.0...v0.5.0-dev
[0.4.0]: https://github.com/ajayspatil7/oculi/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ajayspatil7/oculi/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ajayspatil7/oculi/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ajayspatil7/oculi/releases/tag/v0.1.0
