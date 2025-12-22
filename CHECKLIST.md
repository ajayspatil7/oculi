# Spectra: Attention Metrics Data Capture

## Objective

Capture detailed per-token attention metrics from Llama-3-8B for downstream analysis.

**Metrics Captured:**

- Query L2 norms (‖Q‖)
- Attention entropy (H)
- Maximum attention weights (α_max)
- Effective attention span (k_eff)

---

## ✅ Completed

### Infrastructure

- [x] **Git repository** — Project structure on GitHub
- [x] **Data preprocessing** —scripts/preprocess_data.py` for SlimPajama-6B
  - Full dataset download to `/data/raw`
  - Source filtering (CommonCrawl, C4, Wikipedia, StackExchange)
  - Non-overlapping 512-token chunks
  - NumPy .npz shard output
  - **Status:** 4 shards, 12,000 sequences created

### Core Implementation

- [x] **Attention hooks** — `src/hooks.py` with `AttentionProfiler`
  - Pre-RoPE Q/K/V capture
  - Attention recomputation
  - 32 layers × 32 heads = 1,024 attention heads
- [x] **Metrics module** — `src/metrics.py`
  - Query norm computation (L2)
  - Attention entropy (mask-aware, NaN-safe)
  - Max attention weight computation
  - Effective attention span (k_eff, 90% threshold)
- [x] **Data loader** — `src/data_loader.py`
  - Load from .npz shards
  - Multi-sample support
  - Fallback to single sample
- [x] **Main capture script** — `scripts/run_experiment.py`
  - Per-token metrics collection
  - Incremental CSV writing
  - Progress tracking with ETA
- [x] **Configuration info script** — `scripts/show_experiment_config.py`
- [x] **NPZ explorer** — `scripts/explore_npz.py`

### Experiments

- [x] **64-sample capture** — Processing on A10G GPU
  - Expected output: ~33.5M rows, ~768 MB CSV
  - Metrics: query_norm, entropy, max_attn, k_eff

---

## 📊 Current Status

**Data Ready:**

- 4 shards with 12,000 sequences (512 tokens each)
- ~6.14M tokens from SlimPajama-6B test split

**Script Ready:**

- Captures 4 metrics per (sample, layer, head, token)
- Saves to CSV: `attention_metrics_{timestamp}.csv`

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/ajayspatil7/spectra.git
cd spectra
pip install -r requirements.txt

# Download and preprocess data (one-time)
python scripts/preprocess_data.py --n-samples 50000 --output-dir data/processed

# Run metrics capture (64 samples)
python scripts/run_experiment.py --context-length 512 --n-samples 64

# View experiment configuration
python scripts/show_experiment_config.py

# Explore preprocessed shards
python scripts/explore_npz.py data/processed/shard_00000.npz
```

---

## Output CSV Format

| Column       | Description                                |
| ------------ | ------------------------------------------ |
| `sample_id`  | Sample index (0-63)                        |
| `layer`      | Layer index (0-31)                         |
| `head`       | Head index (0-31)                          |
| `token_pos`  | Token position (0-511)                     |
| `query_norm` | L2 norm of query vector                    |
| `entropy`    | Attention entropy (NaN for first 2 tokens) |
| `max_attn`   | Maximum attention weight                   |
| `k_eff`      | Effective attention span (90% threshold)   |

**Total rows for 64 samples:** 33,554,432

---

## File Structure

```
Spectra/
├── src/
│   ├── config.py           # Experiment configuration
│   ├── hooks.py            # AttentionProfiler (Q/K/V capture)
│   ├── metrics.py          # All metric computations
│   └── data_loader.py      # Data loading utilities
├── scripts/
│   ├── run_experiment.py   # Main metrics capture script
│   ├── preprocess_data.py  # SlimPajama-6B preprocessing
│   ├── explore_npz.py      # NPZ file investigation
│   └── show_experiment_config.py  # Config info display
├── data/
│   ├── raw/                # Downloaded dataset (SlimPajama-6B)
│   └── processed/          # Preprocessed .npz shards
├── results/                # CSV output files
├── CHECKLIST.md            # Progress tracking
└── README.md               # This file
```

---

## Key Design Decisions

### Pre-RoPE Analysis

Query and Key vectors are captured **before** RoPE (Rotary Position Embedding). This analyzes the intrinsic geometry of query vectors in position-agnostic space. RoPE preserves norms (rotation), so ‖Q‖ is unchanged.

### Attention Recomputation

We manually recompute attention (`softmax(Q @ K^T / sqrt(d))`) rather than capturing from FlashAttention, which doesn't expose intermediate weights. This ensures full control and consistency.

### Metrics Captured

- **Query Norm:** L2 norm of Q vectors
- **Entropy:** `-Σ p·log(p)` over valid attention weights
- **Max Attention:** Peak attention weight per token
- **k_eff:** Minimum keys needed for 90% attention mass

---

## Environment Setup

### Prerequisites

- NVIDIA GPU (≥ 16 GB VRAM, 24 GB recommended)
- CUDA 12.1+
- Python 3.10+

### Installation

```bash
conda create -n spectra python=3.10 -y
conda activate spectra
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Running on SageMaker

```bash
# After launching GPU instance (ml.g5.xlarge or higher):
git clone https://github.com/ajayspatil7/spectra.git
cd spectra
pip install -r requirements.txt

# Preprocess data
python scripts/preprocess_data.py --n-samples 50000

# Capture metrics
python scripts/run_experiment.py --n-samples 64
```

---

## License

MIT
