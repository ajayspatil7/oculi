# Spectra: Attention Metrics Data Capture

## Objective

Capture detailed per-token attention metrics from Llama-3-8B to understand attention patterns in large language models.

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/ajayspatil7/spectra.git
cd spectra
pip install -r requirements.txt

# Download and preprocess data
python scripts/preprocess_data.py --n-samples 50000 --output-dir data/processed

# Capture metrics (64 samples → ~768 MB CSV)
python scripts/run_experiment.py --context-length 512 --n-samples 64

# View configuration details
python scripts/show_experiment_config.py
```

---

## Metrics Captured

For each **token** in each **head** of each **layer**:

| Metric               | Symbol | Description                                   |
| -------------------- | ------ | --------------------------------------------- |
| Query Norm           | ‖Q‖    | L2 norm of query vector                       |
| Attention Entropy    | H      | Information entropy of attention distribution |
| Max Attention Weight | α_max  | Peak attention weight assigned                |
| Effective Span       | k_eff  | Min keys needed for 90% attention mass        |

---

## Experimental Configuration

| Parameter              | Value                                 |
| ---------------------- | ------------------------------------- |
| **Model**              | meta-llama/Meta-Llama-3-8B            |
| **Precision**          | float16                               |
| **Context Length**     | 512 tokens                            |
| **Architecture**       | 32 layers, 32 heads (GQA: 8 KV heads) |
| **Attention Analysis** | Pre-RoPE (position-agnostic)          |
| **GPU Requirement**    | CUDA (≥16 GB VRAM)                    |

---

## Output Format

**CSV File:** `results/attention_metrics_{timestamp}.csv`

**Columns:**

- `sample_id`, `layer`, `head`, `token_pos`
- `query_norm`, `entropy`, `max_attn`, `k_eff`

**Size for 64 samples:**

- Rows: **33,554,432**
- File size: **~768 MB**

---

## Project Structure

```
Spectra/
├── src/
│   ├── config.py           # Experiment configuration
│   ├── hooks.py            # AttentionProfiler (captures Q/K/V)
│   ├── metrics.py          # Metric computations
│   └── data_loader.py      # Data loading from shards
├── scripts/
│   ├── run_experiment.py   # Main capture script
│   ├── preprocess_data.py  # SlimPajama-6B preprocessing
│   ├── explore_npz.py      # Shard exploration tool
│   └── show_experiment_config.py  # Configuration display
├── data/
│   ├── raw/                # Downloaded dataset
│   └── processed/          # .npz token shards
└── results/                # CSV output files
```

---

## Key Design Decisions

### Pre-RoPE Analysis

Query and Key vectors are captured **before** Rotary Position Embedding (RoPE) is applied. This analyzes the **intrinsic geometry** of queries in position-agnostic space.

**Rationale:**

- RoPE is a rotation (preserves norms)
- Isolates semantic vs positional components
- ‖Q_before_RoPE‖ = ‖Q_after_RoPE‖

### Manual Attention Recomputation

We compute `attention = softmax(Q @ K^T / sqrt(d))` manually rather than using FlashAttention.

**Rationale:**

- FlashAttention doesn't expose intermediate attention weights
- Ensures consistency between Q norms and attention probabilities
- Full control over causal masking

### Metrics Definition

**Query Norm (‖Q‖):**

```
‖Q‖ = sqrt(Σ Q_i²)  (L2 norm along head dimension)
```

**Attention Entropy (H):**

```
H = -Σ p_i · log(p_i)  (Shannon entropy over valid attention weights)
```

**Effective Span (k_eff):**

```
k_eff = argmin_k { Σ(top-k weights) ≥ 0.9 }
```

---

## Data Source

**Dataset:** SlimPajama-6B (test split)

**Filtered Sources:**

- CommonCrawl (general web)
- C4 (cleaned web text)
- Wikipedia (encyclopedic)
- StackExchange (Q&A)

**Excluded:** GitHub (code), ArXiv (papers), Books

**Preprocessing:**

- 512-token fixed-length chunks
- Non-overlapping (no padding)
- Llama-3-8B tokenizer

---

## Environment Setup

### Prerequisites

- **GPU:** NVIDIA with ≥16 GB VRAM (A10G recommended)
- **CUDA:** 12.1+
- **Python:** 3.10+

### Installation

```bash
conda create -n spectra python=3.10 -y
conda activate spectra
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Verify Setup

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## Running on AWS SageMaker

```bash
# Launch ml.g5.xlarge instance (or higher)
git clone https://github.com/ajayspatil7/spectra.git
cd spectra
pip install -r requirements.txt

# Preprocess data (one-time, ~5 min)
python scripts/preprocess_data.py --n-samples 50000

# Capture metrics (~15-20 min for 64 samples)
python scripts/run_experiment.py --n-samples 64
```

---

## Usage Examples

### Full Pipeline

```bash
# 1. Preprocess 50K samples
python scripts/preprocess_data.py --n-samples 50000

# 2. Capture metrics from 64 samples
python scripts/run_experiment.py --n-samples 64

# 3. View configuration
python scripts/show_experiment_config.py

# 4. Explore output
python scripts/explore_npz.py data/processed/shard_00000.npz
```

### Custom Configuration

```bash
# Different sample count
python scripts/run_experiment.py --n-samples 128

# Different context length (must match preprocessing)
python scripts/run_experiment.py --context-length 512 --n-samples 50
```

---

## Expected Runtime (A10G GPU)

| Operation                    | Time           |
| ---------------------------- | -------------- |
| Preprocessing (50K samples)  | ~5 minutes     |
| Metrics capture (64 samples) | ~15-20 minutes |
| Per sample processing        | ~15-20 seconds |

---

## License

MIT

---

## Author

Ajay S Patil
