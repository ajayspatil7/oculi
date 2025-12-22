# Spectra Phase 1 Checklist

**Objective**: Empirically verify whether Query Norm (‖Q‖) predicts Attention Entropy across layers and heads in Llama-3-8B.

---

## ✅ Completed

### Environment & Repository

- [x] **Initialize Git repository** — Created project structure, pushed to GitHub
- [x] **Define experiment config** — `src/config.py` with frozen hyperparameters (fp16, 4K context, batch=1)
- [x] **Requirements file** — `requirements.txt` with PyTorch, transformers, scipy, matplotlib

### Experiment Zero (Validation)

- [x] **Basic inference script** — `notebooks/experiment_zero/basic_inference.py`
- [x] **Validate on SageMaker** — Tesla T4, model loads (12.83 GB), inference works (1.24 tok/s)
- [x] **Model dissection script** — `notebooks/experiment_zero/dissect_model.py`
  - Architecture overview (32 layers, 32 heads, 8 KV heads, GQA)
  - Q/K/V projection visualization
  - Manual attention computation (step-by-step)
  - Query norm computation (‖Q‖₂)
  - Attention entropy computation (mask-aware, NaN-safe)
  - Per-head correlation demo

### Core Implementation

- [x] **Data loader** — `src/data_loader.py` with sample text, file loading, dataset support
- [x] **Attention hooks** — `src/hooks.py` with `AttentionProfiler` class for all 32 layers
- [x] **Metrics module** — `src/metrics.py` with query norm, entropy, correlations, randomization control
- [x] **Main experiment script** — `scripts/run_experiment.py` complete pipeline
- [x] **Visualization script** — `scripts/visualize.py` with heatmaps, scatter, histograms

---

## 🔲 To Do

### Execution

- [ ] **Run full experiment on SageMaker** — `python scripts/run_experiment.py --context-length 4096`
- [ ] **Generate visualizations** — `python scripts/visualize.py --latest`
- [ ] **Verify randomization control** — Check shuffled correlations → ~0

### Deliverables

- [ ] **Write interpretation** — Document findings in `results/FINDINGS.md`
- [ ] **Go/No-Go decision** — Based on |r| ≥ 0.5, p < 0.01 criteria
- [ ] **Final commit** — Tag as `phase1-complete`

---

## Success Criteria (Fixed Before Analysis)

| Metric                   | Threshold                           |
| ------------------------ | ----------------------------------- |
| Correlation magnitude    | \|r\| ≥ 0.5 in meaningful subset    |
| Statistical significance | p < 0.01                            |
| Randomization control    | Shuffled correlations → ~0          |
| Reproducibility          | Results hold across multiple inputs |

---

## File Structure

```
Spectra/
├── src/
│   ├── config.py        ✅ Done
│   ├── hooks.py         ✅ Done
│   ├── metrics.py       ✅ Done
│   └── data_loader.py   ✅ Done
├── scripts/
│   ├── run_experiment.py    ✅ Done
│   └── visualize.py         ✅ Done
├── notebooks/
│   └── experiment_zero/
│       ├── basic_inference.py   ✅ Done
│       └── dissect_model.py     ✅ Done
├── results/                     🔲 To Do (experiment outputs)
├── CHECKLIST.md                 ✅ This file
└── README.md                    ✅ Done
```

---

## Quick Start

```bash
# On SageMaker, after git pull:
cd ~/Spectra

# Run the full experiment
python scripts/run_experiment.py --context-length 4096

# Generate visualizations
python scripts/visualize.py --latest

# Results will be in results/ directory
```
