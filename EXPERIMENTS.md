# SPECTRA — Experiment File Mapping

## Summary

This document maps all files involved in each experiment:

- **Source files** (scripts/modules used to run experiments)
- **Output files** (results generated after running)

---

## 📁 Directory Structure (Reorganized)

```
Spectra/
├── experiments/
│   ├── exp0_observation/     # EXP0: Observational correlation analysis
│   │   ├── analysis/         # Analysis notebooks
│   │   │   └── analysis_outputs/  # Correlation plots & CSVs
│   │   ├── dissect_model.py
│   │   └── basic_inference.py
│   │
│   ├── exp1_intervention/    # EXP1: Causal Q intervention
│   │   ├── *.csv             # Intervention results
│   │   └── *.png             # Plots
│   │
│   ├── exp2_gain/            # EXP2: Gain/sensitivity analysis
│   │   ├── compute_gain.py
│   │   ├── gain_summary.csv
│   │   └── *.png             # Fit plots
│   │
│   └── exp3_key_scaling/     # EXP3: Q vs K control experiment
│       ├── run_qk_comparison.py
│       ├── *.csv             # Q vs K data
│       └── *.png             # Comparison plots
│
├── scripts/                   # Standalone run scripts
│   ├── run_experiment.py     # EXP0 data collection
│   ├── run_intervention.py   # EXP1 intervention
│   └── plot_intervention.py  # EXP1 plotting
│
├── src/                       # Core modules
│   ├── hooks.py              # AttentionProfiler
│   ├── metrics.py            # Entropy, max_attn, k_eff
│   ├── intervention.py       # InterventionProfiler
│   └── data_loader.py        # Data loading
│
└── results/                   # Legacy (now empty, moved to experiments/)
```

---

## EXP0 — Observational Analysis

**Purpose:** Measure correlation between Q norm and attention entropy.

### Source Files

| File                  | Purpose                     | Location                       |
| --------------------- | --------------------------- | ------------------------------ |
| `run_experiment.py`   | Main data collection script | `scripts/`                     |
| `hooks.py`            | AttentionProfiler class     | `src/`                         |
| `metrics.py`          | Entropy, max_attn, k_eff    | `src/`                         |
| `dissect_model.py`    | Analysis notebook           | `experiments/experiment_zero/` |
| `basic_inference.py`  | Testing inference           | `experiments/experiment_zero/` |
| `main_analysis.ipynb` | Analysis notebook           | `experiments/analysis/`        |
| `heavy.ipynb`         | Heavy analysis              | `experiments/analysis/`        |

### Output Files

| File                    | Description                       | Location                                 |
| ----------------------- | --------------------------------- | ---------------------------------------- |
| `head_summary.csv`      | Per-head correlation statistics   | `results/`                               |
| `raw_data_*.pkl`        | Raw captured data                 | `results/`                               |
| `consistent_heads.csv`  | Heads with consistent correlation | `experiments/analysis/analysis_outputs/` |
| `significant_heads.csv` | Statistically significant heads   | `experiments/analysis/analysis_outputs/` |
| `layer_statistics.csv`  | Layer-level statistics            | `experiments/analysis/analysis_outputs/` |
| `*.png` plots           | Visualizations                    | `experiments/analysis/analysis_outputs/` |

---

## EXP1 — Causal Intervention on Query Norm

**Purpose:** Scale Q vectors and measure causal effect on entropy.

### Source Files

| File                   | Purpose                    | Location   |
| ---------------------- | -------------------------- | ---------- |
| `intervention.py`      | InterventionProfiler class | `src/`     |
| `run_intervention.py`  | Main experiment script     | `scripts/` |
| `plot_intervention.py` | Generate plots from CSV    | `scripts/` |

### Output Files

| File                                         | Description            | Location                |
| -------------------------------------------- | ---------------------- | ----------------------- |
| `q_scale_intervention_L12_H0_target_*.csv`   | Target head results    | `results/intervention/` |
| `q_scale_intervention_L12_H15_control_*.csv` | Control head results   | `results/intervention/` |
| `entropy_vs_scale_L12_H*.png`                | Entropy response plots | `results/intervention/` |
| `max_attn_vs_scale_L12_H*.png`               | Max attention plots    | `results/intervention/` |
| `keff_vs_scale_L12_H*.png`                   | k_eff plots            | `results/intervention/` |
| `intervention_combined_L12_H*.png`           | Combined 3-panel plots | `results/intervention/` |

---

## EXP2 — Gain Analysis

**Purpose:** Quantify sensitivity (gain) of each head to Q scaling.

### Source Files

| File              | Purpose                 | Location            |
| ----------------- | ----------------------- | ------------------- |
| `compute_gain.py` | Gain calculation script | `experiments/gain/` |

### Output Files

| File                           | Description               | Location            |
| ------------------------------ | ------------------------- | ------------------- |
| `gain_summary.csv`             | Gain values for all heads | `experiments/gain/` |
| `gain_fit_L12_H0_target.png`   | Target head linear fit    | `experiments/gain/` |
| `gain_fit_L12_H15_control.png` | Control head linear fit   | `experiments/gain/` |
| `gain_comparison.png`          | Bar chart comparison      | `experiments/gain/` |

---

## EXP3 — Q vs K Scaling Control

**Purpose:** Compare Q-scaling vs K-scaling to test query-specificity.

### Source Files

| File                   | Purpose                  | Location                   |
| ---------------------- | ------------------------ | -------------------------- |
| `run_qk_comparison.py` | Q vs K comparison script | `experiments/key-scaling/` |

### Output Files

| File                                | Description                 | Location                   |
| ----------------------------------- | --------------------------- | -------------------------- |
| `qk_comparison_results.csv`         | Q vs K data                 | `experiments/key-scaling/` |
| `qk_entropy_comparison_L12_H0.png`  | Raw entropy comparison      | `experiments/key-scaling/` |
| `qk_delta_entropy_L12_H0.png`       | Delta entropy (publication) | `experiments/key-scaling/` |
| `qk_max_attn_comparison_L12_H0.png` | Max attention comparison    | `experiments/key-scaling/` |
| `qk_keff_comparison_L12_H0.png`     | k_eff comparison            | `experiments/key-scaling/` |
| `sensitivity_summary.csv`           | Q/K sensitivity ratio       | `experiments/key-scaling/` |

---

## Core Modules (`src/`)

| File              | Purpose                                          |
| ----------------- | ------------------------------------------------ |
| `hooks.py`        | AttentionProfiler — captures Q, K, V, attention  |
| `metrics.py`      | compute_entropy, compute_max_attn, compute_k_eff |
| `intervention.py` | InterventionProfiler — Q scaling intervention    |
| `data_loader.py`  | Load preprocessed data shards                    |
| `config.py`       | ExperimentConfig, InterventionConfig             |

---

## Run Commands Summary

```bash
# EXP0: Observational analysis
python scripts/run_experiment.py

# EXP1: Causal intervention
python scripts/run_intervention.py --target-layer 12 --target-head 0
python scripts/plot_intervention.py --input results/intervention/q_scale_intervention_results.csv

# EXP2: Gain analysis
python experiments/gain/compute_gain.py

# EXP3: Q vs K comparison
python experiments/key-scaling/run_qk_comparison.py --target-layer 12 --target-head 0
```
