# EXP1 — Causal Intervention on Query Norm

**Status:** ✅ Completed

## Purpose

Scale Q vectors directly and measure causal effect on attention entropy.

- Q → αQ (scale only Q)
- K, V held constant
- Proves causality, not just correlation

## What This Establishes

- Query norm is a **causal variable** controlling attention entropy
- Monotonic, smooth response
- First hard causal result

## Key Files

| File                           | Description                |
| ------------------------------ | -------------------------- |
| `scripts/run_intervention.py`  | Main experiment script     |
| `scripts/plot_intervention.py` | Plot generator             |
| `src/intervention.py`          | InterventionProfiler class |

## Results in This Directory

| File                                   | Description            |
| -------------------------------------- | ---------------------- |
| `q_scale_intervention_*_target_*.csv`  | Target head results    |
| `q_scale_intervention_*_control_*.csv` | Control head results   |
| `entropy_vs_scale_*.png`               | Entropy response plots |
| `max_attn_vs_scale_*.png`              | Max attention plots    |
| `keff_vs_scale_*.png`                  | k_eff plots            |
| `intervention_combined_*.png`          | 3-panel combined plots |

## Run Commands

```bash
# Target head
python scripts/run_intervention.py --target-layer 12 --target-head 0

# Control head
python scripts/run_intervention.py --target-layer 12 --target-head 15 --head-type control

# Generate plots
python scripts/plot_intervention.py --input experiments/exp1_intervention/q_scale_intervention_results.csv
```
