# EXP0 — Observational Analysis

**Status:** ✅ Completed

## Purpose

Measure correlation between query norm (‖Q‖) and attention entropy. Pure observation, no interventions.

## What This Establishes

- Strong, consistent correlation: higher ‖Q‖ → lower entropy (sharper attention)
- Pattern structured by layer and head (not random)
- Motivates causal testing (correlation ≠ causation)

## Key Files

| File                 | Description                    |
| -------------------- | ------------------------------ |
| `dissect_model.py`   | Core analysis script           |
| `basic_inference.py` | Test script                    |
| `analysis/`          | Analysis notebooks and outputs |

## Outputs

Located in `analysis/analysis_outputs/`:

- `consistent_heads.csv` — Heads with consistent correlation
- `significant_heads.csv` — Statistically significant heads
- `layer_statistics.csv` — Per-layer statistics
- Various `.png` plots

## Run Command

```bash
python scripts/run_experiment.py
```
