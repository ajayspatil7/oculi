# EXP2 — Gain Analysis (Sensitivity Analysis)

**Status:** ✅ Completed

## Purpose

Quantify how strongly each head responds to Q scaling.

- Gain = d(attention sharpness) / d(log α)
- Identifies heads with high vs low sensitivity

## What This Establishes

- Gain is highly heterogeneous across heads
- Some heads: very sensitive | Some heads: nearly flat
- Scaling is selectively exploited, not uniform
- Kills the "trivial softmax math" objection

## Key Files

| File              | Description             |
| ----------------- | ----------------------- |
| `compute_gain.py` | Gain computation script |

## Results in This Directory

| File                    | Description               |
| ----------------------- | ------------------------- |
| `gain_summary.csv`      | Gain values for all heads |
| `gain_fit_L12_H*_*.png` | Linear fit plots          |
| `gain_comparison.png`   | Bar chart comparison      |

## Run Command

```bash
python experiments/exp2_gain/compute_gain.py
```
