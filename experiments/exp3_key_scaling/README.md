# EXP3 — Q vs K Scaling Control

**Status:** ✅ Completed

## Purpose

Compare Q-scaling vs K-scaling to determine if the effect is query-specific or just generic logit magnitude.

## What This Establishes

- On L12 H0: Q-scaling ≡ K-scaling (identical curves)
- This head behaves as generic softmax-temperature head
- Entropy depends on logit magnitude, not Q vs K identity
- **Important:** Does not invalidate EXP1/EXP2, identifies one control regime

## Key Files

| File                   | Description              |
| ---------------------- | ------------------------ |
| `run_qk_comparison.py` | Q vs K comparison script |

## Results in This Directory

| File                           | Description               |
| ------------------------------ | ------------------------- |
| `qk_comparison_results.csv`    | Q vs K raw data           |
| `qk_entropy_comparison_*.png`  | Raw entropy comparison    |
| `qk_delta_entropy_*.png`       | Delta entropy (for paper) |
| `qk_max_attn_comparison_*.png` | Max attention comparison  |
| `qk_keff_comparison_*.png`     | k_eff comparison          |
| `sensitivity_summary.csv`      | Q/K sensitivity ratio     |

## Run Command

```bash
python experiments/exp3_key_scaling/run_qk_comparison.py --target-layer 12 --target-head 0
```
