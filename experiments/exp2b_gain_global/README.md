# EXP2b — Global Gain Analysis

**Status:** 🔄 Ready to run

## Purpose

Compute gain (sensitivity to Q scaling) for **ALL heads across ALL layers**.

This produces a complete gain map, enabling:

- Stratification of heads into high/medium/low gain categories
- Principled selection of heads for subsequent experiments (EXP3b)

## What This Fixes

EXP2 (local) only analyzed L12 H0/H15. This was insufficient to:

- Know if those heads are representative
- Select heads for further testing

EXP2b scans the **entire model** (32 layers × 32 heads = 1024 heads).

## Run Command

```bash
python experiments/exp2b_gain_global/compute_global_gain.py
```

**Runtime estimate:** ~30-60 minutes on A10G GPU

## Outputs (all saved here)

| File                        | Description               |
| --------------------------- | ------------------------- |
| `gain_summary_global.csv`   | One row per (layer, head) |
| `gain_heatmap_max_attn.png` | Layer × Head heatmap      |
| `gain_heatmap_entropy.png`  | Entropy gain heatmap      |
| `gain_distribution.png`     | Distribution of gains     |
| `gain_by_layer.png`         | Mean gain per layer       |

## Output Columns

```csv
layer,head,gain_entropy,gain_max_attn,r_squared_entropy,r_squared_max_attn,is_monotonic_entropy,is_monotonic_max_attn,baseline_entropy,baseline_max_attn,gain_category
```

- `gain_category`: high / medium / low (based on percentiles)

## What This Unlocks

After running, you can:

1. Sort by `|gain_max_attn|`
2. Pick high-gain heads → test with EXP3b (Q vs K)
3. Pick low-gain heads → use as true controls
4. Analyze layer-wise patterns
