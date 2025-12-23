# Key-Scaling Control Experiment

**Purpose:** Demonstrate that **query norm**, not generic logit magnitude, functions as the primary control signal for attention entropy.

## The Argument

A skeptical reviewer could say:

> "Scaling Q just changes logit magnitude. Any scaling would have the same effect."

We kill that argument by showing:

- **Q-scaling** → strong, structured entropy response
- **K-scaling** → weaker or qualitatively different response

If both were identical, the claim collapses. If Q dominates, the claim is strengthened.

## Usage

```bash
python notebooks/key-scaling/run_qk_comparison.py \
    --target-layer 12 --target-head 0
```

## Outputs (all saved here)

| File                           | Description                             |
| ------------------------------ | --------------------------------------- |
| `qk_comparison_results.csv`    | Raw results for all alphas              |
| `qk_entropy_comparison_*.png`  | **Primary plot**: Q vs K entropy curves |
| `qk_max_attn_comparison_*.png` | Max attention comparison                |
| `qk_keff_comparison_*.png`     | Effective span comparison               |
| `sensitivity_summary.csv`      | Sensitivity ratios (Q/K)                |

## Expected Outcome

| Metric            | Expected                      |
| ----------------- | ----------------------------- |
| Q range / K range | > 1.2 (Q has stronger effect) |
| Slope ratio       | > 1.0 (Q is steeper)          |

## Interpretation Language

> "While both query and key scaling affect attention logits, entropy exhibits significantly higher sensitivity to query scaling than key scaling. This asymmetry indicates that residual query norm functions as a primary control signal for attention sharpness."
