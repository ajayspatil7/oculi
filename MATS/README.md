# MATS 10.0: Sycophancy Entropy Control

**Research Question:** Does attention entropy act as a causal bottleneck for sycophancy? Can we "force" a model to be faithful to its reasoning by sharpening attention on logic-carrying heads?

## Hypothesis

When a model is "rationalizing" (writing a CoT that justifies a wrong answer suggested by a user):

- **Logic Heads** become diffuse (high entropy) — struggling to bridge truth and forced conclusion
- **Sycophancy Heads** become sharp (low entropy) — locking onto the user's hint

## Model

**Qwen/Qwen2.5-7B-Instruct**

- 28 layers, 28 Q-heads, 4 KV-heads (7:1 GQA ratio)
- TransformerLens for hook-based intervention

## Experiments

| Phase | Experiment      | Hours  | Description                             |
| ----- | --------------- | ------ | --------------------------------------- |
| 0     | Setup           | 0-1    | Environment validation, sanity check    |
| 1     | Sanity          | Hour 1 | ΔEntropy detection (CRITICAL GATE)      |
| 2     | Rationalization | 2-5    | Identify Logic/Sycophancy heads         |
| 3     | Restoration     | 6-9    | Sharpen Logic Heads → measure flip rate |
| 4     | Jamming         | 10-12  | Flatten Sycophancy Heads                |
| 5     | Control         | 13-14  | Verify intervention is non-destructive  |

## Quick Start (SageMaker)

```bash
# Clone and setup
cd /home/ec2-user/SageMaker
git clone <repo> && cd Spectra/MATS
pip install -r requirements.txt

# Run full pipeline
python run_pipeline.py

# Run specific phase
python run_pipeline.py --phase sanity
```

## Results

Results are saved to `results/<timestamp>/` with:

- Entropy heatmaps
- Head identification CSV
- Intervention metrics
- Git commit hash for reproducibility

## Author

**Ajay S Patil**  
MATS 10.0 Application — Neel Nanda Track
