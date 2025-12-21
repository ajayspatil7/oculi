# Experiment Zero: Basic Llama Inference

## Purpose

Validate that the environment is correctly set up for Phase 1 experiments:

- CUDA is available and working
- Llama-3-8B loads successfully with fp16 precision
- Basic inference runs without errors
- Memory usage is within bounds

## Prerequisites

1. **HuggingFace Access**: Llama-3 requires access approval

   ```bash
   huggingface-cli login
   ```

2. **GPU Requirements**: ≥24 GB VRAM recommended for Llama-3-8B in fp16

## Files

| File                 | Description                                |
| -------------------- | ------------------------------------------ |
| `basic_inference.py` | Main inference script with CUDA validation |

## Running on SageMaker

```bash
# Activate environment
conda activate phase1_llm

# Run the experiment
python basic_inference.py
```

## Expected Output

1. GPU info (device name, memory, CUDA version)
2. Model loading confirmation with timing
3. Memory usage before/after loading
4. Sample text generation
5. Inference timing (tokens/second)

## Success Criteria

- ✅ No CUDA errors
- ✅ Model loads in < 5 minutes
- ✅ Memory usage < 20 GB for fp16
- ✅ Inference produces coherent output
