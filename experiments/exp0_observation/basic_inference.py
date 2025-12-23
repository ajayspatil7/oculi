"""
Experiment Zero: Basic Llama Inference Script
==============================================

This script verifies:
1. CUDA availability and GPU configuration
2. Model loading with fp16 precision
3. Basic text generation inference
4. Memory usage tracking

Run this on SageMaker GPU instance to validate environment setup.

Usage:
    python basic_inference.py
"""

import torch
import gc
import time
from pathlib import Path

# Enforce CUDA-only execution
assert torch.cuda.is_available(), (
    "CUDA is not available. This script requires a GPU.\n"
    "Please run on a CUDA-enabled machine (e.g., SageMaker GPU instance)."
)


def print_gpu_info():
    """Print detailed GPU information."""
    print("=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)
    print(f"PyTorch Version:    {torch.__version__}")
    print(f"CUDA Available:     {torch.cuda.is_available()}")
    print(f"CUDA Version:       {torch.version.cuda}")
    print(f"cuDNN Version:      {torch.backends.cudnn.version()}")
    print(f"Device Count:       {torch.cuda.device_count()}")
    print(f"Current Device:     {torch.cuda.current_device()}")
    print(f"Device Name:        {torch.cuda.get_device_name(0)}")
    
    props = torch.cuda.get_device_properties(0)
    print(f"Total Memory:       {props.total_memory / 1e9:.2f} GB")
    print(f"Multi-Processor:    {props.multi_processor_count}")
    print("=" * 60)


def get_memory_stats():
    """Get current GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return allocated, reserved


def print_memory(prefix=""):
    """Print memory usage with optional prefix."""
    allocated, reserved = get_memory_stats()
    print(f"{prefix}Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def load_model_and_tokenizer(model_name: str = "meta-llama/Meta-Llama-3-8B"):
    """
    Load Llama model and tokenizer with fp16 precision.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        tuple: (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\nLoading model: {model_name}")
    print("-" * 40)
    
    start_time = time.time()
    print_memory("Before loading: ")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with fp16 precision
    print("Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",  # Automatically place on GPU
        trust_remote_code=True,
    )
    
    # Ensure model is in eval mode
    model.eval()
    
    load_time = time.time() - start_time
    print_memory("After loading:  ")
    print(f"Load time: {load_time:.2f} seconds")
    
    return model, tokenizer


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 50):
    """
    Run inference on the model.
    
    Args:
        model: Loaded language model
        tokenizer: Tokenizer for the model
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        str: Generated text
    """
    print(f"\nRunning inference...")
    print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
    print("-" * 40)
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to("cuda")
    
    input_length = inputs.input_ids.shape[1]
    print(f"Input tokens: {input_length}")
    
    # Run inference with no gradient computation
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    torch.cuda.synchronize()
    inference_time = time.time() - start_time
    
    # Decode output
    generated_tokens = outputs.shape[1] - input_length
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Generated tokens: {generated_tokens}")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Tokens/second: {generated_tokens / inference_time:.2f}")
    print_memory("After inference: ")
    
    return generated_text


def main():
    """Main entry point for experiment zero."""
    print("\n" + "=" * 60)
    print("EXPERIMENT ZERO: Basic Llama Inference")
    print("=" * 60)
    
    # Step 1: Print GPU info
    print_gpu_info()
    
    # Step 2: Load model
    # NOTE: You may need to authenticate with HuggingFace for Llama access
    # Run: huggingface-cli login
    model, tokenizer = load_model_and_tokenizer()
    
    # Step 3: Run test inference
    test_prompt = (
        "The relationship between attention mechanisms and computational efficiency "
        "in transformer models is an important research topic. Specifically,"
    )
    
    generated = run_inference(model, tokenizer, test_prompt, max_new_tokens=100)
    
    print("\n" + "=" * 60)
    print("GENERATED OUTPUT")
    print("=" * 60)
    print(generated)
    print("=" * 60)
    
    # Step 4: Final memory report
    print("\n" + "=" * 60)
    print("FINAL STATUS")
    print("=" * 60)
    print_memory("Final: ")
    print("✅ Experiment Zero completed successfully!")
    print("=" * 60)
    
    # Cleanup
    del model
    del tokenizer
    clear_memory()
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
