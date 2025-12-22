"""
Attention Metrics Data Capture Script
======================================

This script captures detailed per-token attention metrics from Llama-3-8B.

For each (sample, layer, head, token), captures:
- Query L2 norm
- Attention entropy
- Max attention weight
- Effective attention span
- Token position, layer, head IDs

Saves all metrics to CSV (no correlation analysis).

Usage:
    python scripts/run_experiment.py --context-length 512 --n-samples 50
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_long_context, load_from_shards
from src.hooks import AttentionProfiler
from src.metrics import (
    compute_attention_entropy,
    compute_max_attention_weight,
    compute_effective_attention_span,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Capture Attention Metrics to CSV")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Model name or path")
    parser.add_argument("--context-length", type=int, default=512,
                        help="Context length in tokens")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Number of samples to process")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing preprocessed shards")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for CSV")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def print_gpu_info():
    """Print GPU information."""
    print("=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)


def load_model(model_name: str):
    """Load model and tokenizer."""
    print(f"\nLoading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}, "
          f"Heads: {model.config.num_attention_heads}")
    
    return model, tokenizer


def collect_metrics_for_sample(
    profiler: AttentionProfiler,
    input_ids: torch.Tensor,
    sample_id: int
) -> pd.DataFrame:
    """
    Collect all metrics for a single sample.
    
    Returns DataFrame with one row per (layer, head, token).
    """
    # Profile attention
    layer_data = profiler.profile(input_ids, compute_attn_probs=True)
    
    # Get dimensions
    n_layers = profiler.n_layers
    n_heads = profiler.n_heads
    seq_len = input_ids.shape[1]
    
    # Collect all metrics
    rows = []
    
    for layer_idx in range(n_layers):
        data = layer_data[layer_idx]
        
        # Compute metrics for this layer
        Q = data.query  # [1, n_heads, seq_len, head_dim]
        attn_probs = data.attn_probs  # [1, n_heads, seq_len, seq_len]
        
        # Query norms: [1, n_heads, seq_len]
        q_norms = torch.norm(Q, p=2, dim=-1)
        
        # Attention entropy: [1, n_heads, seq_len]
        entropy = compute_attention_entropy(attn_probs)
        
        # Max attention weight: [1, n_heads, seq_len]
        max_attn = compute_max_attention_weight(attn_probs)
        
        # Effective span: [1, n_heads, seq_len]
        k_eff = compute_effective_attention_span(attn_probs, threshold=0.9)
        
        # Convert to numpy and squeeze batch dimension
        q_norms = q_norms.squeeze(0).cpu().numpy()  # [n_heads, seq_len]
        entropy = entropy.squeeze(0).cpu().numpy()
        max_attn = max_attn.squeeze(0).cpu().numpy()
        k_eff = k_eff.squeeze(0).cpu().numpy()
        
        # Create rows for each (head, token) combination
        for head_idx in range(n_heads):
            for token_idx in range(seq_len):
                rows.append({
                    'sample_id': sample_id,
                    'layer': layer_idx,
                    'head': head_idx,
                    'token_pos': token_idx,
                    'query_norm': q_norms[head_idx, token_idx],
                    'entropy': entropy[head_idx, token_idx],
                    'max_attn': max_attn[head_idx, token_idx],
                    'k_eff': k_eff[head_idx, token_idx],
                })
    
    return pd.DataFrame(rows)


def run_experiment(args):
    """Run the metrics capture experiment."""
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"attention_metrics_{timestamp}.csv"
    
    print("\n" + "=" * 60)
    print("ATTENTION METRICS CAPTURE")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"Context length: {args.context_length}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Output CSV: {csv_path}")
    
    # Check GPU
    print_gpu_info()
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    
    # Load model
    start_time = time.time()
    model, tokenizer = load_model(args.model)
    load_time = time.time() - start_time
    print(f"Model load time: {load_time:.1f}s")
    
    # Prepare input data
    print(f"\nPreparing input data...")
    
    if args.n_samples > 1:
        samples = load_from_shards(
            data_dir=args.data_dir,
            n_samples=args.n_samples,
            device="cuda"
        )
        if samples is None:
            print("Falling back to single sample mode")
            samples = [load_long_context(tokenizer, target_length=args.context_length)]
    else:
        samples = [load_long_context(tokenizer, target_length=args.context_length)]
    
    print(f"Processing {len(samples)} samples...")
    
    # Expected rows
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    seq_len = args.context_length
    total_rows = len(samples) * n_layers * n_heads * seq_len
    print(f"Expected CSV rows: {total_rows:,}")
    
    # Process samples and save incrementally
    first_write = True
    total_time = 0
    
    for sample_idx, inputs in enumerate(tqdm(samples, desc="Processing samples")):
        sample_start = time.time()
        
        # Create profiler for this sample
        profiler = AttentionProfiler(model)
        
        # Collect metrics
        df = collect_metrics_for_sample(profiler, inputs["input_ids"], sample_idx)
        
        # Save to CSV (append mode after first write)
        if first_write:
            df.to_csv(csv_path, index=False, mode='w')
            first_write = False
        else:
            df.to_csv(csv_path, index=False, mode='a', header=False)
        
        sample_time = time.time() - sample_start
        total_time += sample_time
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Progress update
        if (sample_idx + 1) % 10 == 0:
            avg_time = total_time / (sample_idx + 1)
            remaining = (len(samples) - sample_idx - 1) * avg_time
            print(f"  [{sample_idx + 1}/{len(samples)}] "
                  f"Avg: {avg_time:.1f}s/sample, "
                  f"Remaining: {remaining/60:.1f}min")
    
    # Final summary
    print("\n" + "=" * 60)
    print("CAPTURE COMPLETE")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Average time per sample: {total_time/len(samples):.1f}s")
    print(f"Output CSV: {csv_path}")
    print(f"File size: {csv_path.stat().st_size / 1e9:.2f} GB")
    
    # Read and validate
    print("\nValidating CSV...")
    df_final = pd.read_csv(csv_path)
    print(f"Total rows: {len(df_final):,}")
    print(f"Columns: {list(df_final.columns)}")
    print(f"\nFirst few rows:")
    print(df_final.head(10))
    
    print("\n✅ Experiment complete!")
    
    return csv_path


if __name__ == "__main__":
    args = parse_args()
    csv_path = run_experiment(args)
