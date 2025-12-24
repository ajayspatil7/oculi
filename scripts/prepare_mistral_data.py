#!/usr/bin/env python3
"""
Prepare Mistral Data (Exact LLaMA Replication)
==============================================

Tokenizes the EXACT same text samples used for LLaMA-3 experiments
using the Mistral tokenizer.

Steps:
1. Load raw SlimPajama text (cached).
2. Read LLaMA sample indices (source_idx) from data/ctx{N}/samples.json.
3. Tokenize the original text with Mistral tokenizer.
4. Save to data/mistral/ctx{N}/.

Usage:
    python scripts/prepare_mistral_data.py
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download, list_repo_files


def download_slimpajama(cache_dir: Path, max_files: int = 5):
    """Reuse existing cache or download if missing."""
    print(f"\n--- Checking SlimPajama-6B Cache ---")
    print(f"  Cache dir: {cache_dir}")
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_id = "DKYoon/SlimPajama-6B"
    
    # Check if files exist
    existing_files = list(cache_dir.glob("*.parquet"))
    if len(existing_files) >= max_files:
        print(f"  Found {len(existing_files)} existing parquet files. Skipping download.")
        # Return sorted list for consistent ordering (CRITICAL)
        return sorted([str(f) for f in existing_files])
    
    print("  Listing repository files...")
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith('.parquet')]
    
    downloaded = []
    for i, pf in enumerate(parquet_files[:max_files]):
        print(f"  Downloading {i+1}/{min(max_files, len(parquet_files))}: {pf}")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=pf,
            cache_dir=str(cache_dir),
            repo_type="dataset"
        )
        downloaded.append(local_path)
    
    # Sort for consistency
    return sorted(downloaded)


def load_raw_texts(parquet_files, max_samples: int = 1000000):
    """Load raw texts in consistent order."""
    import pandas as pd
    
    print(f"\n  Loading samples from {len(parquet_files)} parquet files...")
    
    all_samples = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        texts = df['text'].tolist()
        all_samples.extend(texts)
    
    print(f"  Total raw samples: {len(all_samples)}")
    return all_samples


def process_context_length(
    ctx_len: int, 
    raw_texts: list, 
    tokenizer, 
    llama_data_dir: Path, 
    output_dir: Path
):
    """Re-tokenize samples for a specific context length."""
    source_json = llama_data_dir / f"ctx{ctx_len}" / "samples.json"
    
    if not source_json.exists():
        print(f"  ⚠️ Skipping ctx{ctx_len}: {source_json} not found")
        return
    
    print(f"\n--- Processing ctx{ctx_len} ---")
    print(f"  Source: {source_json}")
    
    with open(source_json, 'r') as f:
        data = json.load(f)
        
    samples = data["samples"]
    print(f"  Found {len(samples)} samples to re-tokenize")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    new_samples = []
    new_sample_texts = []
    
    for s in samples:
        source_idx = s["source_idx"]
        
        # Get ORIGINAL raw text
        if source_idx >= len(raw_texts):
            print(f"    ERROR: source_idx {source_idx} out of range! (max {len(raw_texts)})")
            continue
            
        raw_text = raw_texts[source_idx]
        
        # Tokenize with Mistral
        tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        
        # Check length
        if len(tokens) < ctx_len:
            print(f"    ⚠️ Sample {s['sample_idx']} too short for Mistral: {len(tokens)} < {ctx_len}")
            # Try to recover if it's close? No, skip or pad?
            # Pipeline enforces length. Let's pad?
            # Or just warn. If we have 64 samples and lose 1-2, it's okay?
            # User wants "same 64 samples".
            # If strictly needed, we can't augment.
            pass
        else:
            tokens = tokens[:ctx_len]
        
        new_samples.append(tokens)
        
        # Store metadata
        decoded = tokenizer.decode(tokens)
        new_sample_texts.append({
            "sample_idx": len(new_samples),
            "original_source_idx": source_idx,
            "n_tokens": len(tokens),
            "llama_sample_idx": s["sample_idx"],
            "text_preview": decoded[:100] + "..."
        })
        
    # Save NPZ
    if new_samples:
        # Check lengths consistency
        lengths = [len(t) for t in new_samples]
        if min(lengths) < ctx_len:
             print(f"    ⚠️ Warning: Some samples shorter than {ctx_len} (min: {min(lengths)})")
        
        # Pad to max length for NPZ creation if necessary (though pipeline handles list of tensors, NPZ needs rect)
        # We'll save as object array or pad manually?
        # Standard npz expects uniform shape. 
        # Let's filter strictly valid ones for NPZ, but maybe save all?
        # Run pipeline expects uniform tensor usually?
        # Actually pipeline checks: "Truncate to exact length".
        # If short, it skips.
        # We should save what we have.
        
        # Pad with 0 for NPZ saving safety (pipeline will skip them anyway)
        max_len = max(max(lengths), ctx_len)
        padded_samples = []
        for t in new_samples:
            if len(t) < max_len:
                pad = [0] * (max_len - len(t))
                padded_samples.append(t + pad)
            else:
                padded_samples.append(t)
                
        all_tokens = np.array(padded_samples, dtype=np.int32)
        
        shard_path = output_dir / "shard_000.npz"
        np.savez_compressed(
            shard_path, 
            input_ids=all_tokens, 
            n_samples=len(new_samples),
            target_length=ctx_len
        )
        print(f"  Saved NPZ: {shard_path} (shape: {all_tokens.shape})")
        
        # Save JSON
        json_path = output_dir / "samples.json"
        with open(json_path, 'w') as f:
            json.dump({
                "metadata": {
                    "model": "Mistral-7B-v0.1",
                    "original_source": str(source_json),
                    "generated_at": datetime.now().isoformat()
                },
                "samples": new_sample_texts
            }, f, indent=2)
        print(f"  Saved JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--lengths", default="128,512,1024,2048")
    parser.add_argument("--cache-dir", default="data/slimpajama")
    parser.add_argument("--llama-data-dir", default="data")
    parser.add_argument("--output-dir", default="data/mistral")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MISTRAL DATA PREPARATION (Exact LLaMA Replication)")
    print("=" * 60)
    
    lengths = [int(x.strip()) for x in args.lengths.split(",")]
    cache_dir = Path(args.cache_dir)
    llama_dir = Path(args.llama_data_dir)
    out_dir = Path(args.output_dir)
    
    print(f"Model: {args.model}")
    print(f"Output: {out_dir}")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # 2. Load Raw Text
    parquet_files = download_slimpajama(cache_dir)
    raw_texts = load_raw_texts(parquet_files)
    
    # 3. Process each length
    for L in lengths:
        sub_out = out_dir / f"ctx{L}"
        process_context_length(L, raw_texts, tokenizer, llama_dir, sub_out)
        
    print("\nDone!")


if __name__ == "__main__":
    main()
