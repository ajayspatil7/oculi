"""
EXP0 — Observational Entropy-Q Correlation (Multi-Sample)
==========================================================

Computes correlation between Query Norm and Attention Entropy.
NOW USES ALL SAMPLES and aggregates (q, entropy) pairs across the entire dataset.

Correctness Fixes:
1. Multi-sample aggregation: Uses 64 samples instead of 1.
2. Batched processing: Efficiently processes chunks of samples.
3. Correct indexing: Iterates over batch dimension properly.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ..registry import experiment
from ..hooks import UnifiedHooks
from ..metrics import compute_attention_entropy, compute_query_norms


@experiment("exp0")
def run_exp0(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run EXP0: Observational correlation with multi-sample aggregation.
    """
    adapter = context["adapter"]
    # Get all samples (list of [1, seq_len] tensors)
    all_samples = context.get("all_samples", [context["input_ids"]])
    
    # Stack samples into batches
    input_tensor = torch.cat(all_samples, dim=0)  # [n_samples, seq_len]
    n_samples = input_tensor.shape[0]
    
    output_dir = Path(context["output_dir"]) / "exp0"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("EXP0 — Observational Correlation (Multi-Sample)")
    print("=" * 60)
    print(f"  Samples: {n_samples}")
    print(f"  Context Length: {input_tensor.shape[1]}")
    
    # Storage: (layer, head) -> lists of q, e
    # Using arrays for memory efficiency if possible, but list extend is fast
    # Storing raw values to compute correlation at the end
    aggregated_data = defaultdict(lambda: {'q': [], 'e': []})
    
    # Process in batches to manage VRAM
    BATCH_SIZE = 4  # Conservative batch size
    
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"  Processing {n_samples} samples in {n_batches} batches...")
    
    for b_idx in tqdm(range(n_batches), desc="Batches"):
        start_idx = b_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, n_samples)
        batch_input = input_tensor[start_idx:end_idx]
        
        # Forward pass with hooks
        with UnifiedHooks(adapter) as hooks:
            adapter.forward(batch_input)
            
            # Process layers
            for layer in range(adapter.info.n_layers):
                # Q: [batch, heads, seq, dim]
                Q, K, V = hooks.compute_qkv(layer)
                
                # Attn: [batch, heads, seq, seq]
                attn_probs = hooks.compute_attention_probs(Q, K)
                
                # Metrics
                entropy = compute_attention_entropy(attn_probs, ignore_first_n=2)  # [batch, heads, seq]
                q_norms = compute_query_norms(Q)  # [batch, heads, seq]
                
                # Move to CPU numpy for aggregation
                entropy_np = entropy.cpu().float().numpy()
                q_norms_np = q_norms.cpu().float().numpy()
                
                # Collect valid pairs for each head
                for head in range(adapter.info.n_heads):
                    # Flatten batch and seq dimensions for this head
                    e_flat = entropy_np[:, head, :].flatten()
                    q_flat = q_norms_np[:, head, :].flatten()
                    
                    # Filter NaNs
                    valid_mask = ~np.isnan(e_flat) & ~np.isnan(q_flat)
                    
                    if valid_mask.sum() > 0:
                        aggregated_data[(layer, head)]['e'].extend(e_flat[valid_mask])
                        aggregated_data[(layer, head)]['q'].extend(q_flat[valid_mask])
    
    # Compute correlations
    print("\n  Computing correlations...")
    results = []
    
    for (layer, head), data in tqdm(aggregated_data.items(), desc="Analysis"):
        e = np.array(data['e'])
        q = np.array(data['q'])
        
        # Minimum points check
        if len(e) < 30:
            continue
            
        # Variance check
        if np.std(e) < 1e-8 or np.std(q) < 1e-8:
            continue
            
        try:
            r_pearson, p_pearson = stats.pearsonr(q, e)
            r_spearman, p_spearman = stats.spearmanr(q, e)
            
            # Check for NaN results
            if np.isnan(r_pearson) or np.isnan(r_spearman):
                continue
                
            results.append({
                'layer': layer,
                'head': head,
                'r_pearson': r_pearson,
                'p_pearson': p_pearson,
                'r_spearman': r_spearman,
                'p_spearman': p_spearman,
                'mean_entropy': np.mean(e),
                'mean_q_norm': np.mean(q),
                'n_points': len(e)
            })
        except Exception:
            continue
            
    # Save results
    df = pd.DataFrame(results)
    if len(df) > 0:
        csv_path = output_dir / "correlation_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        
        # Plots
        try:
            _plot_correlation_heatmap(df, output_dir / "correlation_heatmap.png", adapter.info)
            _plot_layer_summary(df, output_dir / "layer_summary.png")
            _plot_scatter_examples(df, aggregated_data, output_dir)
        except Exception as e:
            print(f"  Warning: Plotting failed: {e}")
    else:
        print("  Warning: No valid correlations found.")
        
    return {
        "exp": "exp0",
        "n_samples": n_samples,
        "n_results": len(df)
    }


def _plot_correlation_heatmap(df: pd.DataFrame, output_path: Path, model_info):
    """Plot heatmap of Pearson correlation per head."""
    # Create grid
    grid = np.full((model_info.n_layers, model_info.n_heads), np.nan)
    
    for _, row in df.iterrows():
        l, h = int(row['layer']), int(row['head'])
        grid[l, h] = row['r_pearson']
        
    plt.figure(figsize=(12, 8))
    sns.heatmap(grid, cmap="RdBu_r", center=0, vmin=-1, vmax=1, 
                cbar_kws={'label': 'Pearson Correlation'})
    plt.title("EXP0: Query Norm vs Entropy Correlation")
    plt.xlabel("Head Index")
    plt.ylabel("Layer Index")
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_layer_summary(df: pd.DataFrame, output_path: Path):
    """Plot average correlation per layer."""
    layer_avg = df.groupby('layer')['r_pearson'].mean()
    layer_std = df.groupby('layer')['r_pearson'].std()
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(layer_avg.index, layer_avg.values, yerr=layer_std.values, 
                 fmt='o-', capsize=5)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Average Correlation by Layer")
    plt.xlabel("Layer")
    plt.ylabel("Mean Pearson Correlation")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_scatter_examples(df: pd.DataFrame, aggregated_data: Dict, output_dir: Path):
    """Plot scatter plots for the most positive and negative correlated heads."""
    # Find extremes
    if len(df) == 0:
        return
        
    min_row = df.loc[df['r_pearson'].idxmin()]
    max_row = df.loc[df['r_pearson'].idxmax()]
    
    extremes = [
        (min_row, "Most Negative"),
        (max_row, "Most Positive")
    ]
    
    rows, cols = 1, 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5))
    
    for idx, (row, label) in enumerate(extremes):
        l, h = int(row['layer']), int(row['head'])
        data = aggregated_data[(l, h)]
        
        # Subsample for plotting if too many points
        e = np.array(data['e'])
        q = np.array(data['q'])
        if len(e) > 5000:
            indices = np.random.choice(len(e), 5000, replace=False)
            e = e[indices]
            q = q[indices]
            
        ax = axes[idx]
        ax.scatter(q, e, alpha=0.1, s=1)
        ax.set_title(f"{label}: L{l}H{h} (r={row['r_pearson']:.2f})")
        ax.set_xlabel("Query Norm")
        ax.set_ylabel("Entropy")
        
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_examples.png", dpi=150)
    plt.close()
