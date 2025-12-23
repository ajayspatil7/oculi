"""
EXP0 — Observational Correlation Analysis
==========================================

Measures correlation between query norm and attention entropy.
No interventions, pure observation.
"""

from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from ..registry import experiment
from ..hooks import UnifiedHooks
from ..metrics import compute_attention_entropy, compute_query_norms, aggregate_metrics


@experiment("exp0")
def run_exp0(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run EXP0: Observational correlation analysis.
    
    Args:
        context: Pipeline context containing:
            - adapter: ModelAdapter
            - input_ids: torch.Tensor
            - output_dir: Path
            - config: Dict
            
    Returns:
        Dict with exp0 results
    """
    adapter = context["adapter"]
    input_ids = context["input_ids"]
    output_dir = Path(context["output_dir"]) / "exp0"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("EXP0 — Observational Correlation Analysis")
    print("=" * 60)
    
    results = []
    
    with UnifiedHooks(adapter) as hooks:
        # Forward pass to capture hidden states
        adapter.forward(input_ids)
        
        # Analyze each layer and head
        for layer in tqdm(range(adapter.info.n_layers), desc="Layers"):
            Q, K, V = hooks.compute_qkv(layer)
            attn_probs = hooks.compute_attention_probs(Q, K)
            
            # Compute metrics
            entropy = compute_attention_entropy(attn_probs, ignore_first_n=2)
            q_norms = compute_query_norms(Q)
            
            # Per-head correlation
            for head in range(adapter.info.n_heads):
                head_entropy = entropy[0, head, :].cpu().numpy()
                head_q_norm = q_norms[0, head, :].cpu().numpy()
                
                # Remove NaN
                valid = ~np.isnan(head_entropy)
                if valid.sum() < 10:
                    continue
                
                e = head_entropy[valid]
                q = head_q_norm[valid]
                
                # Correlation
                r_pearson, p_pearson = stats.pearsonr(q, e)
                r_spearman, p_spearman = stats.spearmanr(q, e)
                
                results.append({
                    'layer': layer,
                    'head': head,
                    'r_pearson': r_pearson,
                    'p_pearson': p_pearson,
                    'r_spearman': r_spearman,
                    'p_spearman': p_spearman,
                    'mean_entropy': np.mean(e),
                    'mean_q_norm': np.mean(q),
                    'n_tokens': len(e)
                })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    csv_path = output_dir / "correlation_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    # Generate plots
    _plot_correlation_heatmap(df, output_dir / "correlation_heatmap.png", adapter.info)
    _plot_layer_summary(df, output_dir / "layer_summary.png")
    
    # Summary stats
    print(f"\n  Mean Pearson r: {df['r_pearson'].mean():.4f}")
    print(f"  Significant heads (p<0.05): {(df['p_pearson'] < 0.05).sum()}/{len(df)}")
    
    return {
        "exp": "exp0",
        "success": True,
        "csv_path": str(csv_path),
        "mean_r": df['r_pearson'].mean(),
        "significant_count": int((df['p_pearson'] < 0.05).sum())
    }


def _plot_correlation_heatmap(df: pd.DataFrame, output_path: Path, model_info):
    """Plot correlation heatmap."""
    heatmap = df.pivot(index='layer', columns='head', values='r_pearson').values
    
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(heatmap, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Q-Norm vs Entropy Correlation (Pearson r)', fontsize=14)
    
    plt.colorbar(im, ax=ax, label='r')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_layer_summary(df: pd.DataFrame, output_path: Path):
    """Plot mean correlation per layer."""
    layer_means = df.groupby('layer')['r_pearson'].mean()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(layer_means.index, layer_means.values, color='#3498db', edgecolor='white')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Pearson r', fontsize=12)
    ax.set_title('Mean Q-Norm vs Entropy Correlation by Layer', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
