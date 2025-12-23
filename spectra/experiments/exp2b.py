"""
EXP2b — Global Gain Analysis
=============================

Computes gain (sensitivity to Q scaling) for ALL heads across ALL layers.
This is the "selection oracle" for EXP3b.
"""

from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..registry import experiment
from ..hooks import UnifiedHooks
from ..metrics import compute_attention_entropy, aggregate_metrics, compute_linear_gain


@experiment("exp2b")
def run_exp2b(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run EXP2b: Global gain analysis.
    
    Args:
        context: Pipeline context
        
    Returns:
        Dict with exp2b results including head categorization
    """
    adapter = context["adapter"]
    input_ids = context["input_ids"]
    output_dir = Path(context["output_dir"]) / "exp2b"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = context.get("config", {})
    alphas = config.get("alphas", {}).get("default", [0.5, 0.75, 1.0, 1.25, 1.5])
    
    print("\n" + "=" * 60)
    print("EXP2b — Global Gain Analysis")
    print("=" * 60)
    print(f"  Layers: {adapter.info.n_layers}")
    print(f"  Heads: {adapter.info.n_heads}")
    print(f"  Total: {adapter.info.total_heads}")
    print(f"  Alphas: {alphas}")
    
    results = []
    
    with UnifiedHooks(adapter) as hooks:
        # Initial forward pass
        adapter.forward(input_ids)
        
        total_heads = adapter.info.n_layers * adapter.info.n_heads
        pbar = tqdm(total=total_heads, desc="Computing gain")
        
        for layer in range(adapter.info.n_layers):
            hidden_states = hooks.get_hidden_states(layer)
            
            for head in range(adapter.info.n_heads):
                # Collect entropy across alphas
                entropies = []
                
                for alpha in alphas:
                    Q, K, V = hooks.compute_qkv_scaled(
                        layer, head,
                        q_scale=alpha, k_scale=1.0,
                        hidden_states=hidden_states
                    )
                    
                    attn_probs = hooks.compute_attention_probs(Q, K)
                    target_attn = attn_probs[:, head:head+1, :, :]
                    entropy = compute_attention_entropy(target_attn, ignore_first_n=2)
                    mean_ent, _ = aggregate_metrics(entropy)
                    entropies.append(mean_ent)
                
                # Compute gain
                entropies = np.array(entropies)
                alphas_arr = np.array(alphas)
                gain, r_squared = compute_linear_gain(alphas_arr, entropies)
                
                results.append({
                    'layer': layer,
                    'head': head,
                    'gain_entropy': gain,
                    'r_squared': r_squared,
                    'baseline_entropy': entropies[alphas.index(1.0)] if 1.0 in alphas else entropies[len(alphas)//2]
                })
                
                pbar.update(1)
            
            # Clear cache periodically
            if layer % 4 == 0:
                torch.cuda.empty_cache()
        
        pbar.close()
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Categorize heads
    df['abs_gain'] = df['gain_entropy'].abs()
    p90 = df['abs_gain'].quantile(0.90)
    p10 = df['abs_gain'].quantile(0.10)
    
    df['gain_category'] = df['abs_gain'].apply(
        lambda x: 'high' if x >= p90 else ('low' if x <= p10 else 'medium')
    )
    
    # Save results
    csv_path = output_dir / "gain_summary_global.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    
    # Generate plots
    _plot_gain_heatmap(df, output_dir / "gain_heatmap.png", adapter.info)
    _plot_gain_distribution(df, output_dir / "gain_distribution.png")
    
    # Identify key heads for EXP3b
    high_gain_head = df.loc[df['abs_gain'].idxmax()]
    low_gain_head = df.loc[df['abs_gain'].idxmin()]
    medium_gain_head = df.iloc[(df['abs_gain'] - df['abs_gain'].median()).abs().idxmin()]
    
    selected_heads = {
        'high_gain': {'layer': int(high_gain_head['layer']), 'head': int(high_gain_head['head']), 'gain': float(high_gain_head['gain_entropy'])},
        'medium_gain': {'layer': int(medium_gain_head['layer']), 'head': int(medium_gain_head['head']), 'gain': float(medium_gain_head['gain_entropy'])},
        'low_gain': {'layer': int(low_gain_head['layer']), 'head': int(low_gain_head['head']), 'gain': float(low_gain_head['gain_entropy'])}
    }
    
    print("\n  Selected heads for EXP3b:")
    for name, info in selected_heads.items():
        print(f"    {name}: L{info['layer']} H{info['head']} (gain={info['gain']:.4f})")
    
    return {
        "exp": "exp2b",
        "success": True,
        "csv_path": str(csv_path),
        "selected_heads": selected_heads,
        "mean_gain": df['gain_entropy'].mean(),
        "std_gain": df['gain_entropy'].std()
    }


def _plot_gain_heatmap(df: pd.DataFrame, output_path: Path, model_info):
    """Plot gain heatmap."""
    heatmap = df.pivot(index='layer', columns='head', values='gain_entropy').values
    
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(heatmap, cmap='RdBu_r', aspect='auto')
    
    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Gain Heatmap (Entropy Sensitivity to Q Scaling)', fontsize=14)
    
    plt.colorbar(im, ax=ax, label='Gain')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_gain_distribution(df: pd.DataFrame, output_path: Path):
    """Plot gain distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['gain_entropy'], bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(df['gain_entropy'].median(), color='green', linestyle=':', linewidth=2)
    ax.set_xlabel('Gain (Entropy)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Entropy Gain Across Heads', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
