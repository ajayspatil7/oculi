"""
EXP3b — Gain-Conditioned Q vs K Scaling (Multi-Sample)
=======================================================

Final experiment: Tests Q-K asymmetry across high/medium/low gain heads.
NOW USES ALL SAMPLES for proper variance estimation.
"""

from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..registry import experiment
from ..hooks import UnifiedHooks
from ..metrics import compute_attention_entropy, aggregate_metrics, aggregate_across_samples


@experiment("exp3b")
def run_exp3b(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run EXP3b: Gain-conditioned Q vs K comparison with multi-sample aggregation.
    
    Args:
        context: Pipeline context (must contain exp2b results and all_samples)
        
    Returns:
        Dict with exp3b results
    """
    adapter = context["adapter"]
    all_samples = context.get("all_samples", [context["input_ids"]])
    output_dir = Path(context["output_dir"]) / "exp3b"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = context.get("config", {})
    alphas = config.get("alphas", {}).get("default", [0.5, 0.75, 1.0, 1.25, 1.5])
    
    # Get selected heads from EXP2b
    exp2b_results = context.get("exp2b_results", {})
    selected_heads = exp2b_results.get("selected_heads", {})
    
    if not selected_heads:
        print("  ⚠️ No heads selected from EXP2b, using defaults")
        selected_heads = {
            'high_gain': {'layer': 20, 'head': 19},
            'medium_gain': {'layer': 14, 'head': 8},
            'low_gain': {'layer': 0, 'head': 23}
        }
    
    print("\n" + "=" * 60)
    print("EXP3b — Gain-Conditioned Q vs K Scaling")
    print("=" * 60)
    print("  FINAL EXPERIMENT")
    print(f"  Samples: {len(all_samples)}")
    print("\n  Target heads:")
    for name, info in selected_heads.items():
        print(f"    {name}: Layer {info['layer']}, Head {info['head']}")
    print(f"\n  Alphas: {alphas}")
    
    results = []
    
    # Process each selected head
    for head_name, head_info in selected_heads.items():
        layer = int(head_info['layer'])
        head = int(head_info['head'])
        
        print(f"\n  --- {head_name}: L{layer} H{head} ---")
        
        # Test both Q and K scaling
        for scaling_type in ['Q', 'K']:
            print(f"    {scaling_type}-scaling:", end=" ")
            
            for alpha in alphas:
                sample_entropies = []
                
                # Iterate all samples (batched loop could be optimized, but explicit loop is safer for logic)
                # Actually, given small number of heads, we can afford explicit loops.
                for sample_idx, input_ids in enumerate(all_samples):
                    with UnifiedHooks(adapter) as hooks:
                        adapter.forward(input_ids)
                        hidden_states = hooks.get_hidden_states(layer)
                        
                        # Apply scaling
                        q_scale = alpha if scaling_type == 'Q' else 1.0
                        k_scale = alpha if scaling_type == 'K' else 1.0
                        
                        Q, K, V = hooks.compute_qkv_scaled(
                            layer, head,
                            q_scale=q_scale, k_scale=k_scale,
                            hidden_states=hidden_states
                        )
                        
                        attn_probs = hooks.compute_attention_probs(Q, K)
                        target_attn = attn_probs[:, head:head+1, :, :]
                        
                        entropy = compute_attention_entropy(target_attn, ignore_first_n=2)
                        mean_ent, _ = aggregate_metrics(entropy)
                        sample_entropies.append(mean_ent)
                
                # Aggregate across samples
                mean_entropy, std_entropy, n_valid = aggregate_across_samples(sample_entropies)
                
                results.append({
                    'head_name': head_name,
                    'layer': layer,
                    'head': head,
                    'scaling_type': scaling_type,
                    'alpha': alpha,
                    'mean_entropy': mean_entropy,
                    'std_entropy': std_entropy,
                    'n_samples': n_valid
                })
                
                print(f"{alpha:.2f}→{mean_entropy:.3f} (±{std_entropy:.3f})", end="  ")
            print()
            
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    csv_path = output_dir / "qk_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    
    # Plot
    _plot_qk_comparison(df, output_dir / "qk_comparison.png")
    
    return {
        "exp": "exp3b",
        "success": True,
        "csv_path": str(csv_path)
    }


def _plot_qk_comparison(df: pd.DataFrame, output_path: Path):
    """Plot Q vs K scaling comparison for each head type."""
    head_names = df['head_name'].unique()
    fig, axes = plt.subplots(1, len(head_names), figsize=(15, 5))
    if len(head_names) == 1:
        axes = [axes]
    
    for ax, name in zip(axes, head_names):
        subset = df[df['head_name'] == name]
        
        # Q-scaling
        q_data = subset[subset['scaling_type'] == 'Q']
        ax.errorbar(q_data['alpha'], q_data['mean_entropy'], yerr=q_data['std_entropy'], 
                    label='Q-scaling', fmt='o-', color='#3498db', capsize=5)
        
        # K-scaling
        k_data = subset[subset['scaling_type'] == 'K']
        ax.errorbar(k_data['alpha'], k_data['mean_entropy'], yerr=k_data['std_entropy'], 
                    label='K-scaling', fmt='x--', color='#e74c3c', capsize=5)
        
        ax.set_title(f"{name} (L{subset['layer'].iloc[0]} H{subset['head'].iloc[0]})")
        ax.set_xlabel("Scale Factor (α)")
        ax.set_ylabel("Entropy")
        ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
