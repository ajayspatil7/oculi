"""
EXP3b — Gain-Conditioned Q vs K Scaling
========================================

Final experiment: Tests Q-K asymmetry across high/medium/low gain heads.
This is a CONFIRMATORY experiment, not exploratory.
"""

from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..registry import experiment
from ..hooks import UnifiedHooks
from ..metrics import compute_attention_entropy, aggregate_metrics


@experiment("exp3b")
def run_exp3b(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run EXP3b: Gain-conditioned Q vs K comparison.
    
    Uses heads selected from EXP2b results.
    
    Args:
        context: Pipeline context (must contain exp2b results)
        
    Returns:
        Dict with exp3b results
    """
    adapter = context["adapter"]
    input_ids = context["input_ids"]
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
    print("\n  Target heads:")
    for name, info in selected_heads.items():
        print(f"    {name}: Layer {info['layer']}, Head {info['head']}")
    print(f"\n  Alphas: {alphas}")
    
    all_results = {}
    all_metrics = {}
    
    with UnifiedHooks(adapter) as hooks:
        adapter.forward(input_ids)
        
        for head_name, head_info in selected_heads.items():
            layer = head_info['layer']
            head = head_info['head']
            hidden_states = hooks.get_hidden_states(layer)
            
            print(f"\n  --- {head_name}: L{layer} H{head} ---")
            results = []
            
            # Q-scaling
            print("    Q-scaling:", end=" ")
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
                
                results.append({
                    'alpha': alpha,
                    'scaling_type': 'Q',
                    'entropy': mean_ent,
                    'layer': layer,
                    'head': head,
                    'head_name': head_name
                })
                print(f"{alpha:.2f}→{mean_ent:.3f}", end=" ")
            print()
            
            # K-scaling
            print("    K-scaling:", end=" ")
            for alpha in alphas:
                Q, K, V = hooks.compute_qkv_scaled(
                    layer, head,
                    q_scale=1.0, k_scale=alpha,
                    hidden_states=hidden_states
                )
                attn_probs = hooks.compute_attention_probs(Q, K)
                target_attn = attn_probs[:, head:head+1, :, :]
                entropy = compute_attention_entropy(target_attn, ignore_first_n=2)
                mean_ent, _ = aggregate_metrics(entropy)
                
                results.append({
                    'alpha': alpha,
                    'scaling_type': 'K',
                    'entropy': mean_ent,
                    'layer': layer,
                    'head': head,
                    'head_name': head_name
                })
                print(f"{alpha:.2f}→{mean_ent:.3f}", end=" ")
            print()
            
            df = pd.DataFrame(results)
            all_results[head_name] = df
            
            # Compute asymmetry
            q_data = df[df['scaling_type'] == 'Q']['entropy']
            k_data = df[df['scaling_type'] == 'K']['entropy']
            q_range = q_data.max() - q_data.min()
            k_range = k_data.max() - k_data.min()
            asymmetry = q_range / k_range if k_range > 0 else float('inf')
            
            all_metrics[head_name] = {
                'q_range': q_range,
                'k_range': k_range,
                'asymmetry_ratio': asymmetry,
                'is_q_dominant': asymmetry > 1.2
            }
            
            # Save per-head results
            head_dir = output_dir / f"L{layer}_H{head}"
            head_dir.mkdir(exist_ok=True)
            df.to_csv(head_dir / "qk_results.csv", index=False)
            _plot_delta_entropy(df, head_dir / "qk_delta_entropy.png", head_name)
    
    # Combined plot
    _plot_combined(all_results, output_dir / "exp3b_combined.png")
    
    # Save combined results
    combined_df = pd.concat(all_results.values(), ignore_index=True)
    combined_df.to_csv(output_dir / "exp3b_all_results.csv", index=False)
    
    # Summary
    summary_data = []
    for name, metrics in all_metrics.items():
        info = selected_heads[name]
        summary_data.append({
            'head_name': name,
            'layer': info['layer'],
            'head': info['head'],
            **metrics
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "exp3b_summary.csv", index=False)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, metrics in all_metrics.items():
        print(f"\n  {name}:")
        print(f"    Q range: {metrics['q_range']:.4f}")
        print(f"    K range: {metrics['k_range']:.4f}")
        print(f"    Asymmetry: {metrics['asymmetry_ratio']:.2f}")
        print(f"    → {'Q-dominant' if metrics['is_q_dominant'] else 'Symmetric'}")
    
    return {
        "exp": "exp3b",
        "success": True,
        "metrics": all_metrics,
        "csv_path": str(output_dir / "exp3b_all_results.csv")
    }


def _plot_delta_entropy(df: pd.DataFrame, output_path: Path, head_name: str):
    """Plot delta entropy."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    q_data = df[df['scaling_type'] == 'Q'].sort_values('alpha')
    k_data = df[df['scaling_type'] == 'K'].sort_values('alpha')
    
    alphas = q_data['alpha'].values
    q_baseline = q_data[q_data['alpha'] == 1.0]['entropy'].values[0]
    k_baseline = k_data[k_data['alpha'] == 1.0]['entropy'].values[0]
    
    q_delta = q_data['entropy'].values - q_baseline
    k_delta = k_data['entropy'].values - k_baseline
    
    ax.plot(alphas, q_delta, 'o-', color='#3498db', linewidth=2.5, markersize=10, label='Q-scaling')
    ax.plot(alphas, k_delta, 's--', color='#e74c3c', linewidth=2.5, markersize=10, label='K-scaling')
    
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    
    layer = df['layer'].iloc[0]
    head = df['head'].iloc[0]
    ax.set_xlabel('α', fontsize=12)
    ax.set_ylabel('ΔH', fontsize=12)
    ax.set_title(f'{head_name}: L{layer} H{head}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_combined(all_results: Dict, output_path: Path):
    """Plot all heads combined."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (name, df) in enumerate(all_results.items()):
        ax = axes[idx]
        
        q_data = df[df['scaling_type'] == 'Q'].sort_values('alpha')
        k_data = df[df['scaling_type'] == 'K'].sort_values('alpha')
        
        alphas = q_data['alpha'].values
        q_baseline = q_data[q_data['alpha'] == 1.0]['entropy'].values[0]
        k_baseline = k_data[k_data['alpha'] == 1.0]['entropy'].values[0]
        
        q_delta = q_data['entropy'].values - q_baseline
        k_delta = k_data['entropy'].values - k_baseline
        
        ax.plot(alphas, q_delta, 'o-', color='#3498db', linewidth=2, label='Q')
        ax.plot(alphas, k_delta, 's--', color='#e74c3c', linewidth=2, label='K')
        
        ax.axvline(1.0, color='gray', linestyle=':', alpha=0.7)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('α', fontsize=11)
        ax.set_ylabel('ΔH', fontsize=11)
        ax.set_title(name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.suptitle('EXP3b: Gain-Conditioned Q vs K Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
