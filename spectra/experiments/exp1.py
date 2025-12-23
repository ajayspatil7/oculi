"""
EXP1 — Causal Q-Scaling Intervention (Multi-Sample)
=====================================================

Scales Q vectors and measures causal effect on entropy.
NOW USES ALL SAMPLES for proper variance estimation.
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
from ..metrics import (
    compute_attention_entropy, 
    aggregate_metrics, 
    aggregate_across_samples,
    check_monotonicity
)


@experiment("exp1")
def run_exp1(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run EXP1: Causal Q-scaling intervention with multi-sample aggregation.
    
    Args:
        context: Pipeline context (must contain 'all_samples')
        
    Returns:
        Dict with exp1 results
    """
    adapter = context["adapter"]
    all_samples = context.get("all_samples", [context["input_ids"]])
    output_dir = Path(context["output_dir"]) / "exp1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = context.get("config", {})
    exp_config = config.get("experiments", {}).get("exp1", {})
    
    # Get target layer/head (from config or defaults)
    target_layer = exp_config.get("target_layer", adapter.info.n_layers // 2)
    target_head = exp_config.get("target_head", 0)
    alphas = config.get("alphas", {}).get("default", [0.5, 0.75, 1.0, 1.25, 1.5])
    
    print("\n" + "=" * 60)
    print("EXP1 — Causal Q-Scaling Intervention")
    print("=" * 60)
    print(f"  Target: Layer {target_layer}, Head {target_head}")
    print(f"  Alphas: {alphas}")
    print(f"  Samples: {len(all_samples)}")
    
    # For each alpha, aggregate across ALL samples
    results = []
    
    for alpha in tqdm(alphas, desc="Scaling"):
        sample_entropies = []
        
        for sample_idx, input_ids in enumerate(all_samples):
            with UnifiedHooks(adapter) as hooks:
                adapter.forward(input_ids)
                hidden_states = hooks.get_hidden_states(target_layer)
                
                # Compute Q, K with scaling
                Q, K, V = hooks.compute_qkv_scaled(
                    target_layer, target_head, 
                    q_scale=alpha, k_scale=1.0,
                    hidden_states=hidden_states
                )
                
                # Compute attention
                attn_probs = hooks.compute_attention_probs(Q, K)
                
                # Extract target head
                target_attn = attn_probs[:, target_head:target_head+1, :, :]
                
                # Compute entropy (mean over positions for this sample)
                entropy = compute_attention_entropy(target_attn, ignore_first_n=2)
                mean_ent, _ = aggregate_metrics(entropy)
                sample_entropies.append(mean_ent)
        
        # Aggregate ACROSS samples (this gives proper variance!)
        mean_entropy, std_entropy, n_valid = aggregate_across_samples(sample_entropies)
        
        results.append({
            'alpha': alpha,
            'mean_entropy': mean_entropy,
            'std_entropy': std_entropy,
            'n_samples': n_valid,
            'layer': target_layer,
            'head': target_head
        })
        
        print(f"  α={alpha:.2f}: entropy={mean_entropy:.4f} ± {std_entropy:.4f} (n={n_valid})")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Check monotonicity
    entropies = df['mean_entropy'].values
    is_monotonic = check_monotonicity(entropies, "decreasing")
    
    # Save results
    csv_path = output_dir / "intervention_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    
    # Plot
    _plot_intervention(df, output_dir / "entropy_vs_scale.png")
    
    print(f"\n  Monotonicity: {'✅ PASS' if is_monotonic else '❌ FAIL'}")
    
    return {
        "exp": "exp1",
        "success": True,
        "is_monotonic": is_monotonic,
        "csv_path": str(csv_path),
        "target_layer": target_layer,
        "target_head": target_head,
        "n_samples": len(all_samples)
    }


def _plot_intervention(df: pd.DataFrame, output_path: Path):
    """Plot entropy vs scale with error bars from multi-sample variance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    alphas = df['alpha'].values
    entropies = df['mean_entropy'].values
    stds = df['std_entropy'].values
    
    ax.errorbar(alphas, entropies, yerr=stds, fmt='o-', 
                color='#3498db', linewidth=2, markersize=10, capsize=5)
    
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Scale Factor (α)', fontsize=12)
    ax.set_ylabel('Attention Entropy', fontsize=12)
    ax.set_title(f'EXP1: Entropy Response to Q Scaling (n={df["n_samples"].iloc[0]} samples)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
