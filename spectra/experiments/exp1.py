"""
EXP1 — Causal Q-Scaling Intervention
=====================================

Scales Q vectors and measures causal effect on entropy.
Proves causality, not just correlation.
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
from ..metrics import compute_attention_entropy, aggregate_metrics, check_monotonicity


@experiment("exp1")
def run_exp1(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run EXP1: Causal Q-scaling intervention.
    
    Args:
        context: Pipeline context
        
    Returns:
        Dict with exp1 results
    """
    adapter = context["adapter"]
    input_ids = context["input_ids"]
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
    
    results = []
    
    with UnifiedHooks(adapter) as hooks:
        # Initial forward pass
        adapter.forward(input_ids)
        hidden_states = hooks.get_hidden_states(target_layer)
        
        for alpha in tqdm(alphas, desc="Scaling"):
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
            
            # Compute entropy
            entropy = compute_attention_entropy(target_attn, ignore_first_n=2)
            mean_entropy, std_entropy = aggregate_metrics(entropy)
            
            results.append({
                'alpha': alpha,
                'mean_entropy': mean_entropy,
                'std_entropy': std_entropy,
                'layer': target_layer,
                'head': target_head
            })
            
            print(f"  α={alpha:.2f}: entropy={mean_entropy:.4f} ± {std_entropy:.4f}")
    
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
        "target_head": target_head
    }


def _plot_intervention(df: pd.DataFrame, output_path: Path):
    """Plot entropy vs scale."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    alphas = df['alpha'].values
    entropies = df['mean_entropy'].values
    stds = df['std_entropy'].values
    
    ax.errorbar(alphas, entropies, yerr=stds, fmt='o-', 
                color='#3498db', linewidth=2, markersize=10, capsize=5)
    
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Scale Factor (α)', fontsize=12)
    ax.set_ylabel('Attention Entropy', fontsize=12)
    ax.set_title('EXP1: Entropy Response to Q Scaling', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
