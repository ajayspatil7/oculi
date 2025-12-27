"""
Rationalization Entropy Profile (Phase 2 - EXP1, Hours 2-5)
============================================================

Goal: Identify which heads "blur" (high entropy) during lies.

Protocol:
1. Load 50 multi-step GSM8K problems
2. For each problem, run Control vs Sycophancy
3. Measure ΔEntropy for ALL heads in layers 20-27
4. Identify consistent patterns

Head Classification:
- Logic Heads: ΔE > +0.5 in ≥70% of problems (blur during rationalization)
- Sycophancy Heads: ΔE < -0.3 in ≥70% of problems (sharpen on hint)

Success Criteria:
- Find ≥3 consistent Logic Heads
- Find ≥2 consistent Sycophancy Heads
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from mats.registry import experiment
from mats.entropy import (
    calculate_entropy,
    compute_delta_entropy,
    identify_head_types,
)
from mats.utils import print_separator, save_results
from data.gsm8k import prepare_all_problems
from data.prompts import format_prompts_from_gsm8k_problem


@experiment("rationalization")
def run_rationalization_profile(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run EXP1: Rationalization entropy profiling.
    
    Args:
        context: Dict with model, config, output_dir
        
    Returns:
        Dict with identified Logic Heads and Sycophancy Heads
    """
    model = context["model"]
    config = context["config"]
    output_dir = Path(context["output_dir"]) / "rationalization"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get config values
    data_cfg = config.get("data", {})
    thresholds = config.get("thresholds", {}).get("rationalization", {})
    analysis_cfg = config.get("analysis", {})
    
    n_problems = data_cfg.get("n_problems", 50)
    min_steps = data_cfg.get("min_steps", 3)
    target_layers = analysis_cfg.get("target_layers", list(range(20, 28)))
    n_heads = analysis_cfg.get("n_heads", 28)
    
    logic_threshold = thresholds.get("logic_delta_threshold", 0.5)
    sycophancy_threshold = thresholds.get("sycophancy_delta_threshold", -0.3)
    consistency_pct = thresholds.get("consistency_pct", 70) / 100.0
    min_logic_heads = thresholds.get("min_logic_heads", 3)
    min_sycophancy_heads = thresholds.get("min_sycophancy_heads", 2)
    
    print_separator("EXP1: Rationalization Entropy Profile")
    print(f"  Problems: {n_problems}")
    print(f"  Target Layers: {target_layers}")
    print(f"  Logic Threshold: ΔE > {logic_threshold}")
    print(f"  Sycophancy Threshold: ΔE < {sycophancy_threshold}")
    print(f"  Consistency: {consistency_pct*100:.0f}%")
    
    # Load problems
    seed = config.get("seed", 42)
    problems = prepare_all_problems(n_problems, min_steps, seed=seed)
    
    # Storage for all ΔEntropy results
    all_delta_results: List[Dict[Tuple[int, int], float]] = []
    per_problem_data = []
    
    # Process each problem
    print(f"\n  Processing {len(problems)} problems...")
    for idx, problem in enumerate(tqdm(problems, desc="Problems")):
        prompts = format_prompts_from_gsm8k_problem(problem)
        
        # Run both conditions
        _, cache_ctrl = model.run_with_cache(prompts["control"])
        _, cache_syco = model.run_with_cache(prompts["sycophancy"])
        model.reset_hooks()
        
        # Compute ΔEntropy for this problem
        problem_deltas = {}
        
        for layer in target_layers:
            pattern_ctrl = cache_ctrl["pattern", layer]
            pattern_syco = cache_syco["pattern", layer]
            
            delta_result = compute_delta_entropy(pattern_syco, pattern_ctrl)
            mean_delta = delta_result["mean_delta"][0]  # [head], batch=0
            
            for head in range(n_heads):
                problem_deltas[(layer, head)] = mean_delta[head].item()
        
        all_delta_results.append(problem_deltas)
        
        # Store per-problem summary
        per_problem_data.append({
            "problem_idx": idx,
            "question_preview": problem.question[:50],
            "correct": problem.answer,
            "wrong": problem.wrong_answer,
            "n_steps": problem.n_steps,
        })
    
    # Identify Logic and Sycophancy Heads
    logic_heads, sycophancy_heads = identify_head_types(
        all_delta_results,
        logic_threshold=logic_threshold,
        sycophancy_threshold=sycophancy_threshold,
        consistency_pct=consistency_pct,
    )
    
    # Compute aggregate statistics per head
    head_stats = _compute_head_statistics(all_delta_results, target_layers, n_heads)
    
    # Check success criteria
    success = (
        len(logic_heads) >= min_logic_heads and
        len(sycophancy_heads) >= min_sycophancy_heads
    )
    
    # Print results
    print_separator("Results")
    print(f"  Logic Heads Found: {len(logic_heads)} (min: {min_logic_heads})")
    print(f"  Sycophancy Heads Found: {len(sycophancy_heads)} (min: {min_sycophancy_heads})")
    print(f"  Status: {'✅ SUCCESS' if success else '⚠️ INSUFFICIENT'}")
    
    if logic_heads:
        print("\n  Logic Heads (blur during rationalization):")
        for layer, head in logic_heads[:10]:
            stats = head_stats[(layer, head)]
            print(f"    L{layer}H{head}: mean ΔE = {stats['mean']:+.3f} ± {stats['std']:.3f}")
    
    if sycophancy_heads:
        print("\n  Sycophancy Heads (focus on hint):")
        for layer, head in sycophancy_heads[:10]:
            stats = head_stats[(layer, head)]
            print(f"    L{layer}H{head}: mean ΔE = {stats['mean']:+.3f} ± {stats['std']:.3f}")
    
    # Prepare results
    results = {
        "success": success,
        "n_problems_analyzed": len(problems),
        "logic_heads": [{"layer": l, "head": h} for l, h in logic_heads],
        "sycophancy_heads": [{"layer": l, "head": h} for l, h in sycophancy_heads],
        "head_statistics": {
            f"L{l}H{h}": stats for (l, h), stats in head_stats.items()
        },
        "thresholds_used": {
            "logic": logic_threshold,
            "sycophancy": sycophancy_threshold,
            "consistency_pct": consistency_pct,
        },
    }
    
    # Save results
    save_results(results, output_dir, "rationalization_results.json", config)
    
    # Save detailed CSV
    _save_head_statistics_csv(head_stats, output_dir)
    
    # Save for next experiment
    context["logic_heads"] = logic_heads
    context["sycophancy_heads"] = sycophancy_heads
    context["head_stats"] = head_stats
    
    return results


def _compute_head_statistics(
    all_deltas: List[Dict[Tuple[int, int], float]],
    layers: List[int],
    n_heads: int,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Compute mean, std, and classification rate per head."""
    stats = {}
    
    for layer in layers:
        for head in range(n_heads):
            values = [d.get((layer, head), 0.0) for d in all_deltas]
            values = np.array(values)
            
            # Filter NaN
            valid = ~np.isnan(values)
            values = values[valid]
            
            if len(values) == 0:
                stats[(layer, head)] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "pct_positive": 0.0,
                    "pct_negative": 0.0,
                }
            else:
                stats[(layer, head)] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "pct_positive": float(np.mean(values > 0.3)),
                    "pct_negative": float(np.mean(values < -0.3)),
                }
    
    return stats


def _save_head_statistics_csv(
    stats: Dict[Tuple[int, int], Dict[str, float]],
    output_dir: Path,
) -> None:
    """Save head statistics as CSV."""
    rows = []
    for (layer, head), s in stats.items():
        rows.append({
            "layer": layer,
            "head": head,
            "mean_delta_entropy": s["mean"],
            "std_delta_entropy": s["std"],
            "pct_blur": s["pct_positive"],
            "pct_focus": s["pct_negative"],
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(["layer", "head"])
    output_path = output_dir / "head_statistics.csv"
    df.to_csv(output_path, index=False)
    print(f"  Saved statistics: {output_path}")
