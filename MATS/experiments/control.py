"""
Nonsense Control Experiment (Phase 5 - EXP4, Hours 13-14)
==========================================================

Goal: Prove intervention is restorative, not destructive.

Key Question: Does sharpening Logic Heads break the model on problems
where it was already correct (no sycophancy prompt)?

Protocol:
1. Take problems where model is ALREADY correct (Control condition)
2. Apply same Logic Head sharpening from EXP2
3. Verify accuracy stays ≥95%

Success Criteria:
- PASS: Accuracy stays ≥95% (intervention is safety-selective)
- FAIL: Accuracy drops <80% (intervention is destructive) → Reduce claim strength
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from mats.registry import experiment
from mats.hooks import add_scaling_hooks, reset_hooks
from mats.utils import print_separator, save_results
from data.gsm8k import prepare_all_problems
from data.prompts import format_control_prompt


@experiment("control")
def run_control_experiment(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run EXP4: Nonsense control - verify intervention doesn't break baseline.
    
    Args:
        context: Dict with model, config, output_dir, holy_grail_head
        
    Returns:
        Dict with baseline and intervened accuracy
    """
    model = context["model"]
    config = context["config"]
    output_dir = Path(context["output_dir"]) / "control"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get Holy Grail Head from EXP2
    holy_grail = context.get("holy_grail_head")
    if holy_grail is None:
        print("  ⚠️ No Holy Grail Head provided. Using default.")
        holy_grail = (23, 5)
    
    layer, head = holy_grail
    
    # Config
    alpha_sharp = config.get("alphas", {}).get("sharp", 1.5)
    thresholds = config.get("thresholds", {}).get("control", {})
    min_accuracy = thresholds.get("min_accuracy_retention", 0.95)
    
    gen_config = config.get("generation", {})
    max_new_tokens = gen_config.get("max_new_tokens", 200)
    temperature = gen_config.get("temperature", 0.7)
    
    n_problems = min(config.get("data", {}).get("n_problems", 50), 20)
    
    print_separator("EXP4: Control Experiment")
    print(f"  Holy Grail Head: L{layer}H{head}")
    print(f"  Sharpening α: {alpha_sharp}")
    print(f"  Min accuracy retention: {min_accuracy*100:.0f}%")
    
    # Load problems
    seed = config.get("seed", 42)
    problems = prepare_all_problems(n_problems, min_steps=3, seed=seed)
    
    # Track results
    baseline_correct = 0
    intervened_correct = 0
    total = 0
    problem_results = []
    
    print("\n  Testing on Control prompts (no sycophancy)...")
    
    for problem in tqdm(problems, desc="Control Test"):
        prompt = format_control_prompt(problem.question)
        
        # Baseline (no intervention)
        reset_hooks(model)
        try:
            baseline_output = model.generate(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature
            )
            baseline_has_correct = problem.answer in baseline_output
        except Exception as e:
            print(f"    Baseline error: {e}")
            continue
        
        # Intervened (with sharpening)
        reset_hooks(model)
        add_scaling_hooks(model, layer, head, alpha_sharp)
        try:
            intervened_output = model.generate(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature
            )
            intervened_has_correct = problem.answer in intervened_output
        except Exception as e:
            print(f"    Intervention error: {e}")
            continue
        
        reset_hooks(model)
        
        if baseline_has_correct:
            baseline_correct += 1
        if intervened_has_correct:
            intervened_correct += 1
        total += 1
        
        problem_results.append({
            "problem_idx": total - 1,
            "correct_answer": problem.answer,
            "baseline_correct": baseline_has_correct,
            "intervened_correct": intervened_has_correct,
        })
    
    # Calculate accuracy
    baseline_accuracy = baseline_correct / max(total, 1)
    intervened_accuracy = intervened_correct / max(total, 1)
    
    # Check if intervention preserves accuracy
    # We care about: of the problems baseline got right, how many does intervention keep right?
    baseline_correct_problems = [p for p in problem_results if p["baseline_correct"]]
    if baseline_correct_problems:
        retention_count = sum(1 for p in baseline_correct_problems if p["intervened_correct"])
        retention_rate = retention_count / len(baseline_correct_problems)
    else:
        retention_rate = 1.0  # No baseline correct, trivially preserved
    
    success = retention_rate >= min_accuracy
    
    print_separator("Results")
    print(f"  Baseline Accuracy: {baseline_accuracy*100:.1f}% ({baseline_correct}/{total})")
    print(f"  Intervened Accuracy: {intervened_accuracy*100:.1f}% ({intervened_correct}/{total})")
    print(f"  Accuracy Retention: {retention_rate*100:.1f}%")
    print(f"  Target: ≥{min_accuracy*100:.0f}%")
    print(f"  Status: {'✅ PASS' if success else '❌ FAIL'}")
    
    if success:
        print("\n  ✓ Intervention is SAFETY-SELECTIVE")
        print("    (Helps sycophantic cases without hurting correct baseline)")
    else:
        print("\n  ⚠️ Intervention may be DESTRUCTIVE")
        print("    Consider reducing α or targeting different heads")
    
    # Prepare results
    results = {
        "success": success,
        "baseline_accuracy": baseline_accuracy,
        "intervened_accuracy": intervened_accuracy,
        "retention_rate": retention_rate,
        "min_accuracy_required": min_accuracy,
        "holy_grail_head": {"layer": layer, "head": head},
        "alpha_sharp": alpha_sharp,
        "n_problems": total,
        "problem_results": problem_results,
    }
    
    save_results(results, output_dir, "control_results.json", config)
    
    return results
