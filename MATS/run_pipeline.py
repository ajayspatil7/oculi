#!/usr/bin/env python3
"""
MATS 10.0 Pipeline Runner
==========================

SINGLE ENTRY POINT for all experiments.

Usage:
    # Run full pipeline
    python run_pipeline.py
    
    # Run specific phase
    python run_pipeline.py --phase sanity
    
    # Dry run (no model loading)
    python run_pipeline.py --dry-run
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add MATS to path
sys.path.insert(0, str(Path(__file__).parent))

from mats.utils import set_seed, load_config, create_output_dir, save_results, print_separator
from mats.registry import run_experiment


# Experiment execution order
EXPERIMENT_ORDER = [
    "sanity",         # Phase 1: Critical gate (Hour 1)
    "rationalization", # Phase 2: Identify heads (Hours 2-5)
    "restoration",     # Phase 3: Logic sharpening (Hours 6-9)
    "jamming",         # Phase 4: Sycophancy flattening (Hours 10-12)
    "control",         # Phase 5: Baseline check (Hours 13-14)
]


def load_model_if_needed(config: dict, dry_run: bool = False):
    """Load model or return None for dry run."""
    if dry_run:
        print("  [DRY RUN] Skipping model loading")
        return None
    
    from mats.model import load_model
    
    model_config = config.get("model", {})
    model_name = model_config.get("name", "Qwen/Qwen2.5-7B-Instruct")
    device = model_config.get("device", "cuda")
    
    return load_model(model_name, device=device)


def run_pipeline(
    config_path: str = "config.yaml",
    phases: Optional[List[str]] = None,
    dry_run: bool = False,
):
    """
    Run the full MATS experiment pipeline.
    
    Args:
        config_path: Path to configuration file
        phases: List of phases to run (None = all)
        dry_run: If True, skip model loading (for testing)
    """
    print_separator("MATS 10.0: Sycophancy Entropy Control")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Config: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Set seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Create output directory
    output_dir = create_output_dir(config.get("output", {}).get("base_dir", "results"))
    
    # Load model
    print_separator("Model Loading")
    model = load_model_if_needed(config, dry_run)
    
    # Determine which experiments to run
    if phases:
        experiments_to_run = [p for p in phases if p in EXPERIMENT_ORDER]
    else:
        experiments_to_run = EXPERIMENT_ORDER
    
    print(f"\n  Experiments: {experiments_to_run}")
    
    # Create shared context
    context = {
        "model": model,
        "config": config,
        "output_dir": output_dir,
    }
    
    # Run experiments
    all_results = {}
    
    for exp_name in experiments_to_run:
        print_separator(f"Running: {exp_name.upper()}")
        
        if dry_run:
            print(f"  [DRY RUN] Would run experiment: {exp_name}")
            all_results[exp_name] = {"dry_run": True}
            continue
        
        try:
            # Import and register experiments
            import experiments  # noqa: F401
            
            result = run_experiment(exp_name, context)
            all_results[exp_name] = result
            
            # Check for early exit conditions
            if exp_name == "sanity" and not result.get("passed", False):
                print("\n  ⚠️ SANITY CHECK FAILED")
                print("  Pipeline halted. See pivot recommendations in results.")
                break
                
        except Exception as e:
            print(f"  ❌ Error in {exp_name}: {e}")
            all_results[exp_name] = {"error": str(e)}
            # Continue with other experiments
    
    # Save aggregate results
    print_separator("Pipeline Complete")
    save_results(all_results, output_dir, "pipeline_results.json", config)
    
    print(f"\n  Results saved to: {output_dir}")
    print(f"  Completed: {datetime.now().isoformat()}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="MATS 10.0 Sycophancy Entropy Control Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--phase",
        type=str,
        nargs="+",
        choices=EXPERIMENT_ORDER,
        help="Specific phase(s) to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model loading (for testing)",
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        config_path=args.config,
        phases=args.phase,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
