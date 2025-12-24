#!/usr/bin/env python3
"""
Spectra Unified Pipeline Runner
================================

SINGLE ENTRY POINT for all experiments.

Only experiment.yaml changes between runs.
This file NEVER changes.

Usage:
    python run_pipeline.py
    python run_pipeline.py --config config/experiment.yaml
    python run_pipeline.py --experiments exp0,exp1
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import numpy as np
import yaml

# Import spectra modules
from spectra import load_model
from spectra.registry import ExperimentRegistry
from spectra.experiments import exp0, exp1, exp2b, exp3b  # Register experiments


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except:
        return None


def load_data(config: Dict, tokenizer, ctx_len: int = None) -> torch.Tensor:
    """Load single input sample for backward compatibility."""
    all_samples = load_all_samples(config, tokenizer, ctx_len)
    return all_samples[0] if all_samples else None


def load_all_samples(config: Dict, tokenizer, ctx_len: int = None, max_samples: int = 64) -> list:
    """
    Load ALL samples for multi-sample aggregation.
    
    Returns:
        List of input_ids tensors [batch=1, seq_len]
    """
    from src.data_loader import load_from_shards, load_long_context
    
    data_config = config.get("data", {})
    base_dir = data_config.get("directory", "data/processed")
    
    # Try context-specific directory
    ctx_dir_template = data_config.get("context_dir_template", "data/ctx{ctx_len}")
    
    if ctx_len:
        ctx_specific_dir = ctx_dir_template.format(ctx_len=ctx_len)
        samples = load_from_shards(ctx_specific_dir, n_samples=max_samples, device="cuda")
        if samples and len(samples) > 0:
            print(f"  Data source: {ctx_specific_dir} ({len(samples)} samples)")
            return [s["input_ids"] for s in samples]
    
    # Fallback to default directory
    samples = load_from_shards(base_dir, n_samples=max_samples, device="cuda")
    
    if samples and len(samples) > 0:
        print(f"  Data source: {base_dir} ({len(samples)} samples)")
        return [s["input_ids"] for s in samples]
    else:
        # Fallback to generated sample
        print(f"  Data source: generated (1 sample)")
        sample = load_long_context(tokenizer, target_length=ctx_len or 512)
        return [sample["input_ids"]]


def create_metadata(
    config: Dict,
    adapter,
    context_length: int,
    experiment_results: Dict
) -> Dict:
    """Create reproducibility metadata."""
    return {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "seed": config.get("experiment", {}).get("seed", 42),
        "model": adapter.to_dict(),
        "context_length": context_length,
        "alphas": config.get("alphas", {}).get("default"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "experiments_run": list(experiment_results.keys()),
        "config_snapshot": config
    }


def save_metadata(metadata: Dict, output_dir: Path):
    """Save metadata to JSON."""
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved: {metadata_path}")


def run_pipeline(config: Dict, experiments_to_run: Optional[List[str]] = None):
    """
    Run the full experiment pipeline.
    
    Args:
        config: Loaded configuration
        experiments_to_run: Optional list of specific experiments to run
    """
    print("\n" + "=" * 70)
    print("SPECTRA UNIFIED PIPELINE")
    print("=" * 70)
    print(f"  Config: {config.get('experiment', {}).get('name', 'unnamed')}")
    print(f"  Seed: {config.get('experiment', {}).get('seed', 42)}")
    
    # Set seed
    seed = config.get("experiment", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Get models to run
    models_config = config.get("models", [])
    enabled_models = [m for m in models_config if m.get("enabled", True)]
    
    if not enabled_models:
        print("  No enabled models found in config!")
        return
    
    # Get context lengths (handle dict format)
    ctx_config = config.get("context_lengths", [512])
    if isinstance(ctx_config, dict):
        context_lengths = ctx_config.get("enabled", [512])
    else:
        context_lengths = ctx_config
    
    # Determine experiments to run
    exp_config = config.get("experiments", {})
    if experiments_to_run:
        enabled_exps = experiments_to_run
    else:
        enabled_exps = [name for name, cfg in exp_config.items() 
                       if isinstance(cfg, dict) and cfg.get("enabled", False)]
    
    print(f"\n  Models: {[m['name'].split('/')[-1] for m in enabled_models]}")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Experiments: {enabled_exps}")
    print("=" * 70)
    
    # Run for each model
    for model_config in enabled_models:
        model_name = model_config["name"]
        model_short = model_name.split("/")[-1]
        
        print(f"\n{'#' * 70}")
        print(f"# MODEL: {model_short}")
        print(f"{'#' * 70}")
        
        # Load model
        adapter = load_model(model_name, family=model_config.get("family"))
        
        # Run for each context length
        for ctx_len in context_lengths:
            print(f"\n{'=' * 60}")
            print(f"Context Length: {ctx_len}")
            print("=" * 60)
            
            # Create output directory
            output_base = Path(config.get("output", {}).get("base_dir", "results"))
            output_dir = output_base / model_short / f"ctx{ctx_len}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load ALL samples (tries data/ctx{N}/ first, then falls back)
            all_samples = load_all_samples(config, adapter.tokenizer, ctx_len=ctx_len)
            
            # Context length enforcement (check ALL samples)
            valid_samples = []
            for s in all_samples:
                actual_len = s.shape[1]
                if actual_len >= ctx_len:
                    # Truncate to exact length
                    valid_samples.append(s[:, :ctx_len])
            
            all_samples = valid_samples
            
            if not all_samples:
                print(f"  ⚠️ All samples too short (need {ctx_len})")
                print(f"  ⚠️ SKIPPING ctx{ctx_len}")
                continue
            
            print(f"  Input: {len(all_samples)} samples × {ctx_len} tokens")
            
            # Create context for experiments
            context = {
                "adapter": adapter,
                "input_ids": all_samples[0],  # Backward compat: first sample
                "all_samples": all_samples,   # NEW: all samples for aggregation
                "output_dir": str(output_dir),
                "config": config,
                "context_length": ctx_len
            }
            
            experiment_results = {}
            
            # Run experiments in order
            for exp_name in enabled_exps:
                if not ExperimentRegistry.is_registered(exp_name):
                    print(f"\n  ⚠️ Experiment '{exp_name}' not registered, skipping")
                    continue
                
                try:
                    result = ExperimentRegistry.run(exp_name, context)
                    experiment_results[exp_name] = result
                    
                    # Pass results to subsequent experiments
                    if exp_name == "exp2b":
                        context["exp2b_results"] = result
                    
                except Exception as e:
                    print(f"\n  ❌ {exp_name} failed: {e}")
                    experiment_results[exp_name] = {"success": False, "error": str(e)}
                
                torch.cuda.empty_cache()
            
            # Save metadata
            if config.get("reproducibility", {}).get("save_metadata", True):
                metadata = create_metadata(config, adapter, ctx_len, experiment_results)
                save_metadata(metadata, output_dir)
            
            # Summary
            print(f"\n{'=' * 60}")
            print("SUMMARY")
            print("=" * 60)
            for exp, result in experiment_results.items():
                status = "✅" if result.get("success", False) else "❌"
                print(f"  {exp}: {status}")
        
        # Clean up model
        del adapter
        torch.cuda.empty_cache()
    
    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spectra Unified Pipeline Runner"
    )
    parser.add_argument(
        "--config", type=str, default="config/experiment.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--experiments", type=str, default=None,
        help="Comma-separated list of experiments to run (overrides config)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Parse experiments if specified
    experiments = None
    if args.experiments:
        experiments = [e.strip() for e in args.experiments.split(",")]
    
    # Run pipeline
    run_pipeline(config, experiments)


if __name__ == "__main__":
    main()
