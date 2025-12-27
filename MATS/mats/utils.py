"""
Utilities for MATS 10.0
========================

Common utilities for reproducibility, configuration, and results management.
"""

import os
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import yaml
import torch
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enforce deterministic algorithms where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set to {seed}")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load experiment configuration from YAML.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Loaded config from {config_path}")
    return config


def get_git_commit() -> Optional[str]:
    """Get current git commit hash for reproducibility."""
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha[:8]
    except Exception:
        return None


def create_output_dir(base_dir: str = "results") -> Path:
    """
    Create timestamped output directory.
    
    Args:
        base_dir: Base results directory
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created output directory: {output_dir}")
    return output_dir


def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    filename: str = "results.json",
    config: Optional[Dict] = None,
) -> Path:
    """
    Save experiment results with metadata.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        filename: Output filename
        config: Optional config to include in metadata
        
    Returns:
        Path to saved file
    """
    output_path = output_dir / filename
    
    # Add metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
    }
    
    if config:
        metadata["config"] = config
    
    # Combine with results
    output = {
        "metadata": metadata,
        "results": results,
    }
    
    # Convert numpy/torch types for JSON serialization
    output = _make_json_serializable(output)
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Saved results to {output_path}")
    return output_path


def _make_json_serializable(obj: Any) -> Any:
    """Convert numpy/torch types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


def save_config_snapshot(config: Dict, output_dir: Path) -> Path:
    """Save a copy of the config used for this run."""
    output_path = output_dir / "config_snapshot.yaml"
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return output_path


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_separator(title: str = "", char: str = "=", width: int = 60) -> None:
    """Print a separator line for console output."""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * width)
