"""
Experiment Registry for MATS 10.0
==================================

Simple registration system for experiments.
Allows modular experiment definition and dynamic discovery.
"""

from typing import Callable, Dict, Any, Optional, List
from functools import wraps

# Global registry
_EXPERIMENTS: Dict[str, Callable] = {}


def experiment(name: str) -> Callable:
    """
    Decorator to register an experiment function.
    
    Usage:
        @experiment("sanity")
        def run_sanity(context: Dict) -> Dict:
            ...
            return results
    
    Args:
        name: Unique experiment identifier
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Register
        if name in _EXPERIMENTS:
            print(f"Warning: Overwriting experiment '{name}'")
        _EXPERIMENTS[name] = wrapper
        
        return wrapper
    return decorator


def get_experiment(name: str) -> Optional[Callable]:
    """
    Get a registered experiment by name.
    
    Args:
        name: Experiment identifier
        
    Returns:
        Experiment function or None if not found
    """
    return _EXPERIMENTS.get(name)


def list_experiments() -> List[str]:
    """List all registered experiment names."""
    return list(_EXPERIMENTS.keys())


def run_experiment(name: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a registered experiment.
    
    Args:
        name: Experiment identifier
        context: Context dict with model, config, etc.
        
    Returns:
        Experiment results
        
    Raises:
        ValueError: If experiment not found
    """
    exp_fn = get_experiment(name)
    if exp_fn is None:
        available = list_experiments()
        raise ValueError(
            f"Experiment '{name}' not found. Available: {available}"
        )
    
    return exp_fn(context)
