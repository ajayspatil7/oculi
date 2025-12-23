"""
Spectra Experiment Registry
===========================

Centralized registry for all experiments.
Experiments are registered and can be run by name.
"""

from typing import Dict, Callable, Any, Optional
from pathlib import Path
import importlib


class ExperimentRegistry:
    """
    Registry for experiment modules.
    
    Usage:
        registry = ExperimentRegistry()
        registry.register("exp0", exp0_runner)
        registry.run("exp0", context)
    """
    
    _experiments: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str, runner: Callable):
        """Register an experiment runner function."""
        cls._experiments[name] = runner
    
    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        """Get experiment runner by name."""
        return cls._experiments.get(name)
    
    @classmethod
    def run(cls, name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run an experiment by name."""
        runner = cls.get(name)
        if runner is None:
            raise ValueError(f"Unknown experiment: {name}")
        return runner(context)
    
    @classmethod
    def list_experiments(cls) -> list:
        """List all registered experiments."""
        return list(cls._experiments.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if experiment is registered."""
        return name in cls._experiments


def experiment(name: str):
    """
    Decorator to register an experiment runner.
    
    Usage:
        @experiment("exp0")
        def run_exp0(context):
            ...
    """
    def decorator(func: Callable):
        ExperimentRegistry.register(name, func)
        return func
    return decorator
