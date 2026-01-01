"""
Intervention Context Manager
============================

Provides a context manager for applying interventions during generation.
"""

from typing import List, Union, TYPE_CHECKING
from contextlib import contextmanager

from oculi.intervention.scalers import QScaler, KScaler, SpectraScaler
from oculi.intervention.ablation import HeadAblation

if TYPE_CHECKING:
    from oculi.capture.adapter import ModelAdapter

# Type alias for any intervention
Intervention = Union[QScaler, KScaler, SpectraScaler, HeadAblation]


class InterventionContext:
    """
    Context manager for applying interventions during generation.
    
    Interventions are automatically removed on context exit,
    ensuring the model returns to its original state.
    
    Usage:
        >>> scaler = SpectraScaler(layer=23, head=5, alpha=1.5)
        >>> with InterventionContext(adapter, [scaler]):
        ...     output = adapter.generate(prompt)
        >>> # Interventions automatically removed here
        
    Multiple interventions:
        >>> interventions = [
        ...     SpectraScaler(layer=23, head=5, alpha=1.5),
        ...     HeadAblation(layer=20, head=3),
        ... ]
        >>> with InterventionContext(adapter, interventions):
        ...     output = adapter.generate(prompt)
            
    Attributes:
        adapter: The ModelAdapter instance
        interventions: List of Intervention objects
        hook_handles: Internal list of hook handles for cleanup
    """
    
    def __init__(
        self,
        adapter: 'ModelAdapter',
        interventions: List[Intervention]
    ):
        """
        Initialize intervention context.
        
        Args:
            adapter: ModelAdapter instance
            interventions: List of intervention specifications
            
        Raises:
            ValueError: If any intervention fails validation
        """
        self.adapter = adapter
        self.interventions = interventions
        self._hook_handles: List[str] = []
        self._entered = False
        
        # Validate all interventions upfront
        for intervention in interventions:
            intervention.validate(adapter)
    
    def __enter__(self) -> 'InterventionContext':
        """
        Apply all interventions.
        
        This registers hooks with the model that modify forward pass behavior.
        """
        if self._entered:
            raise RuntimeError("InterventionContext already entered")
        
        self._entered = True
        
        for intervention in self.interventions:
            self._apply_intervention(intervention)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Remove all interventions.
        
        Always runs, even if an exception occurred.
        """
        self._cleanup()
        self._entered = False
        return False  # Don't suppress exceptions
    
    def _apply_intervention(self, intervention: Intervention) -> None:
        """
        Apply a single intervention by registering appropriate hooks.
        
        Delegates to private layer for actual hook implementation.
        """
        from oculi._private.hooks.intervention import create_intervention_hook
        
        if isinstance(intervention, SpectraScaler):
            # SpectraScaler needs both Q and K hooks
            q_hook = create_intervention_hook(
                'q', intervention.head, intervention.q_scale_factor
            )
            k_hook = create_intervention_hook(
                'k', intervention.head, intervention.k_scale_factor,
                is_gqa=self.adapter.attention_structure().is_gqa,
                gqa_ratio=self.adapter.attention_structure().gqa_ratio
            )
            
            handle_q = self.adapter.add_hook(
                q_hook, intervention.layer, 'q', 'post_rope'
            )
            handle_k = self.adapter.add_hook(
                k_hook, intervention.layer, 'k', 'post_rope'
            )
            self._hook_handles.extend([handle_q, handle_k])
            
        elif isinstance(intervention, QScaler):
            hook = create_intervention_hook(
                'q', intervention.head, intervention.scale_factor
            )
            handle = self.adapter.add_hook(
                hook, intervention.layer, 'q', 'post_rope'
            )
            self._hook_handles.append(handle)
            
        elif isinstance(intervention, KScaler):
            hook = create_intervention_hook(
                'k', intervention.kv_head, intervention.scale_factor
            )
            handle = self.adapter.add_hook(
                hook, intervention.layer, 'k', 'post_rope'
            )
            self._hook_handles.append(handle)
            
        elif isinstance(intervention, HeadAblation):
            hook = create_intervention_hook(
                'attn_out', intervention.head, 0.0
            )
            handle = self.adapter.add_hook(
                hook, intervention.layer, 'attn_out', 'post_proj'
            )
            self._hook_handles.append(handle)
    
    def _cleanup(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            try:
                self.adapter.remove_hook(handle)
            except Exception:
                pass  # Best effort cleanup
        self._hook_handles.clear()


@contextmanager
def intervention_context(
    adapter: 'ModelAdapter',
    interventions: List[Intervention]
):
    """
    Functional form of InterventionContext.
    
    Usage:
        >>> with intervention_context(adapter, [scaler]) as ctx:
        ...     output = adapter.generate(prompt)
    """
    ctx = InterventionContext(adapter, interventions)
    with ctx:
        yield ctx
