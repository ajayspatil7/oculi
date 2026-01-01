"""
Ablation Interventions
======================

Defines head ablation (zero-out) interventions.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oculi.capture.adapter import ModelAdapter


@dataclass
class HeadAblation:
    """
    Intervention that zeros out a head's output.
    
    Semantic Definition:
        head_output_new = 0
        
    Use for:
        - Measuring causal importance of specific heads
        - Testing necessity vs sufficiency of heads
        - Identifying critical circuit components
        
    Args:
        layer: Target layer index
        head: Target head index
        
    Example:
        >>> ablation = HeadAblation(layer=23, head=5)
        >>> with InterventionContext(adapter, [ablation]):
        ...     # Model now runs without this head
        ...     output = adapter.generate(prompt)
    """
    layer: int
    head: int
    
    def validate(self, adapter: 'ModelAdapter') -> None:
        """Validate intervention parameters."""
        if not 0 <= self.layer < adapter.num_layers():
            raise ValueError(
                f"Layer {self.layer} out of range [0, {adapter.num_layers()})"
            )
        if not 0 <= self.head < adapter.num_heads(self.layer):
            raise ValueError(
                f"Head {self.head} out of range [0, {adapter.num_heads(self.layer)})"
            )
    
    @property
    def component(self) -> str:
        return "attn_out"
    
    @property
    def scale_factor(self) -> float:
        """Zero scaling = ablation."""
        return 0.0
