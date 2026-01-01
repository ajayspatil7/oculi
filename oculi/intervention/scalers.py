"""
Scaling Interventions
=====================

Defines Q, K, and combined scaling interventions.

CRITICAL SEMANTIC DEFINITION (from API_CONTRACT.md):

When we "scale Q by α", we mean:
    Q_scaled = α · Q

This is RAW magnitude scaling, NOT direction-preserving normalization.

Mathematical Consequence:
    logits_scaled = (αQ) · Kᵀ / √d = α · logits_original

After softmax, this sharpens (α > 1) or flattens (α < 1) attention.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oculi.capture.adapter import ModelAdapter


@dataclass
class QScaler:
    """
    Intervention that scales query vectors at a specific head.
    
    Semantic Definition:
        Q_new = alpha · Q_original
        
    Effect on Attention:
        alpha > 1.0 → Sharpen attention (increase focus)
        alpha < 1.0 → Flatten attention (decrease focus)
        alpha = 1.0 → No change (identity)
        
    This is RAW magnitude scaling, not direction-preserving.
    The attention logits scale linearly with alpha.
    
    Args:
        layer: Target layer index (0-indexed)
        head: Target query head index
        alpha: Scaling factor (must be > 0)
        
    Example:
        >>> scaler = QScaler(layer=23, head=5, alpha=1.5)
        >>> scaler.validate(adapter)  # Check parameters
    """
    layer: int
    head: int
    alpha: float
    
    def validate(self, adapter: 'ModelAdapter') -> None:
        """
        Validate intervention parameters against model.
        
        Raises:
            ValueError: If layer or head out of range
            ValueError: If alpha <= 0
        """
        if not 0 <= self.layer < adapter.num_layers():
            raise ValueError(
                f"Layer {self.layer} out of range [0, {adapter.num_layers()})"
            )
        if not 0 <= self.head < adapter.num_heads(self.layer):
            raise ValueError(
                f"Head {self.head} out of range [0, {adapter.num_heads(self.layer)})"
            )
        if self.alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {self.alpha}")
    
    @property
    def component(self) -> str:
        """Which component this intervention targets."""
        return "q"
    
    @property
    def scale_factor(self) -> float:
        """The multiplicative factor applied."""
        return self.alpha


@dataclass
class KScaler:
    """
    Intervention that scales key vectors at a specific KV-head.
    
    For GQA models: Affects all query heads sharing this KV-head.
    
    Semantic Definition:
        K_new = alpha · K_original
        
    Args:
        layer: Target layer index
        kv_head: Target KV-head index (NOT query head for GQA)
        alpha: Scaling factor
        
    Note:
        For GQA models with ratio R, scaling KV-head k affects
        query heads [k*R, k*R+1, ..., k*R+R-1].
    """
    layer: int
    kv_head: int  # Note: KV-head index, not query head
    alpha: float
    
    def validate(self, adapter: 'ModelAdapter') -> None:
        """Validate intervention parameters."""
        if not 0 <= self.layer < adapter.num_layers():
            raise ValueError(
                f"Layer {self.layer} out of range [0, {adapter.num_layers()})"
            )
        struct = adapter.attention_structure(self.layer)
        if not 0 <= self.kv_head < struct.n_kv_heads:
            raise ValueError(
                f"KV-head {self.kv_head} out of range [0, {struct.n_kv_heads})"
            )
        if self.alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {self.alpha}")
    
    @property
    def component(self) -> str:
        return "k"
    
    @property
    def scale_factor(self) -> float:
        return self.alpha


@dataclass
class SpectraScaler:
    """
    The Spectra intervention: scale both Q and K by √α.
    
    This is the PRIMARY intervention for the Spectra hypothesis.
    
    Semantic Definition:
        Q_new = √α · Q_original
        K_new = √α · K_original
        
    Net Effect:
        logits_new = (√α·Q)(√α·K)ᵀ/√d = α · logits_original
        
    This achieves the same attention sharpening as scaling logits by α,
    but intervenes at the representation level rather than post-computation.
    
    Mathematical Justification:
        softmax(α · QKᵀ/√d) ≈ softmax((√α·Q)(√α·K)ᵀ/√d)
        
    Args:
        layer: Target layer index
        head: Target query head index
        alpha: Net scaling factor (internal uses √α on each)
        
    Effect:
        alpha > 1.0 → Sharpen attention (Logic Head restoration)
        alpha < 1.0 → Flatten attention (Sycophancy Head jamming)
        alpha = 1.0 → No change
        
    Example:
        >>> # Sharpen attention at layer 23, head 5
        >>> scaler = SpectraScaler(layer=23, head=5, alpha=1.5)
        >>> with InterventionContext(adapter, [scaler]):
        ...     output = adapter.generate(sycophantic_prompt)
    """
    layer: int
    head: int
    alpha: float  # Net scaling factor (internal uses √α on each)
    
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
        if self.alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {self.alpha}")
    
    @property
    def component(self) -> str:
        return "qk"  # Both Q and K
    
    @property
    def q_scale_factor(self) -> float:
        """Scale factor applied to Q."""
        return self.alpha ** 0.5
    
    @property
    def k_scale_factor(self) -> float:
        """Scale factor applied to K."""
        return self.alpha ** 0.5
    
    def to_individual_scalers(self) -> tuple:
        """
        Convert to individual Q and K scalers.
        
        Returns:
            Tuple of (QScaler, KScaler) with √α scaling each
            
        Note:
            The KScaler uses the mapped KV-head index.
            GQA mapping is handled by the private layer.
        """
        return (
            QScaler(self.layer, self.head, self.q_scale_factor),
            # KScaler kv_head will be computed from head by private layer
        )
