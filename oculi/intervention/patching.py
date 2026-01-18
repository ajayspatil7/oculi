"""
Activation Patching
===================

Causal intervention system for establishing which components are responsible 
for model outputs.

The activation patching paradigm:
1. Run model on "clean" input → get clean activations
2. Run model on "corrupted" input → get corrupted output
3. Patch clean activations into corrupted run → measure recovery

If patching component X recovers the clean behavior, X is causally important.

Usage:
    >>> from oculi.intervention import ActivationPatch, PatchingContext
    >>> 
    >>> # Capture from clean run
    >>> clean = adapter.capture_full(clean_ids)
    >>> 
    >>> # Patch layer 20 MLP output
    >>> patch = ActivationPatch(
    ...     layer=20,
    ...     component='mlp_out',
    ...     source_activation=clean.mlp.output[20]
    ... )
    >>> 
    >>> with PatchingContext(adapter, [patch]):
    ...     patched_output = model(corrupted_ids)

Components:
    - 'residual_pre_attn': Layer input
    - 'residual_post_attn': After attention
    - 'residual_post_mlp': After MLP  
    - 'attn_out': Attention output
    - 'mlp_out': MLP output
    - 'head': Single attention head output (requires head parameter)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable, Dict, Any, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from oculi.models.base import AttentionAdapter


# =============================================================================
# PATCH CONFIGURATION
# =============================================================================

# Valid component names for patching
VALID_COMPONENTS = frozenset([
    'residual_pre_attn',
    'residual_post_attn', 
    'residual_post_mlp',
    'attn_out',
    'mlp_out',
    'head',
])


@dataclass
class PatchConfig:
    """
    Configuration for what to patch.
    
    Attributes:
        layer: Layer index to patch (0-indexed)
        component: Which component to patch (see VALID_COMPONENTS)
        head: Head index (required if component='head')
        tokens: Token positions to patch (None = all tokens)
        
    Example:
        >>> # Patch MLP output at layer 20
        >>> config = PatchConfig(layer=20, component='mlp_out')
        >>> 
        >>> # Patch specific head
        >>> config = PatchConfig(layer=20, component='head', head=5)
        >>> 
        >>> # Patch only token positions 2 and 3
        >>> config = PatchConfig(layer=20, component='mlp_out', tokens=[2, 3])
    """
    layer: int
    component: str
    head: Optional[int] = None
    tokens: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.component not in VALID_COMPONENTS:
            raise ValueError(
                f"Invalid component '{self.component}'. "
                f"Must be one of: {sorted(VALID_COMPONENTS)}"
            )
        
        if self.component == 'head' and self.head is None:
            raise ValueError("head parameter is required when component='head'")
        
        if self.component != 'head' and self.head is not None:
            raise ValueError(
                f"head parameter should only be set for component='head', "
                f"got component='{self.component}'"
            )
    
    def validate(self, adapter: 'AttentionAdapter') -> None:
        """
        Validate config against adapter architecture.
        
        Args:
            adapter: The adapter to validate against
            
        Raises:
            ValueError: If config is invalid for this adapter
        """
        n_layers = adapter.num_layers()
        
        if not (0 <= self.layer < n_layers):
            raise ValueError(
                f"Layer {self.layer} out of range. Model has {n_layers} layers (0-{n_layers-1})"
            )
        
        if self.head is not None:
            n_heads = adapter.num_heads()
            if not (0 <= self.head < n_heads):
                raise ValueError(
                    f"Head {self.head} out of range. Model has {n_heads} heads (0-{n_heads-1})"
                )


# =============================================================================
# ACTIVATION PATCH
# =============================================================================

@dataclass
class ActivationPatch:
    """
    A single patch operation.
    
    Contains the configuration and source activation to patch.
    
    Attributes:
        config: PatchConfig specifying what to patch
        source_activation: Tensor from source run to patch in
        
    Shape Requirements:
        - residual_* / mlp_out / attn_out: [T, H] where T=tokens, H=hidden_dim
        - head: [T, D] where D=head_dim
        
    Example:
        >>> clean = adapter.capture_full(clean_ids)
        >>> 
        >>> # Patch MLP output from layer 20
        >>> patch = ActivationPatch(
        ...     config=PatchConfig(layer=20, component='mlp_out'),
        ...     source_activation=clean.mlp.output[20]  # [T, H]
        ... )
    """
    config: PatchConfig
    source_activation: torch.Tensor
    
    # Convenience properties
    @property
    def layer(self) -> int:
        return self.config.layer
    
    @property
    def component(self) -> str:
        return self.config.component
    
    @property
    def head(self) -> Optional[int]:
        return self.config.head
    
    @property
    def tokens(self) -> Optional[List[int]]:
        return self.config.tokens
    
    def validate(self, adapter: 'AttentionAdapter') -> None:
        """
        Validate patch against adapter architecture.
        
        Checks:
        - Config is valid for adapter
        - Source activation has correct shape
        - Source activation is on correct device
        
        Args:
            adapter: The adapter to validate against
            
        Raises:
            ValueError: If patch is invalid
        """
        self.config.validate(adapter)
        
        # Check device compatibility
        model_device = adapter.device
        if self.source_activation.device != model_device:
            raise ValueError(
                f"Source activation on {self.source_activation.device} but "
                f"model on {model_device}. Move activation to correct device."
            )
        
        # Shape validation based on component type
        expected_shape = self._expected_shape(adapter)
        actual_shape = self.source_activation.shape
        
        if len(actual_shape) != len(expected_shape):
            raise ValueError(
                f"Source activation has wrong number of dimensions. "
                f"Expected {len(expected_shape)}D, got {len(actual_shape)}D."
            )
        
        # Check each dimension (allow -1 for "any")
        for i, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
            if expected != -1 and expected != actual:
                raise ValueError(
                    f"Source activation shape mismatch at dimension {i}. "
                    f"Expected {expected_shape}, got {actual_shape}."
                )
    
    def _expected_shape(self, adapter: 'AttentionAdapter') -> Tuple[int, ...]:
        """Get expected shape for source activation."""
        head_dim = adapter.head_dim()
        hidden_dim = adapter.num_heads() * head_dim

        if self.component in ('residual_pre_attn', 'residual_post_attn',
                              'residual_post_mlp', 'attn_out', 'mlp_out'):
            return (-1, hidden_dim)  # [T, H], T can vary
        elif self.component == 'head':
            return (-1, head_dim)  # [T, D], T can vary
        else:
            raise ValueError(f"Unknown component: {self.component}")
    
    def apply(self, target_activation: torch.Tensor) -> torch.Tensor:
        """
        Apply this patch to a target activation.
        
        Args:
            target_activation: Activation from the run being patched
            
        Returns:
            Patched activation (source values replacing target values)
        """
        if self.tokens is None:
            # Replace all tokens
            return self.source_activation.clone()
        else:
            # Replace only specific tokens
            result = target_activation.clone()
            for token_idx in self.tokens:
                if token_idx < result.shape[0]:
                    result[token_idx] = self.source_activation[token_idx]
            return result


# =============================================================================
# PATCHING RESULT
# =============================================================================

@dataclass
class PatchingResult:
    """
    Result of a single patching experiment.
    
    Attributes:
        config: The patch configuration used
        metric_clean: Metric value on clean run
        metric_corrupted: Metric value on corrupted run  
        metric_patched: Metric value after patching
        recovery: Recovery score (0 = no recovery, 1 = full recovery)
        
    Recovery Formula:
        recovery = (metric_patched - metric_corrupted) / (metric_clean - metric_corrupted)
        
    Interpretation:
        - recovery ≈ 1.0: This component fully explains the difference
        - recovery ≈ 0.0: This component doesn't matter
        - recovery > 1.0: Patching overcompensated
        - recovery < 0.0: Patching made things worse
    """
    config: PatchConfig
    metric_clean: float
    metric_corrupted: float
    metric_patched: float
    recovery: float
    
    @classmethod
    def compute(
        cls,
        config: PatchConfig,
        metric_clean: float,
        metric_corrupted: float,
        metric_patched: float,
    ) -> 'PatchingResult':
        """
        Compute a PatchingResult with recovery score.
        
        Args:
            config: Patch configuration
            metric_clean: Metric from clean run
            metric_corrupted: Metric from corrupted run
            metric_patched: Metric after patching
            
        Returns:
            PatchingResult with computed recovery
        """
        diff = metric_clean - metric_corrupted
        if abs(diff) < 1e-10:
            # No difference between clean and corrupted
            recovery = 0.0
        else:
            recovery = (metric_patched - metric_corrupted) / diff
        
        return cls(
            config=config,
            metric_clean=metric_clean,
            metric_corrupted=metric_corrupted,
            metric_patched=metric_patched,
            recovery=recovery,
        )
    
    def __repr__(self) -> str:
        return (
            f"PatchingResult(layer={self.config.layer}, "
            f"component='{self.config.component}', "
            f"recovery={self.recovery:.3f})"
        )


# =============================================================================
# PATCHING RESULTS COLLECTION
# =============================================================================

@dataclass
class PatchingSweepResult:
    """
    Results from a systematic patching sweep.
    
    Contains results for multiple layer/component combinations.
    
    Attributes:
        results: List of individual PatchingResult objects
        layers: Layers that were swept
        components: Components that were swept
        
    Methods:
        recovery_matrix(): Get [layers x components] recovery scores
        top_results(k): Get top-k results by recovery
        by_layer(layer): Get all results for a specific layer
        by_component(component): Get all results for a specific component
    """
    results: List[PatchingResult]
    layers: List[int]
    components: List[str]
    
    @property
    def n_layers(self) -> int:
        return len(self.layers)
    
    @property
    def n_components(self) -> int:
        return len(self.components)
    
    def recovery_matrix(self) -> torch.Tensor:
        """
        Get recovery scores as a [layers x components] matrix.
        
        Returns:
            Tensor of shape [n_layers, n_components] with recovery scores
        """
        matrix = torch.zeros(self.n_layers, self.n_components)
        
        layer_to_idx = {l: i for i, l in enumerate(self.layers)}
        comp_to_idx = {c: i for i, c in enumerate(self.components)}
        
        for result in self.results:
            layer_idx = layer_to_idx.get(result.config.layer)
            comp_idx = comp_to_idx.get(result.config.component)
            
            if layer_idx is not None and comp_idx is not None:
                matrix[layer_idx, comp_idx] = result.recovery
        
        return matrix
    
    def top_results(self, k: int = 10) -> List[PatchingResult]:
        """Get top-k results by recovery score."""
        sorted_results = sorted(self.results, key=lambda r: r.recovery, reverse=True)
        return sorted_results[:k]
    
    def by_layer(self, layer: int) -> List[PatchingResult]:
        """Get all results for a specific layer."""
        return [r for r in self.results if r.config.layer == layer]
    
    def by_component(self, component: str) -> List[PatchingResult]:
        """Get all results for a specific component."""
        return [r for r in self.results if r.config.component == component]
    
    def summary(self) -> str:
        """Get a text summary of the sweep results."""
        lines = [
            f"Patching Sweep Results",
            f"=" * 50,
            f"Layers: {self.layers[0]}-{self.layers[-1]} ({self.n_layers} total)",
            f"Components: {', '.join(self.components)}",
            f"",
            f"Top 5 by Recovery:",
        ]
        
        for result in self.top_results(5):
            lines.append(
                f"  L{result.config.layer:2d} {result.config.component:20s}: "
                f"recovery={result.recovery:.3f}"
            )
        
        return "\n".join(lines)
