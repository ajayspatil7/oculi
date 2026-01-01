"""
Intervention Metrics
====================

Quantify the effects of interventions.

Provides metrics for measuring:
- Delta entropy (change in attention focus)
- Effect size (Cohen's d, standardized effect)
- Accuracy changes
- Head importance scores
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch

from oculi.capture.structures import AttentionCapture


@dataclass
class EffectMeasurement:
    """
    Container for intervention effect measurements.
    
    Attributes:
        delta: Raw difference (treatment - control)
        effect_size: Cohen's d or similar standardized effect
        p_value: Statistical significance (if computed)
        confidence_interval: Optional (low, high) bounds
    """
    delta: float
    effect_size: float
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


class InterventionMetrics:
    """
    Metrics for quantifying intervention effects.
    
    All methods are static and pure: captures/measurements -> metrics
    
    Example:
        >>> # Compare entropy before and after intervention
        >>> delta = InterventionMetrics.delta_entropy(
        ...     capture_intervention, capture_baseline
        ... )
        
        >>> # Compute effect size
        >>> effect = InterventionMetrics.effect_size(
        ...     treatment_values, control_values
        ... )
    """
    
    @staticmethod
    def delta_entropy(
        capture_intervention: AttentionCapture,
        capture_baseline: AttentionCapture,
        method: str = 'mean',
        ignore_first: int = 2
    ) -> torch.Tensor:
        """
        Change in entropy due to intervention.
        
        Args:
            capture_intervention: Capture with intervention applied
            capture_baseline: Capture without intervention
            method: 'mean' for mean entropy, 'final' for final token
            ignore_first: Tokens to ignore (insufficient context)
            
        Returns:
            Tensor of shape [n_layers, n_heads]
            Positive = intervention increased entropy (diffused attention)
            Negative = intervention decreased entropy (focused attention)
        """
        from oculi.analysis.entropy import EntropyAnalysis
        
        return EntropyAnalysis.delta_entropy(
            capture_intervention,
            capture_baseline,
            method=method,
            ignore_first=ignore_first
        )
    
    @staticmethod
    def delta_entropy_at_head(
        capture_intervention: AttentionCapture,
        capture_baseline: AttentionCapture,
        layer: int,
        head: int,
        method: str = 'mean'
    ) -> float:
        """
        Change in entropy at a specific head.
        
        Args:
            capture_intervention: Capture with intervention
            capture_baseline: Capture without intervention
            layer: Target layer
            head: Target head
            method: Aggregation method
            
        Returns:
            Scalar delta entropy value
        """
        delta = InterventionMetrics.delta_entropy(
            capture_intervention, capture_baseline, method=method
        )
        return delta[layer, head].item()
    
    @staticmethod
    def effect_size(
        treatment: torch.Tensor,
        control: torch.Tensor,
        method: str = 'cohens_d'
    ) -> torch.Tensor:
        """
        Standardized effect size.
        
        Args:
            treatment: Values under treatment condition
            control: Values under control condition
            method: 
                'cohens_d': (mean_t - mean_c) / pooled_std
                'hedges_g': Cohen's d with small-sample correction
                
        Returns:
            Effect size tensor with same shape as input (minus last dim)
        """
        # Compute means
        mean_t = torch.nanmean(treatment, dim=-1)
        mean_c = torch.nanmean(control, dim=-1)
        
        # Compute pooled standard deviation
        n_t = (~torch.isnan(treatment)).sum(dim=-1).float()
        n_c = (~torch.isnan(control)).sum(dim=-1).float()
        
        var_t = torch.nanmean((treatment - mean_t.unsqueeze(-1)) ** 2, dim=-1)
        var_c = torch.nanmean((control - mean_c.unsqueeze(-1)) ** 2, dim=-1)
        
        # Pooled std: sqrt(((n_t-1)*var_t + (n_c-1)*var_c) / (n_t + n_c - 2))
        pooled_var = ((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2).clamp(min=1)
        pooled_std = torch.sqrt(pooled_var).clamp(min=1e-8)
        
        # Cohen's d
        d = (mean_t - mean_c) / pooled_std
        
        if method == 'hedges_g':
            # Small sample correction
            # g = d * (1 - 3 / (4*(n_t + n_c) - 9))
            correction = 1 - 3 / (4 * (n_t + n_c) - 9).clamp(min=1)
            d = d * correction
        
        return d
    
    @staticmethod
    def effect_size_entropy(
        capture_intervention: AttentionCapture,
        capture_baseline: AttentionCapture,
        ignore_first: int = 2
    ) -> torch.Tensor:
        """
        Effect size for entropy change.
        
        Args:
            capture_intervention: Capture with intervention
            capture_baseline: Capture without intervention
            ignore_first: Tokens to ignore
            
        Returns:
            Effect size tensor [n_layers, n_heads]
        """
        from oculi.analysis.entropy import EntropyAnalysis
        
        entropy_int = EntropyAnalysis.token_entropy(
            capture_intervention, ignore_first=ignore_first
        )
        entropy_base = EntropyAnalysis.token_entropy(
            capture_baseline, ignore_first=ignore_first
        )
        
        return InterventionMetrics.effect_size(entropy_int, entropy_base)
    
    @staticmethod
    def head_importance(
        capture_full: AttentionCapture,
        capture_ablated: AttentionCapture,
        metric: str = 'entropy'
    ) -> torch.Tensor:
        """
        Measure importance of a head by ablation effect.
        
        Args:
            capture_full: Capture with all heads active
            capture_ablated: Capture with target head ablated
            metric: 'entropy' or 'norm'
            
        Returns:
            Importance score tensor
        """
        if metric == 'entropy':
            from oculi.analysis.entropy import EntropyAnalysis
            full = EntropyAnalysis.mean_entropy(capture_full)
            ablated = EntropyAnalysis.mean_entropy(capture_ablated)
        else:
            from oculi.analysis.norms import NormAnalysis
            full = NormAnalysis.q_norms(capture_full).mean(dim=-1)
            ablated = NormAnalysis.q_norms(capture_ablated).mean(dim=-1)
        
        # Importance = how much does ablation change the metric
        return torch.abs(ablated - full)
    
    @staticmethod
    def accuracy_delta(
        correct_baseline: List[bool],
        correct_intervention: List[bool]
    ) -> Dict[str, float]:
        """
        Change in accuracy due to intervention.
        
        Args:
            correct_baseline: List of correct/incorrect for baseline
            correct_intervention: List of correct/incorrect for intervention
            
        Returns:
            Dict with 'baseline_acc', 'intervention_acc', 'delta', 'relative_change'
        """
        n = len(correct_baseline)
        if n != len(correct_intervention):
            raise ValueError("Lists must have same length")
        
        baseline_acc = sum(correct_baseline) / n
        intervention_acc = sum(correct_intervention) / n
        delta = intervention_acc - baseline_acc
        
        relative = delta / max(baseline_acc, 1e-8) if baseline_acc > 0 else float('inf')
        
        return {
            'baseline_acc': baseline_acc,
            'intervention_acc': intervention_acc,
            'delta': delta,
            'relative_change': relative,
            'n_samples': n
        }
    
    @staticmethod
    def find_goldilocks_alpha(
        alphas: List[float],
        accuracies: List[float]
    ) -> Dict[str, float]:
        """
        Find the optimal alpha value (peak of Goldilocks curve).
        
        Args:
            alphas: List of alpha values tested
            accuracies: Corresponding accuracy values
            
        Returns:
            Dict with 'optimal_alpha', 'peak_accuracy', 'baseline_accuracy'
        """
        if len(alphas) != len(accuracies):
            raise ValueError("alphas and accuracies must have same length")
        
        peak_idx = max(range(len(accuracies)), key=lambda i: accuracies[i])
        
        # Find baseline (alpha = 1.0)
        baseline_idx = None
        for i, a in enumerate(alphas):
            if abs(a - 1.0) < 1e-6:
                baseline_idx = i
                break
        
        baseline_acc = accuracies[baseline_idx] if baseline_idx is not None else None
        
        return {
            'optimal_alpha': alphas[peak_idx],
            'peak_accuracy': accuracies[peak_idx],
            'baseline_accuracy': baseline_acc,
            'improvement': (accuracies[peak_idx] - baseline_acc) if baseline_acc else None
        }
