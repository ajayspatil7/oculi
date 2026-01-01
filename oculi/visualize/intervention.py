"""
Intervention Effect Visualization
=================================

Plots for intervention analysis results.
"""

from typing import List, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _check_matplotlib():
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for visualization.")


class InterventionPlots:
    """
    Intervention effect visualization.
    
    Primary plot: Goldilocks curve (accuracy vs alpha).
    """
    
    @staticmethod
    def alpha_curve(
        alphas: List[float],
        metric_values: List[float],
        metric_name: str = "Accuracy",
        title: str = "Intervention Effect",
        figsize: tuple = (10, 6),
        annotate_peak: bool = True
    ) -> 'matplotlib.figure.Figure':
        """
        Plot metric vs alpha (the "Goldilocks curve").
        
        Annotates peak value and location.
        
        Args:
            alphas: List of alpha values tested
            metric_values: Corresponding metric values
            metric_name: Name of the metric for axis label
            title: Plot title
            figsize: Figure size
            annotate_peak: Whether to annotate the peak
            
        Returns:
            matplotlib Figure object
        """
        _check_matplotlib()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(alphas, metric_values, 'o-', linewidth=2, markersize=8,
                color='#2ecc71')
        ax.fill_between(alphas, 0, metric_values, alpha=0.2, color='#2ecc71')
        
        # Find and annotate peak
        if annotate_peak:
            peak_idx = np.argmax(metric_values)
            peak_alpha = alphas[peak_idx]
            peak_value = metric_values[peak_idx]
            
            ax.axvline(peak_alpha, color='gold', linestyle='--', 
                       linewidth=2, alpha=0.7)
            ax.annotate(
                f'Peak: α={peak_alpha}\n{metric_name}={peak_value:.3f}',
                xy=(peak_alpha, peak_value),
                xytext=(peak_alpha + 0.5, peak_value * 0.95),
                fontsize=11,
                arrowprops=dict(arrowstyle='->', color='gold')
            )
        
        ax.set_xlabel('α (Scaling Factor)', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Mark baseline (alpha=1.0)
        if 1.0 in alphas:
            baseline_idx = alphas.index(1.0)
            ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
            ax.annotate('Baseline', xy=(1.0, metric_values[baseline_idx]),
                       fontsize=9, alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def multi_metric_curve(
        alphas: List[float],
        metrics: dict,  # {name: values}
        title: str = "Multi-Metric Intervention Effect",
        figsize: tuple = (12, 6)
    ) -> 'matplotlib.figure.Figure':
        """
        Plot multiple metrics vs alpha on same figure.
        
        Args:
            alphas: List of alpha values
            metrics: Dict mapping metric name to list of values
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        _check_matplotlib()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
        
        for (name, values), color in zip(metrics.items(), colors):
            ax.plot(alphas, values, 'o-', linewidth=2, markersize=6,
                    label=name, color=color)
        
        ax.set_xlabel('α (Scaling Factor)', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def head_comparison(
        heads: List[tuple],  # [(layer, head), ...]
        effects: List[float],
        title: str = "Head Intervention Effects",
        figsize: tuple = (12, 6)
    ) -> 'matplotlib.figure.Figure':
        """
        Bar chart comparing intervention effects across heads.
        
        Args:
            heads: List of (layer, head) tuples
            effects: Effect magnitude for each head
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        _check_matplotlib()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        labels = [f"L{l}H{h}" for l, h in heads]
        x = np.arange(len(labels))
        
        colors = ['#2ecc71' if e > 0 else '#e74c3c' for e in effects]
        ax.bar(x, effects, color=colors, edgecolor='white')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel('Head', fontsize=12)
        ax.set_ylabel('Effect Magnitude', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.axhline(0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        return fig
