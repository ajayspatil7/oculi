"""
Correlation Visualization
=========================

Plots for correlation analysis results.
"""

from typing import Optional
import torch
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


class CorrelationPlots:
    """
    Correlation visualization utilities.
    """
    
    @staticmethod
    def scatter(
        x: torch.Tensor,
        y: torch.Tensor,
        x_label: str = "X",
        y_label: str = "Y",
        title: str = "Correlation",
        show_regression: bool = True,
        figsize: tuple = (8, 8)
    ) -> 'matplotlib.figure.Figure':
        """
        Scatter plot with regression line and correlation coefficient.
        
        Args:
            x, y: Tensors of values to plot
            x_label, y_label: Axis labels
            title: Plot title
            show_regression: Whether to show regression line
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        _check_matplotlib()
        
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        
        # Flatten and remove NaN
        x_flat = x.flatten()
        y_flat = y.flatten()
        mask = ~(np.isnan(x_flat) | np.isnan(y_flat))
        x_clean = x_flat[mask]
        y_clean = y_flat[mask]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter
        ax.scatter(x_clean, y_clean, alpha=0.5, s=10)
        
        # Regression line
        if show_regression and len(x_clean) > 1:
            coeffs = np.polyfit(x_clean, y_clean, 1)
            r = np.corrcoef(x_clean, y_clean)[0, 1]
            
            x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
            y_line = coeffs[0] * x_line + coeffs[1]
            
            ax.plot(x_line, y_line, 'r-', linewidth=2, 
                    label=f'r = {r:.3f}')
            ax.legend(fontsize=11)
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def correlation_heatmap(
        correlation: torch.Tensor,
        title: str = "Correlation by Layer/Head",
        figsize: tuple = (12, 8)
    ) -> 'matplotlib.figure.Figure':
        """
        Heatmap of correlation values across layers and heads.
        
        Args:
            correlation: Shape [n_layers, n_heads]
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        _check_matplotlib()
        
        if isinstance(correlation, torch.Tensor):
            correlation = correlation.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Diverging colormap for correlation (-1 to 1)
        im = ax.imshow(correlation, aspect='auto', cmap='RdBu_r',
                       vmin=-1, vmax=1)
        
        ax.set_xlabel('Head', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Pearson r', fontsize=11)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def norm_entropy_scatter(
        q_norms: torch.Tensor,
        entropy: torch.Tensor,
        layer: int,
        head: int,
        figsize: tuple = (8, 8)
    ) -> 'matplotlib.figure.Figure':
        """
        Convenience method for Q-norm vs entropy scatter.
        
        Args:
            q_norms: Shape [n_layers, n_heads, n_tokens]
            entropy: Shape [n_layers, n_heads, n_tokens]
            layer, head: Which layer/head to plot
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        x = q_norms[layer, head]
        y = entropy[layer, head]
        
        return CorrelationPlots.scatter(
            x, y,
            x_label='Query Norm (||Q||₂)',
            y_label='Attention Entropy (nats)',
            title=f'Layer {layer}, Head {head}',
            figsize=figsize
        )
