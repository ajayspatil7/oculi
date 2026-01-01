"""
Entropy Visualization
=====================

Plots for entropy analysis results.
"""

from typing import Optional, List
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
        raise ImportError(
            "matplotlib required for visualization. "
            "Install with: pip install matplotlib"
        )


class EntropyPlots:
    """
    Entropy visualization utilities.
    
    All methods return matplotlib.Figure objects.
    User is responsible for saving/displaying.
    """
    
    @staticmethod
    def heatmap(
        entropy: torch.Tensor,
        title: str = "Mean Attention Entropy",
        cmap: str = "viridis",
        figsize: tuple = (12, 8)
    ) -> 'matplotlib.figure.Figure':
        """
        Entropy heatmap across layers and heads.
        
        Args:
            entropy: Shape [n_layers, n_heads] (mean across positions)
            title: Plot title
            cmap: Colormap name
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        _check_matplotlib()
        
        # Convert to numpy
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(entropy, aspect='auto', cmap=cmap)
        ax.set_xlabel('Head', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Entropy (nats)', fontsize=11)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def distribution(
        entropy: torch.Tensor,
        title: str = "Entropy Distribution",
        bins: int = 50,
        figsize: tuple = (10, 6)
    ) -> 'matplotlib.figure.Figure':
        """
        Histogram of entropy values.
        
        Args:
            entropy: Any shape tensor of entropy values
            title: Plot title
            bins: Number of histogram bins
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        _check_matplotlib()
        
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.cpu().numpy()
        
        # Flatten and remove NaN
        flat = entropy.flatten()
        flat = flat[~np.isnan(flat)]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.hist(flat, bins=bins, edgecolor='white', alpha=0.7)
        ax.set_xlabel('Entropy (nats)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.axvline(np.mean(flat), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(flat):.2f}')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def delta_heatmap(
        delta_entropy: torch.Tensor,
        title: str = "ΔEntropy (Treatment - Control)",
        figsize: tuple = (12, 8)
    ) -> 'matplotlib.figure.Figure':
        """
        Heatmap of entropy change per layer/head.
        
        Diverging colormap: red = increase, blue = decrease.
        
        Args:
            delta_entropy: Shape [n_layers, n_heads]
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        _check_matplotlib()
        
        if isinstance(delta_entropy, torch.Tensor):
            delta_entropy = delta_entropy.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Diverging colormap centered at 0
        vmax = np.abs(delta_entropy).max()
        im = ax.imshow(delta_entropy, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax)
        
        ax.set_xlabel('Head', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('ΔEntropy', fontsize=11)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def layer_profile(
        entropy: torch.Tensor,
        title: str = "Entropy by Layer",
        figsize: tuple = (10, 6)
    ) -> 'matplotlib.figure.Figure':
        """
        Mean entropy per layer with error bars.
        
        Args:
            entropy: Shape [n_layers, n_heads, n_tokens] or [n_layers, n_heads]
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        _check_matplotlib()
        
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.cpu().numpy()
        
        # Average to [n_layers]
        if entropy.ndim == 3:
            layer_mean = np.nanmean(entropy, axis=(1, 2))
            layer_std = np.nanstd(entropy, axis=(1, 2))
        else:
            layer_mean = np.nanmean(entropy, axis=1)
            layer_std = np.nanstd(entropy, axis=1)
        
        layers = np.arange(len(layer_mean))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.errorbar(layers, layer_mean, yerr=layer_std, fmt='o-', 
                    capsize=3, capthick=1)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Mean Entropy (nats)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
