"""
Stratified Views
================

Helpers for viewing analysis results by layer, head, or token.

These utilities make it easy to extract specific slices from
captured data and analysis results.
"""

from typing import List, Optional, Union, Tuple
from dataclasses import dataclass
import torch
from oculi.capture.structures import AttentionCapture


@dataclass
class StratifiedResult:
    """
    Container for stratified analysis results.
    
    Provides convenient access patterns for layer/head/token indexing.
    """
    data: torch.Tensor
    dim_names: Tuple[str, ...]  # e.g., ('layer', 'head', 'token')
    
    def by_layer(self, layer: int) -> torch.Tensor:
        """Extract data for a specific layer."""
        if 'layer' not in self.dim_names:
            raise ValueError("Data does not have layer dimension")
        layer_dim = self.dim_names.index('layer')
        return self.data.select(layer_dim, layer)
    
    def by_head(self, head: int) -> torch.Tensor:
        """Extract data for a specific head."""
        if 'head' not in self.dim_names:
            raise ValueError("Data does not have head dimension")
        head_dim = self.dim_names.index('head')
        return self.data.select(head_dim, head)
    
    def by_layer_head(self, layer: int, head: int) -> torch.Tensor:
        """Extract data for a specific (layer, head) pair."""
        return self.by_layer(layer).select(
            self.dim_names.index('head') - 1, head
        )
    
    def mean_over(self, dim: str) -> torch.Tensor:
        """Compute mean over specified dimension."""
        if dim not in self.dim_names:
            raise ValueError(f"Dimension '{dim}' not in {self.dim_names}")
        return torch.nanmean(self.data, dim=self.dim_names.index(dim))


class StratifiedView:
    """
    Utility class for stratified views of attention data.
    
    Provides helpers to slice and aggregate data by:
    - Layer
    - Head  
    - Token position
    - Layer groups (early/middle/late)
    - Head groups (by behavior)
    
    Example:
        >>> view = StratifiedView(capture)
        >>> # Get Q norms for layer 20
        >>> q_norms_l20 = view.q_norms_by_layer(20)
        >>> # Get mean entropy for all layers
        >>> mean_entropy = view.entropy_by_layer()
    """
    
    def __init__(self, capture: AttentionCapture):
        """
        Initialize with a capture.
        
        Args:
            capture: AttentionCapture from model
        """
        self.capture = capture
        self._cache = {}
    
    # =========================================================================
    # LAYER VIEWS
    # =========================================================================
    
    def by_layer(
        self,
        data: torch.Tensor,
        layer: int
    ) -> torch.Tensor:
        """
        Extract data for a specific layer.
        
        Args:
            data: Tensor with shape [n_layers, ...]
            layer: Layer index
            
        Returns:
            Tensor with layer dimension removed
        """
        if layer < 0 or layer >= data.shape[0]:
            raise IndexError(f"Layer {layer} out of range [0, {data.shape[0]})")
        return data[layer]
    
    def by_layers(
        self,
        data: torch.Tensor,
        layers: List[int]
    ) -> torch.Tensor:
        """
        Extract data for multiple layers.
        
        Args:
            data: Tensor with shape [n_layers, ...]
            layers: List of layer indices
            
        Returns:
            Tensor with shape [len(layers), ...]
        """
        return data[layers]
    
    def early_layers(
        self,
        data: torch.Tensor,
        fraction: float = 0.33
    ) -> torch.Tensor:
        """Get data for early layers (first third by default)."""
        n_early = max(1, int(data.shape[0] * fraction))
        return data[:n_early]
    
    def middle_layers(
        self,
        data: torch.Tensor,
        fraction: float = 0.33
    ) -> torch.Tensor:
        """Get data for middle layers."""
        n_layers = data.shape[0]
        start = int(n_layers * fraction)
        end = int(n_layers * (1 - fraction))
        return data[start:end]
    
    def late_layers(
        self,
        data: torch.Tensor,
        fraction: float = 0.33
    ) -> torch.Tensor:
        """Get data for late layers (last third by default)."""
        n_late = max(1, int(data.shape[0] * fraction))
        return data[-n_late:]
    
    # =========================================================================
    # HEAD VIEWS
    # =========================================================================
    
    def by_head(
        self,
        data: torch.Tensor,
        head: int
    ) -> torch.Tensor:
        """
        Extract data for a specific head.
        
        Args:
            data: Tensor with shape [..., n_heads, ...]
            head: Head index (dimension 1 by default)
            
        Returns:
            Tensor with head dimension removed
        """
        # Assume head is dimension 1 (after layer)
        return data[:, head]
    
    def by_layer_head(
        self,
        data: torch.Tensor,
        layer: int,
        head: int
    ) -> torch.Tensor:
        """Extract data for a specific (layer, head) pair."""
        return data[layer, head]
    
    # =========================================================================
    # TOKEN VIEWS
    # =========================================================================
    
    def by_token(
        self,
        data: torch.Tensor,
        token: int,
        token_dim: int = -1
    ) -> torch.Tensor:
        """Extract data at a specific token position."""
        return data.select(token_dim, token)
    
    def final_token(
        self,
        data: torch.Tensor,
        token_dim: int = -1
    ) -> torch.Tensor:
        """Extract data at the final token position."""
        return data.select(token_dim, -1)
    
    def token_range(
        self,
        data: torch.Tensor,
        start: int,
        end: int,
        token_dim: int = -1
    ) -> torch.Tensor:
        """Extract data for a range of token positions."""
        return data.narrow(token_dim, start, end - start)
    
    # =========================================================================
    # AGGREGATION VIEWS
    # =========================================================================
    
    def mean_by_layer(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute mean over all dimensions except layer.
        
        Args:
            data: Tensor with shape [n_layers, ...]
            
        Returns:
            Tensor with shape [n_layers]
        """
        dims = list(range(1, data.ndim))
        return torch.nanmean(data, dim=dims)
    
    def mean_by_head(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute mean over all dimensions except head.
        
        Args:
            data: Tensor with shape [n_layers, n_heads, ...]
            
        Returns:
            Tensor with shape [n_heads]
        """
        # Mean over layers and other dims
        dims = [0] + list(range(2, data.ndim))
        return torch.nanmean(data, dim=dims)
    
    def mean_by_layer_head(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute mean over token dimension only.
        
        Args:
            data: Tensor with shape [n_layers, n_heads, n_tokens]
            
        Returns:
            Tensor with shape [n_layers, n_heads]
        """
        if data.ndim < 3:
            return data
        return torch.nanmean(data, dim=-1)


def find_extreme_heads(
    data: torch.Tensor,
    k: int = 5,
    highest: bool = True
) -> List[Tuple[int, int, float]]:
    """
    Find heads with extreme values.
    
    Args:
        data: Tensor with shape [n_layers, n_heads]
        k: Number of extreme heads to find
        highest: If True, find highest values; else lowest
        
    Returns:
        List of (layer, head, value) tuples, sorted by value
    """
    flat = data.flatten()
    
    if highest:
        _, indices = torch.topk(flat, k=min(k, flat.numel()))
    else:
        _, indices = torch.topk(-flat, k=min(k, flat.numel()))
        
    results = []
    n_heads = data.shape[1]
    
    for idx in indices:
        layer = idx.item() // n_heads
        head = idx.item() % n_heads
        value = data[layer, head].item()
        results.append((layer, head, value))
    
    return results
