"""
Correlation Analysis
====================

Statistical correlation between attention metrics.
"""

from typing import Tuple
import torch
import math
from oculi.capture.structures import AttentionCapture


class CorrelationAnalysis:
    """
    Cross-metric correlation analysis.
    
    Computes Pearson and Spearman correlations between metrics.
    Includes p-value computation for statistical significance.
    
    Example:
        >>> corr = CorrelationAnalysis.norm_entropy_correlation(capture)
        >>> print(corr.shape)  # [L, H]
        
        >>> r, p = CorrelationAnalysis.pearson_with_pvalue(x, y)
        >>> significant = p < 0.05
    """
    
    @staticmethod
    def pearson(
        x: torch.Tensor,
        y: torch.Tensor,
        dim: int = -1
    ) -> torch.Tensor:
        """
        Pearson correlation coefficient.
        
        Args:
            x, y: Tensors of same shape
            dim: Dimension to correlate along (reduced in output)
            
        Returns:
            Correlation tensor with dim reduced
            
        Formula:
            r = Σ(x - μₓ)(y - μᵧ) / (n · σₓ · σᵧ)
        """
        # Handle NaN by masking
        valid_mask = ~(torch.isnan(x) | torch.isnan(y))
        
        # Replace NaN with 0 for computation (masked out anyway)
        x_clean = torch.where(valid_mask, x, torch.zeros_like(x))
        y_clean = torch.where(valid_mask, y, torch.zeros_like(y))
        
        n = valid_mask.sum(dim=dim, keepdim=True).float()
        
        # Means
        x_mean = (x_clean * valid_mask).sum(dim=dim, keepdim=True) / n.clamp(min=1)
        y_mean = (y_clean * valid_mask).sum(dim=dim, keepdim=True) / n.clamp(min=1)
        
        # Centered values
        x_centered = (x_clean - x_mean) * valid_mask
        y_centered = (y_clean - y_mean) * valid_mask
        
        # Covariance and standard deviations
        cov = (x_centered * y_centered).sum(dim=dim)
        x_std = torch.sqrt((x_centered ** 2).sum(dim=dim))
        y_std = torch.sqrt((y_centered ** 2).sum(dim=dim))
        
        # Correlation
        r = cov / (x_std * y_std).clamp(min=1e-8)
        
        return r
    
    @staticmethod
    def pearson_with_pvalue(
        x: torch.Tensor,
        y: torch.Tensor,
        dim: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pearson correlation with two-tailed p-value.
        
        Uses t-distribution to compute p-value for H0: r = 0.
        
        Args:
            x, y: Tensors of same shape
            dim: Dimension to correlate along
            
        Returns:
            Tuple of (correlation, p_value) tensors
            
        Formula:
            t = r * sqrt(n-2) / sqrt(1-r²)
            p = 2 * (1 - CDF_t(|t|, df=n-2))
        """
        r = CorrelationAnalysis.pearson(x, y, dim)
        
        # Count valid samples
        valid_mask = ~(torch.isnan(x) | torch.isnan(y))
        n = valid_mask.sum(dim=dim).float()
        
        # Compute t-statistic
        # t = r * sqrt(n-2) / sqrt(1-r^2)
        r_squared = r ** 2
        t_stat = r * torch.sqrt((n - 2).clamp(min=1)) / torch.sqrt((1 - r_squared).clamp(min=1e-10))
        
        # Compute p-value using approximation
        # For large n, t-distribution approaches normal
        # For small n, we use a simple approximation
        df = (n - 2).clamp(min=1)
        
        # p-value approximation using normal distribution for simplicity
        # More accurate would use scipy.stats.t.sf
        p_value = 2 * (1 - CorrelationAnalysis._normal_cdf(torch.abs(t_stat)))
        
        return r, p_value
    
    @staticmethod
    def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
        """
        Standard normal CDF approximation.
        
        Uses error function approximation.
        """
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    
    @staticmethod
    def spearman(
        x: torch.Tensor,
        y: torch.Tensor,
        dim: int = -1
    ) -> torch.Tensor:
        """
        Spearman rank correlation coefficient.
        
        Args:
            x, y: Tensors of same shape
            dim: Dimension to correlate along
            
        Returns:
            Rank correlation tensor with dim reduced
        """
        def _rank(tensor: torch.Tensor, dim: int) -> torch.Tensor:
            """Compute ranks along dimension."""
            sorted_indices = torch.argsort(tensor, dim=dim)
            ranks = torch.zeros_like(tensor)
            
            # This is a simplified ranking (doesn't handle ties properly)
            src = torch.arange(tensor.shape[dim], dtype=tensor.dtype, device=tensor.device)
            
            # Expand src to match tensor shape
            shape = [1] * tensor.ndim
            shape[dim] = tensor.shape[dim]
            src = src.view(shape).expand_as(tensor)
            
            ranks.scatter_(dim, sorted_indices, src)
            return ranks
        
        x_ranks = _rank(x, dim)
        y_ranks = _rank(y, dim)
        
        return CorrelationAnalysis.pearson(x_ranks, y_ranks, dim)
    
    @staticmethod
    def spearman_with_pvalue(
        x: torch.Tensor,
        y: torch.Tensor,
        dim: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Spearman rank correlation with p-value.
        
        Returns:
            Tuple of (rho, p_value) tensors
        """
        def _rank(tensor: torch.Tensor, dim: int) -> torch.Tensor:
            sorted_indices = torch.argsort(tensor, dim=dim)
            ranks = torch.zeros_like(tensor)
            src = torch.arange(tensor.shape[dim], dtype=tensor.dtype, device=tensor.device)
            shape = [1] * tensor.ndim
            shape[dim] = tensor.shape[dim]
            src = src.view(shape).expand_as(tensor)
            ranks.scatter_(dim, sorted_indices, src)
            return ranks
        
        x_ranks = _rank(x, dim)
        y_ranks = _rank(y, dim)
        
        return CorrelationAnalysis.pearson_with_pvalue(x_ranks, y_ranks, dim)
    
    @staticmethod
    def norm_entropy_correlation(
        capture: AttentionCapture,
        ignore_first: int = 2
    ) -> torch.Tensor:
        """
        Correlation between query norm and attention entropy.
        
        This is the core metric for the Spectra hypothesis:
        Strong negative correlation suggests Q magnitude controls attention focus.
        
        Args:
            capture: AttentionCapture from model
            ignore_first: Ignore first N tokens
            
        Returns:
            Tensor of shape [n_layers, n_heads]
            
        Interpretation:
            r < 0: Higher Q norm → lower entropy (focused attention)
            r = 0: No relationship
            r > 0: Higher Q norm → higher entropy (diffuse attention)
        """
        from oculi.analysis.norms import NormAnalysis
        from oculi.analysis.entropy import EntropyAnalysis
        
        q_norms = NormAnalysis.q_norms(capture)  # [L, H, T]
        entropy = EntropyAnalysis.token_entropy(capture, ignore_first=ignore_first)  # [L, H, T]
        
        # Compute correlation along token dimension
        correlation = CorrelationAnalysis.pearson(q_norms, entropy, dim=-1)  # [L, H]
        
        return correlation
    
    @staticmethod
    def norm_entropy_correlation_with_pvalue(
        capture: AttentionCapture,
        ignore_first: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Correlation between Q norm and entropy with p-values.
        
        Returns:
            Tuple of (correlation, p_value) tensors, each [n_layers, n_heads]
        """
        from oculi.analysis.norms import NormAnalysis
        from oculi.analysis.entropy import EntropyAnalysis
        
        q_norms = NormAnalysis.q_norms(capture)
        entropy = EntropyAnalysis.token_entropy(capture, ignore_first=ignore_first)
        
        return CorrelationAnalysis.pearson_with_pvalue(q_norms, entropy, dim=-1)

