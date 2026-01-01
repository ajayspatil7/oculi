"""
Norm Analysis
=============

Compute L2 norms of Q, K, V vectors.

All methods are static and pure: AttentionCapture -> Tensor
"""

import torch
from oculi.capture.structures import AttentionCapture


class NormAnalysis:
    """
    Query, Key, Value norm computations.
    
    All methods return tensors following API contract shapes.
    
    Example:
        >>> capture = model.capture(input_ids)
        >>> q_norms = NormAnalysis.q_norms(capture)
        >>> print(q_norms.shape)  # [L, H, T]
    """
    
    @staticmethod
    def q_norms(capture: AttentionCapture) -> torch.Tensor:
        """
        L2 norms of query vectors.
        
        Args:
            capture: AttentionCapture from model
            
        Returns:
            Tensor of shape [n_layers, n_heads, n_tokens]
            
        Formula:
            ||Q||₂ = √(Σᵢ Qᵢ²)
            
        Raises:
            ValueError: If queries not captured
        """
        if capture.queries is None:
            raise ValueError("Queries not captured. Set capture_queries=True in config.")
        
        # queries: [L, H, T, D] -> norms: [L, H, T]
        return torch.linalg.norm(capture.queries, dim=-1)
    
    @staticmethod
    def k_norms(capture: AttentionCapture) -> torch.Tensor:
        """
        L2 norms of key vectors.
        
        Args:
            capture: AttentionCapture from model
            
        Returns:
            Tensor of shape [n_layers, n_kv_heads, n_tokens]
            
        Note:
            For GQA models, n_kv_heads < n_heads.
            Use capture.gqa_ratio to expand if needed.
        """
        if capture.keys is None:
            raise ValueError("Keys not captured. Set capture_keys=True in config.")
        
        # keys: [L, H_kv, T, D] -> norms: [L, H_kv, T]
        return torch.linalg.norm(capture.keys, dim=-1)
    
    @staticmethod
    def v_norms(capture: AttentionCapture) -> torch.Tensor:
        """
        L2 norms of value vectors.
        
        Args:
            capture: AttentionCapture from model
            
        Returns:
            Tensor of shape [n_layers, n_kv_heads, n_tokens]
        """
        if capture.values is None:
            raise ValueError("Values not captured. Set capture_values=True in config.")
        
        # values: [L, H_kv, T, D] -> norms: [L, H_kv, T]
        return torch.linalg.norm(capture.values, dim=-1)
    
    @staticmethod
    def q_norms_normalized(capture: AttentionCapture) -> torch.Tensor:
        """
        Query norms normalized by head dimension.
        
        Returns:
            Tensor of shape [n_layers, n_heads, n_tokens]
            
        Formula:
            ||Q||₂ / √d
            
        This normalization makes norms comparable across models
        with different head dimensions.
        """
        norms = NormAnalysis.q_norms(capture)
        return norms / (capture.head_dim ** 0.5)
