"""
Attention Pattern Analysis
==========================

Compute metrics from attention patterns.
"""

from typing import List
import torch
from oculi.capture.structures import AttentionCapture


class AttentionAnalysis:
    """
    Attention pattern analysis.
    
    Computes metrics like max attention weight, effective span.
    
    Example:
        >>> capture = model.capture(input_ids)
        >>> max_attn = AttentionAnalysis.max_weight(capture)
        >>> k_eff = AttentionAnalysis.effective_span(capture)
    """
    
    @staticmethod
    def max_weight(capture: AttentionCapture) -> torch.Tensor:
        """
        Maximum attention weight per token.
        
        Args:
            capture: AttentionCapture from model
            
        Returns:
            Tensor of shape [n_layers, n_heads, n_tokens]
            
        Formula:
            max_attn(t) = max≤ₜ p(j|t)
            
        Interpretation:
            High max_weight → attention concentrated on one token
            Low max_weight → attention spread across many tokens
        """
        if capture.patterns is None:
            raise ValueError("Patterns not captured.")
        
        # patterns: [L, H, T, T]
        # Take max along key dimension (last dim)
        max_weights, _ = torch.max(capture.patterns, dim=-1)
        return max_weights  # [L, H, T]
    
    @staticmethod
    def effective_span(
        capture: AttentionCapture,
        threshold: float = 0.9
    ) -> torch.Tensor:
        """
        Minimum tokens needed to capture threshold of attention mass.
        
        Also known as "k_eff" or "effective attention span".
        
        Args:
            capture: AttentionCapture from model
            threshold: Cumulative attention threshold (default 0.9 = 90%)
            
        Returns:
            Tensor of shape [n_layers, n_heads, n_tokens]
            Values are integers (as float32 for consistency)
            
        Formula:
            k_eff(t) = argmin_k { Σᵢ₌₁ᵏ sorted_weights[i] ≥ threshold }
            
            Where sorted_weights are attention weights sorted descending.
            
        Interpretation:
            If k_eff = 5, the top 5 tokens account for ≥90% of attention.
        """
        if capture.patterns is None:
            raise ValueError("Patterns not captured.")
        
        patterns = capture.patterns  # [L, H, T, T]
        
        # Sort attention weights descending
        sorted_patterns, _ = torch.sort(patterns, dim=-1, descending=True)
        
        # Cumulative sum
        cumsum = torch.cumsum(sorted_patterns, dim=-1)
        
        # Find first position where cumsum >= threshold
        # Add 1 because we want count, not index
        k_eff = (cumsum < threshold).sum(dim=-1) + 1
        
        return k_eff.float()  # [L, H, T]
    
    @staticmethod
    def attention_to_position(
        capture: AttentionCapture,
        target_positions: List[int]
    ) -> torch.Tensor:
        """
        Attention weight to specific token positions.
        
        Useful for measuring attention to special tokens (BOS, separators)
        or specific content tokens (hints, question marks).
        
        Args:
            capture: AttentionCapture from model
            target_positions: List of position indices to measure
            
        Returns:
            Tensor of shape [n_layers, n_heads, n_tokens, len(target_positions)]
            
        Example:
            >>> # Measure attention to first 3 tokens
            >>> attn_to_start = AttentionAnalysis.attention_to_position(
            ...     capture, target_positions=[0, 1, 2]
            ... )
        """
        if capture.patterns is None:
            raise ValueError("Patterns not captured.")
        
        # patterns: [L, H, T, T]
        # Index specific positions from key dimension
        result = capture.patterns[..., target_positions]  # [L, H, T, P]
        return result
    
    @staticmethod
    def attention_entropy_ratio(capture: AttentionCapture) -> torch.Tensor:
        """
        Ratio of actual entropy to maximum possible entropy.
        
        Returns:
            Tensor of shape [n_layers, n_heads, n_tokens]
            Values in [0, 1] where 1 = uniform attention
            
        Formula:
            ratio(t) = H(t) / log(t)
            
            Where log(t) is max entropy for uniform attention over t tokens.
        """
        from oculi.analysis.entropy import EntropyAnalysis
        
        entropy = EntropyAnalysis.token_entropy(capture, ignore_first=0)
        
        # Max entropy = log(position_index + 1) due to causal masking
        positions = torch.arange(capture.n_tokens, dtype=entropy.dtype)
        max_entropy = torch.log(positions + 1)  # [T]
        max_entropy[0] = 1.0  # Avoid division by zero
        
        # Broadcast: [L, H, T] / [T]
        ratio = entropy / max_entropy
        
        return ratio
