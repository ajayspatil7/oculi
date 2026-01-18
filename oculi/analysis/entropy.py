"""
Entropy Analysis
================

Compute entropy metrics from attention patterns.

Mathematical Reference (from API_CONTRACT.md):
    Shannon Entropy: H(t) = -Σ₌₀ᵗ p(j|t) · log(p(j|t))
    Effective Rank: exp(H)
"""

from typing import Literal, Optional
import torch
from oculi.capture.structures import AttentionCapture


class EntropyAnalysis:
    """
    Attention entropy computations.
    
    All methods are static and pure: AttentionCapture -> Tensor
    
    Example:
        >>> capture = model.capture(input_ids)
        >>> entropy = EntropyAnalysis.token_entropy(capture)
        >>> print(entropy.shape)  # [L, H, T]
    """
    
    @staticmethod
    def token_entropy(
        capture: AttentionCapture,
        ignore_first: int = 2,
        eps: float = 1e-10
    ) -> torch.Tensor:
        """
        Shannon entropy of attention distribution per token.
        
        Args:
            capture: AttentionCapture from model
            ignore_first: Set first N tokens to NaN (insufficient context)
            eps: Small constant for numerical stability
            
        Returns:
            Tensor of shape [n_layers, n_heads, n_tokens]
            First `ignore_first` positions are NaN.
            
        Formula:
            H(t) = -Σ₌₀ᵗ p(j|t) · log(p(j|t))
            
            Where p(j|t) = attention weight from token t to token j
            Only sums over j ≤ t (causal masking)
            
        Interpretation:
            - High entropy → diffuse attention (attending to many tokens)
            - Low entropy → focused attention (attending to few tokens)
            - H = 0 → deterministic (100% on one token)
            - H = log(t) → uniform over t tokens
        """
        if capture.patterns is None:
            raise ValueError("Patterns not captured. Set capture_patterns=True in config.")
        
        # patterns: [L, H, T, T] - already has causal masking applied
        patterns = capture.patterns
        
        # Clamp for numerical stability (avoid log(0))
        patterns_safe = torch.clamp(patterns, min=eps, max=1.0)
        
        # Shannon entropy: H = -Σ p * log(p)
        log_patterns = torch.log(patterns_safe)
        
        # Only compute entropy over non-zero (valid) attention weights
        # Due to causal masking, patterns[l, h, i, j] = 0 for j > i
        entropy = -torch.sum(patterns * log_patterns, dim=-1)  # [L, H, T]
        
        # Set first positions to NaN (insufficient context)
        if ignore_first > 0:
            entropy[..., :ignore_first] = float('nan')
        
        return entropy
    
    @staticmethod
    def effective_rank(
        capture: AttentionCapture,
        ignore_first: int = 2
    ) -> torch.Tensor:
        """
        Effective number of attended tokens.
        
        Args:
            capture: AttentionCapture from model
            ignore_first: Set first N tokens to NaN
            
        Returns:
            Tensor of shape [n_layers, n_heads, n_tokens]
            
        Formula:
            eff_rank = exp(H)
            
            Where H is Shannon entropy.
            
        Interpretation:
            If eff_rank = 3.5, attention is spread as if uniformly
            attending to ~3.5 tokens.
        """
        entropy = EntropyAnalysis.token_entropy(capture, ignore_first=ignore_first)
        return torch.exp(entropy)
    
    @staticmethod
    def mean_entropy(
        capture: AttentionCapture,
        ignore_first: int = 2
    ) -> torch.Tensor:
        """
        Mean entropy per (layer, head) pair.
        
        Args:
            capture: AttentionCapture from model
            ignore_first: Ignore first N tokens in mean
            
        Returns:
            Tensor of shape [n_layers, n_heads]
        """
        entropy = EntropyAnalysis.token_entropy(capture, ignore_first=ignore_first)
        # Use nanmean to ignore NaN values from ignore_first
        return torch.nanmean(entropy, dim=-1)
    
    @staticmethod
    def delta_entropy(
        capture_treatment: AttentionCapture,
        capture_control: AttentionCapture,
        method: Literal['mean', 'final'] = 'mean',
        ignore_first: int = 2
    ) -> torch.Tensor:
        """
        Difference in entropy between two conditions.
        
        For sycophancy analysis:
            - treatment = sycophantic prompt (with wrong answer hint)
            - control = neutral prompt (no hint)
        
        Args:
            capture_treatment: Capture under treatment condition
            capture_control: Capture under control condition
            method:
                'mean': Compare mean entropy across positions
                'final': Compare entropy at final token only
            ignore_first: Ignore first N tokens
            
        Returns:
            Tensor of shape [n_layers, n_heads]
            
        Formula:
            ΔH = aggregation(H_treatment) - aggregation(H_control)
            
        Interpretation:
            - ΔH > 0 → Treatment causes more diffuse attention
            - ΔH < 0 → Treatment causes more focused attention
        """
        entropy_treat = EntropyAnalysis.token_entropy(
            capture_treatment, ignore_first=ignore_first
        )
        entropy_ctrl = EntropyAnalysis.token_entropy(
            capture_control, ignore_first=ignore_first
        )
        
        if method == 'mean':
            # Mean entropy per head
            treat_agg = torch.nanmean(entropy_treat, dim=-1)  # [L, H]
            ctrl_agg = torch.nanmean(entropy_ctrl, dim=-1)    # [L, H]
        elif method == 'final':
            # Entropy at final token only
            treat_agg = entropy_treat[..., -1]  # [L, H]
            ctrl_agg = entropy_ctrl[..., -1]    # [L, H]
        else:
            raise ValueError(f"method must be 'mean' or 'final', got '{method}'")
        
        return treat_agg - ctrl_agg
    
    @staticmethod
    def entropy_std(
        capture: AttentionCapture,
        ignore_first: int = 2
    ) -> torch.Tensor:
        """
        Standard deviation of entropy per (layer, head).
        
        Returns:
            Tensor of shape [n_layers, n_heads]
        """
        entropy = EntropyAnalysis.token_entropy(capture, ignore_first=ignore_first)
        
        # Manual nanstd (torch doesn't have built-in)
        mean = torch.nanmean(entropy, dim=-1, keepdim=True)
        valid_mask = ~torch.isnan(entropy)
        diff_sq = torch.where(valid_mask, (entropy - mean) ** 2, torch.zeros_like(entropy))
        count = valid_mask.sum(dim=-1).float()
        var = diff_sq.sum(dim=-1) / count.clamp(min=1)
        
        return torch.sqrt(var)
