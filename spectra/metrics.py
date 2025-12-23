"""
Spectra Metrics Module
======================

Unified metrics computation for attention analysis.
Adapted from src/metrics.py with consistent interface.
"""

import torch
import numpy as np
from typing import Optional, Tuple


def compute_attention_entropy(
    attention_probs: torch.Tensor,
    ignore_first_n: int = 2,
    last_k: int = None,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute attention entropy for each query position.
    
    LOCKED DEFINITION (for reproducibility):
    =========================================
    H(t) = -Σ_{i=0}^{t} p(i|t) * log(p(i|t))
    
    Where:
        - t is the query position (token index)
        - p(i|t) is the attention weight from position t to position i
        - The sum is over all valid key positions [0, t] (causal mask)
    
    Entropy measures attention "diffuseness":
        - Low H → focused attention (sharp distribution)
        - High H → diffuse attention (uniform distribution)
    
    Args:
        attention_probs: [batch, heads, seq, seq] attention weights
        ignore_first_n: Ignore first N positions (BOS/padding artifacts)
        last_k: If set, only compute entropy for last K positions
        eps: Small constant for numerical stability
        
    Returns:
        Entropy tensor [batch, heads, seq]
    """
    # Clamp for numerical stability
    probs = torch.clamp(attention_probs, min=eps, max=1.0)
    
    # Compute entropy: H = -Σp*log(p)
    log_probs = torch.log(probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    # Mask first N positions (BOS/padding artifacts)
    if ignore_first_n > 0:
        entropy[:, :, :ignore_first_n] = float('nan')
    
    # If last_k specified, mask earlier positions
    if last_k is not None and last_k > 0:
        seq_len = entropy.shape[-1]
        start_pos = max(seq_len - last_k, ignore_first_n)
        entropy[:, :, :start_pos] = float('nan')
    
    return entropy


def compute_max_attention_weight(
    attention_probs: torch.Tensor
) -> torch.Tensor:
    """
    Compute maximum attention weight per query position.
    
    Args:
        attention_probs: [batch, heads, seq, seq]
        
    Returns:
        Max weights [batch, heads, seq]
    """
    return attention_probs.max(dim=-1).values


def compute_effective_attention_span(
    attention_probs: torch.Tensor,
    threshold: float = 0.9
) -> torch.Tensor:
    """
    Compute effective attention span (k_eff).
    
    k_eff = number of keys needed to capture `threshold` of attention mass.
    
    Args:
        attention_probs: [batch, heads, seq, seq]
        threshold: Cumulative probability threshold
        
    Returns:
        k_eff [batch, heads, seq]
    """
    # Sort attention weights descending
    sorted_probs, _ = torch.sort(attention_probs, dim=-1, descending=True)
    
    # Cumulative sum
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    
    # Find first index where cumsum >= threshold
    k_eff = (cumsum >= threshold).float().argmax(dim=-1) + 1
    
    return k_eff.float()


def compute_query_norms(
    Q: torch.Tensor
) -> torch.Tensor:
    """
    Compute L2 norm of query vectors.
    
    Args:
        Q: [batch, heads, seq, dim]
        
    Returns:
        Norms [batch, heads, seq]
    """
    return torch.norm(Q, p=2, dim=-1)


def aggregate_metrics(
    tensor: torch.Tensor,
    ignore_nan: bool = True
) -> Tuple[float, float]:
    """
    Aggregate metrics to mean and std.
    
    Args:
        tensor: Input tensor
        ignore_nan: Exclude NaN values
        
    Returns:
        (mean, std) tuple
    """
    arr = tensor.cpu().numpy().flatten()
    
    if ignore_nan:
        arr = arr[~np.isnan(arr)]
    
    return float(np.mean(arr)), float(np.std(arr))


def check_monotonicity(
    values: np.ndarray,
    direction: str = "increasing"
) -> bool:
    """
    Check if values are monotonic.
    
    Args:
        values: Array of values
        direction: "increasing" or "decreasing"
        
    Returns:
        True if monotonic
    """
    if direction == "increasing":
        return all(values[i] <= values[i+1] for i in range(len(values)-1))
    else:
        return all(values[i] >= values[i+1] for i in range(len(values)-1))


def compute_linear_gain(
    alphas: np.ndarray,
    values: np.ndarray
) -> Tuple[float, float]:
    """
    Compute gain (slope) via linear regression on log(alpha).
    
    Args:
        alphas: Scale factors
        values: Metric values
        
    Returns:
        (slope, r_squared)
    """
    from scipy import stats
    
    log_alphas = np.log(alphas)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_alphas, values)
    
    return slope, r_value ** 2


def aggregate_across_samples(
    sample_values: list,
    ignore_nan: bool = True
) -> Tuple[float, float, int]:
    """
    Aggregate metrics ACROSS multiple samples.
    
    This provides proper variance estimation for multi-sample experiments.
    
    Args:
        sample_values: List of scalar values (one per sample)
        ignore_nan: Exclude NaN values
        
    Returns:
        (mean, std, n_valid_samples)
    """
    arr = np.array(sample_values)
    
    if ignore_nan:
        arr = arr[~np.isnan(arr)]
    
    n_valid = len(arr)
    
    if n_valid == 0:
        return float('nan'), float('nan'), 0
    
    return float(np.mean(arr)), float(np.std(arr)), n_valid
