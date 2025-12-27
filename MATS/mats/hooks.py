"""
Hook System for MATS 10.0
==========================

Provides hooks for Q/K scaling interventions at specific heads.

TransformerLens Hook Names:
- hook_rot_q: Post-RoPE query vectors [batch, pos, head, d_head]
- hook_rot_k: Post-RoPE key vectors [batch, pos, kv_head, d_head]
- pattern: Attention probabilities [batch, head, query_pos, key_pos]

GQA Considerations:
- Qwen2.5-7B has 28 Q-heads but only 4 KV-heads
- Q-heads 0-6 share KV-head 0, Q-heads 7-13 share KV-head 1, etc.
- When scaling K, we map Q-head to corresponding KV-head via integer division
"""

from typing import Callable, List, Tuple, Optional
import torch

# GQA configuration for Qwen2.5-7B
GQA_RATIO = 7  # 28 Q-heads / 4 KV-heads


def get_spectra_hook(
    alpha: float,
    head_index: int,
    is_key: bool = False,
) -> Callable:
    """
    Create a hook that scales Q or K by sqrt(alpha) for a specific head.
    
    The Spectra hypothesis: scaling attention by alpha is equivalent to
    scaling Q and K each by sqrt(alpha), due to the softmax temperature effect.
    
    Args:
        alpha: Scaling factor (>1 = sharpen, <1 = flatten)
        head_index: Q-head index to scale (0-27 for Qwen2.5-7B)
        is_key: If True, map Q-head to corresponding KV-head for GQA
        
    Returns:
        Hook function compatible with TransformerLens
        
    Example:
        # Sharpen head 5 in layer 15
        model.add_hook("blocks.15.attn.hook_rot_q", get_spectra_hook(1.5, 5))
        model.add_hook("blocks.15.attn.hook_rot_k", get_spectra_hook(1.5, 5, is_key=True))
    """
    scale_factor = alpha ** 0.5
    
    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        """
        Scale specific head's vectors.
        
        Args:
            value: [batch, pos, n_heads, d_head] for Q
                   [batch, pos, n_kv_heads, d_head] for K
            hook: TransformerLens hook object
            
        Returns:
            Modified value tensor
        """
        if is_key:
            # Map Q-head index to KV-head index for GQA
            kv_head_index = head_index // GQA_RATIO
            value[:, :, kv_head_index, :] = value[:, :, kv_head_index, :] * scale_factor
        else:
            # Direct Q-head scaling
            value[:, :, head_index, :] = value[:, :, head_index, :] * scale_factor
        
        return value
    
    return hook_fn


def add_scaling_hooks(
    model,
    layer: int,
    head: int,
    alpha: float,
) -> None:
    """
    Add Q and K scaling hooks to a specific layer/head.
    
    This is the primary intervention for the Spectra hypothesis:
    - alpha > 1: Sharpen attention (for Logic Head restoration)
    - alpha < 1: Flatten attention (for Sycophancy Head jamming)
    
    Args:
        model: ModelWrapper instance
        layer: Layer index (0-27)
        head: Q-head index (0-27)
        alpha: Scaling factor
        
    Example:
        # Sharpen layer 23, head 5 with alpha=1.5
        add_scaling_hooks(model, layer=23, head=5, alpha=1.5)
    """
    # Q hook
    q_hook_name = f"blocks.{layer}.attn.hook_rot_q"
    model.add_hook(q_hook_name, get_spectra_hook(alpha, head, is_key=False))
    
    # K hook (GQA-aware)
    k_hook_name = f"blocks.{layer}.attn.hook_rot_k"
    model.add_hook(k_hook_name, get_spectra_hook(alpha, head, is_key=True))


def add_multi_head_scaling_hooks(
    model,
    heads: List[Tuple[int, int]],
    alpha: float,
) -> None:
    """
    Add scaling hooks to multiple layer/head pairs.
    
    Args:
        model: ModelWrapper instance
        heads: List of (layer, head) tuples
        alpha: Scaling factor (applied to all heads)
        
    Example:
        # Sharpen multiple Logic Heads
        add_multi_head_scaling_hooks(model, [(22, 5), (23, 7), (24, 3)], alpha=1.5)
    """
    for layer, head in heads:
        add_scaling_hooks(model, layer, head, alpha)


def reset_hooks(model) -> None:
    """Remove all hooks from model."""
    model.reset_hooks()


def get_attention_pattern_hook(storage: dict, layer: int) -> Callable:
    """
    Create a hook to capture attention patterns.
    
    Useful for entropy computation when not using run_with_cache.
    
    Args:
        storage: Dictionary to store patterns
        layer: Layer index
        
    Returns:
        Hook function
    """
    def hook_fn(pattern: torch.Tensor, hook) -> torch.Tensor:
        """
        Capture attention pattern.
        
        Args:
            pattern: [batch, n_heads, query_pos, key_pos]
        """
        storage[layer] = pattern.detach().clone()
        return pattern
    
    return hook_fn


def create_hook_name(layer: int, hook_type: str) -> str:
    """
    Generate TransformerLens hook name.
    
    Args:
        layer: Layer index
        hook_type: One of "rot_q", "rot_k", "pattern"
        
    Returns:
        Full hook name string
    """
    if hook_type == "pattern":
        return f"blocks.{layer}.attn.hook_pattern"
    elif hook_type == "rot_q":
        return f"blocks.{layer}.attn.hook_rot_q"
    elif hook_type == "rot_k":
        return f"blocks.{layer}.attn.hook_rot_k"
    else:
        raise ValueError(f"Unknown hook type: {hook_type}")
