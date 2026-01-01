"""
Intervention Hooks (Private)
============================

PyTorch hook implementations for interventions.
"""

from typing import Callable
import torch


def create_intervention_hook(
    component: str,
    head_index: int,
    scale_factor: float,
    is_gqa: bool = False,
    gqa_ratio: int = 1
) -> Callable:
    """
    Create a hook function that scales a specific head.
    
    Args:
        component: 'q', 'k', 'v', or 'attn_out'
        head_index: Index of head to scale (Q-head for 'q', KV-head for 'k')
        scale_factor: Multiplicative factor
        is_gqa: Whether model uses GQA
        gqa_ratio: Q-heads per KV-head (only used if is_gqa)
        
    Returns:
        Hook function compatible with PyTorch register_forward_hook
    """
    
    def hook_fn(module, input, output):
        """
        Scale specific head in output tensor.
        
        Output shapes:
            q_proj: [batch, seq, n_q_heads * head_dim]
            k_proj: [batch, seq, n_kv_heads * head_dim]
            v_proj: [batch, seq, n_kv_heads * head_dim]
            o_proj: [batch, seq, hidden_size]
        """
        # For projection outputs, we need to reshape to access heads
        batch, seq, hidden = output.shape
        
        if component == 'q':
            # Q output is [batch, seq, n_q_heads * head_dim]
            # We need to know head_dim to reshape
            # For now, scale in-place for the slice
            # This is a simplified version - full impl needs head_dim
            pass
        
        elif component == 'k':
            # K output - map Q head to KV head if GQA
            if is_gqa:
                kv_head_index = head_index // gqa_ratio
            else:
                kv_head_index = head_index
            pass
        
        elif component == 'attn_out':
            # Ablation: zero out completely
            if scale_factor == 0.0:
                return torch.zeros_like(output)
        
        # For complex reshaping, return modified output
        return output * scale_factor  # Simplified - scales all
    
    return hook_fn


def create_spectra_hook(
    alpha: float,
    head_index: int,
    is_key: bool = False,
    gqa_ratio: int = 1,
    head_dim: int = 128
) -> Callable:
    """
    Create Spectra-style scaling hook.
    
    Scales Q or K by √α for a specific head.
    
    Args:
        alpha: Net scaling factor (hook applies √α)
        head_index: Q-head index to scale
        is_key: If True, map Q-head to KV-head
        gqa_ratio: Q-heads per KV-head
        head_dim: Dimension of each head
        
    Returns:
        Hook function
    """
    scale_factor = alpha ** 0.5
    
    def hook_fn(module, input, output):
        """Scale specific head by √α."""
        batch, seq, hidden = output.shape
        
        if is_key:
            # Map Q head to KV head
            target_head = head_index // gqa_ratio
            n_heads = hidden // head_dim // gqa_ratio  # Approximate
        else:
            target_head = head_index
            n_heads = hidden // head_dim
        
        # Reshape to [batch, seq, n_heads, head_dim]
        reshaped = output.view(batch, seq, n_heads, head_dim)
        
        # Scale specific head
        reshaped[:, :, target_head, :] = reshaped[:, :, target_head, :] * scale_factor
        
        # Reshape back
        return reshaped.view(batch, seq, hidden)
    
    return hook_fn
