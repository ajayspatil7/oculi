"""
LLaMA Attention Anatomy
=======================

This file documents exactly how LLaMA implements attention.
Open this file when:
    - Numbers look weird
    - Entropy jumps unexpectedly
    - Interventions behave oddly

Module Structure (LLaMA-3-8B):
    model.model.layers[i].self_attn.q_proj  → [hidden, n_heads * head_dim]
    model.model.layers[i].self_attn.k_proj  → [hidden, n_kv_heads * head_dim]
    model.model.layers[i].self_attn.v_proj  → [hidden, n_kv_heads * head_dim]
    model.model.layers[i].self_attn.o_proj  → [n_heads * head_dim, hidden]

GQA (Grouped Query Attention):
    LLaMA-3-8B uses 32 query heads but only 8 KV heads.
    Each KV head is shared by 4 query heads (ratio 4:1).
    
    Before attention computation:
        K, V are expanded: [B, 8, T, D] → [B, 32, T, D]
        using repeat_interleave(4, dim=1)

RoPE (Rotary Position Embeddings):
    Applied AFTER projection, BEFORE attention computation.
    Interleaved format: first half and second half are rotated differently.
    
    cos, sin are precomputed for position indices.
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
"""

from typing import Optional, Tuple, List, Callable
import torch
import torch.nn as nn


# =============================================================================
# Q/K/V EXTRACTION
# =============================================================================

def get_q_proj(model, layer: int) -> nn.Linear:
    """
    Get the Query projection module for a layer.
    
    Args:
        model: The LLaMA model (LlamaForCausalLM)
        layer: Layer index
        
    Returns:
        The q_proj Linear module
        
    Path: model.model.layers[layer].self_attn.q_proj
    Shape: [hidden_size, num_heads * head_dim]
    """
    return model.model.layers[layer].self_attn.q_proj


def get_k_proj(model, layer: int) -> nn.Linear:
    """
    Get the Key projection module for a layer.
    
    Path: model.model.layers[layer].self_attn.k_proj
    Shape: [hidden_size, num_kv_heads * head_dim]
    
    Note: For GQA, this is smaller than q_proj!
          LLaMA-3-8B: 4096 → 1024 (8 heads × 128 dim)
    """
    return model.model.layers[layer].self_attn.k_proj


def get_v_proj(model, layer: int) -> nn.Linear:
    """
    Get the Value projection module for a layer.
    
    Path: model.model.layers[layer].self_attn.v_proj
    Shape: [hidden_size, num_kv_heads * head_dim]
    """
    return model.model.layers[layer].self_attn.v_proj


def get_o_proj(model, layer: int) -> nn.Linear:
    """
    Get the Output projection module for a layer.
    
    Path: model.model.layers[layer].self_attn.o_proj
    Shape: [num_heads * head_dim, hidden_size]
    
    This projects attention output back to hidden dimension.
    """
    return model.model.layers[layer].self_attn.o_proj


def get_attention_module(model, layer: int):
    """
    Get the full attention module for a layer.
    
    Path: model.model.layers[layer].self_attn
    Type: LlamaAttention or LlamaSdpaAttention or LlamaFlashAttention2
    """
    return model.model.layers[layer].self_attn


# =============================================================================
# RESHAPING FOR MULTI-HEAD ATTENTION
# =============================================================================

def reshape_for_heads(
    tensor: torch.Tensor,
    num_heads: int,
    head_dim: int
) -> torch.Tensor:
    """
    Reshape projection output for multi-head attention.
    
    Args:
        tensor: [batch, seq, num_heads * head_dim]
        num_heads: Number of attention heads
        head_dim: Dimension per head
        
    Returns:
        [batch, num_heads, seq, head_dim]
        
    Example:
        >>> q = q_proj(hidden)  # [1, 512, 4096]
        >>> q = reshape_for_heads(q, 32, 128)  # [1, 32, 512, 128]
    """
    batch, seq, _ = tensor.shape
    return tensor.view(batch, seq, num_heads, head_dim).transpose(1, 2)


def flatten_heads(tensor: torch.Tensor) -> torch.Tensor:
    """
    Flatten multi-head tensor back to projection format.
    
    Args:
        tensor: [batch, num_heads, seq, head_dim]
        
    Returns:
        [batch, seq, num_heads * head_dim]
    """
    batch, num_heads, seq, head_dim = tensor.shape
    return tensor.transpose(1, 2).contiguous().view(batch, seq, -1)


# =============================================================================
# GQA (GROUPED QUERY ATTENTION)
# =============================================================================

def expand_kv_for_gqa(
    kv: torch.Tensor,
    n_query_heads: int,
    n_kv_heads: int
) -> torch.Tensor:
    """
    Expand KV heads for Grouped Query Attention.
    
    In GQA, we have fewer KV heads than query heads.
    Each KV head is shared by (n_query_heads // n_kv_heads) query heads.
    
    Args:
        kv: [batch, n_kv_heads, seq, head_dim]
        n_query_heads: Number of query heads (e.g., 32)
        n_kv_heads: Number of KV heads (e.g., 8)
        
    Returns:
        [batch, n_query_heads, seq, head_dim]
        
    Example (LLaMA-3-8B):
        >>> k = k_proj(hidden)  # [1, 8, 512, 128]
        >>> k = expand_kv_for_gqa(k, 32, 8)  # [1, 32, 512, 128]
        # Each of the 8 KV heads is repeated 4 times
    """
    if n_query_heads == n_kv_heads:
        return kv  # MHA, no expansion needed
    
    ratio = n_query_heads // n_kv_heads
    return kv.repeat_interleave(ratio, dim=1)


def get_gqa_group(query_head: int, n_query_heads: int, n_kv_heads: int) -> int:
    """
    Get the KV head index for a given query head.
    
    Args:
        query_head: Query head index (0 to n_query_heads-1)
        n_query_heads: Total query heads
        n_kv_heads: Total KV heads
        
    Returns:
        KV head index that this query head attends to
        
    Example (LLaMA-3-8B with 32 Q heads, 8 KV heads):
        >>> get_gqa_group(0, 32, 8)   # 0  (Q heads 0-3 use KV head 0)
        >>> get_gqa_group(4, 32, 8)   # 1  (Q heads 4-7 use KV head 1)
        >>> get_gqa_group(31, 32, 8)  # 7  (Q heads 28-31 use KV head 7)
    """
    ratio = n_query_heads // n_kv_heads
    return query_head // ratio


# =============================================================================
# ROTARY POSITION EMBEDDINGS (RoPE)
# =============================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half of the hidden dimensions.
    
    RoPE rotates pairs of dimensions together.
    This function prepares the tensor for the rotation formula.
    
    Args:
        x: [..., head_dim]
        
    Returns:
        [..., head_dim] with first and second halves swapped and negated
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to Q and K.
    
    This is the standard RoPE formula:
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
    
    Args:
        q: Query tensor [batch, heads, seq, head_dim]
        k: Key tensor [batch, heads, seq, head_dim]
        cos: Cosine embeddings [seq, head_dim] or [batch, seq, head_dim]
        sin: Sine embeddings [seq, head_dim] or [batch, seq, head_dim]
        position_ids: Optional position indices
        
    Returns:
        (q_rotated, k_rotated) with same shapes as inputs
        
    Note:
        RoPE is applied AFTER projection, BEFORE attention.
        This encodes relative position information that affects
        how tokens attend to each other based on distance.
    """
    # Expand cos/sin to match tensor dimensions
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


# =============================================================================
# ATTENTION COMPUTATION
# =============================================================================

def compute_attention_logits(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    Compute raw attention logits (pre-softmax).
    
    This is the QK^T / sqrt(d_k) computation.
    
    Args:
        query: [batch, heads, seq_q, head_dim]
        key: [batch, heads, seq_k, head_dim]
        scale: Optional scaling factor (default: 1/sqrt(head_dim))
        
    Returns:
        [batch, heads, seq_q, seq_k] attention logits
        
    Note:
        These are PRE-CAUSAL-MASK, PRE-SOFTMAX values.
        To get attention patterns, apply causal mask then softmax.
    """
    if scale is None:
        scale = query.shape[-1] ** -0.5
    
    return torch.matmul(query, key.transpose(-2, -1)) * scale


def apply_causal_mask(
    logits: torch.Tensor,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Apply causal attention mask.
    
    Sets future positions to -inf so they get zero weight after softmax.
    
    Args:
        logits: [batch, heads, seq, seq] attention logits
        dtype: Output dtype
        
    Returns:
        Masked logits with -inf for future positions
    """
    seq_len = logits.shape[-1]
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=logits.device, dtype=torch.bool),
        diagonal=1
    )
    return logits.masked_fill(mask, float('-inf'))


def compute_attention_pattern(
    logits: torch.Tensor,
    apply_causal: bool = True
) -> torch.Tensor:
    """
    Compute attention pattern (probabilities) from logits.
    
    Args:
        logits: [batch, heads, seq, seq] raw attention logits
        apply_causal: Whether to apply causal mask first
        
    Returns:
        [batch, heads, seq, seq] attention probabilities (sum to 1 per row)
    """
    if apply_causal:
        logits = apply_causal_mask(logits)
    
    return torch.softmax(logits, dim=-1)


# =============================================================================
# HOOK UTILITIES
# =============================================================================

def create_capture_hook(
    storage: dict,
    layer_idx: int,
    component: str,
    n_heads: int,
    head_dim: int
) -> Callable:
    """
    Create a forward hook that captures projection output.
    
    Args:
        storage: Dict to store captured tensors
        layer_idx: Layer index for storage key
        component: 'q', 'k', or 'v' for identification
        n_heads: Number of heads for reshaping
        head_dim: Head dimension for reshaping
        
    Returns:
        Hook function compatible with register_forward_hook
    """
    def hook(module, input, output):
        batch, seq, hidden = output.shape
        reshaped = output.view(batch, seq, n_heads, head_dim)
        storage[layer_idx] = reshaped.detach()  # Keep on original device (CUDA/MPS/CPU)

    return hook
