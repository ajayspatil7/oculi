"""
Spectra Hooks Module
====================

Unified hook system for capturing QKV activations across model architectures.
Adapted from src/hooks.py with multi-model support.
"""

from typing import Dict, List, Tuple, Optional, Callable
import torch
import numpy as np

from .models import ModelAdapter


class UnifiedHooks:
    """
    Hook manager for capturing hidden states and computing QKV.
    
    Works with any model via ModelAdapter.
    """
    
    def __init__(self, adapter: ModelAdapter):
        self.adapter = adapter
        self.model = adapter.model
        self.info = adapter.info
        
        self._hidden_states: Dict[int, torch.Tensor] = {}
        self._hooks: List = []
    
    def _create_hook(self, layer_idx: int) -> Callable:
        """Create hook function for a layer."""
        def hook_fn(module, input, output):
            self._hidden_states[layer_idx] = output.detach()
        return hook_fn
    
    def register(self) -> "UnifiedHooks":
        """Register hooks on all layers."""
        self.remove()
        
        for layer_idx in range(self.info.n_layers):
            layernorm = self.adapter.get_input_layernorm(layer_idx)
            hook = layernorm.register_forward_hook(self._create_hook(layer_idx))
            self._hooks.append(hook)
        
        return self
    
    def remove(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._hidden_states = {}
    
    def get_hidden_states(self, layer_idx: int) -> torch.Tensor:
        """Get captured hidden states for a layer."""
        return self._hidden_states.get(layer_idx)
    
    def compute_qkv(
        self,
        layer_idx: int,
        hidden_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Q, K, V tensors for a layer.
        
        Returns:
            Q: [batch, n_heads, seq, head_dim]
            K: [batch, n_kv_heads, seq, head_dim]
            V: [batch, n_kv_heads, seq, head_dim]
        """
        if hidden_states is None:
            hidden_states = self.get_hidden_states(layer_idx)
        
        attn = self.adapter.get_attention(layer_idx)
        batch_size, seq_len, _ = hidden_states.shape
        
        with torch.no_grad():
            Q = attn.q_proj(hidden_states)
            K = attn.k_proj(hidden_states)
            V = attn.v_proj(hidden_states)
            
            # Reshape Q
            Q = Q.view(batch_size, seq_len, self.info.n_heads, self.info.head_dim)
            Q = Q.transpose(1, 2)  # [batch, heads, seq, dim]
            
            # Reshape K, V
            K = K.view(batch_size, seq_len, self.info.n_kv_heads, self.info.head_dim)
            K = K.transpose(1, 2)
            
            V = V.view(batch_size, seq_len, self.info.n_kv_heads, self.info.head_dim)
            V = V.transpose(1, 2)
        
        return Q, K, V
    
    def compute_qkv_scaled(
        self,
        layer_idx: int,
        target_head: int,
        q_scale: float = 1.0,
        k_scale: float = 1.0,
        hidden_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Q, K, V with optional scaling at target head.
        
        V is NEVER modified.
        
        Args:
            layer_idx: Layer to compute for
            target_head: Head to scale
            q_scale: Scale factor for Q
            k_scale: Scale factor for K
            hidden_states: Optional pre-captured states
            
        Returns:
            Q, K, V tensors
        """
        Q, K, V = self.compute_qkv(layer_idx, hidden_states)
        
        # Clone to avoid in-place modification issues
        Q = Q.clone()
        K = K.clone()
        
        # Apply scaling
        if q_scale != 1.0:
            Q[:, target_head, :, :] = Q[:, target_head, :, :] * q_scale
        
        if k_scale != 1.0:
            kv_head_idx = target_head % self.info.n_kv_heads
            K[:, kv_head_idx, :, :] = K[:, kv_head_idx, :, :] * k_scale
        
        return Q, K, V
    
    def compute_attention_probs(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        causal: bool = True
    ) -> torch.Tensor:
        """
        Compute attention probabilities.
        
        Handles GQA by expanding K as needed.
        
        Args:
            Q: Query tensor [batch, heads, seq, dim]
            K: Key tensor [batch, kv_heads, seq, dim]
            causal: Apply causal mask
            
        Returns:
            Attention probabilities [batch, heads, seq, seq]
        """
        batch, n_heads, seq_len, head_dim = Q.shape
        _, n_kv_heads, _, _ = K.shape
        
        # Expand K for GQA
        if n_heads != n_kv_heads:
            n_rep = n_heads // n_kv_heads
            K = K.repeat_interleave(n_rep, dim=1)
        
        # Scaled dot-product
        scale = head_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Causal mask
        if causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(mask, float('-inf'))
        
        return torch.softmax(scores, dim=-1)
    
    def __enter__(self):
        """Context manager support."""
        self.register()
        return self
    
    def __exit__(self, *args):
        """Context manager support."""
        self.remove()
