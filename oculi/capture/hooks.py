"""
Attention Hook System
=====================

Low-level hook registration and management for capturing attention data.
This is the internal machinery used by ModelAdapter implementations.
"""

from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field
import torch


@dataclass
class HookHandle:
    """
    Handle for a registered hook.
    
    Used to track and remove hooks after capture.
    """
    id: str
    layer: int
    component: str
    handle: Any  # torch.utils.hooks.RemovableHandle
    
    def remove(self):
        """Remove this hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@dataclass
class CapturedData:
    """
    Mutable container for data captured during forward pass.
    
    This is internal to AttentionHook and gets converted to
    immutable AttentionCapture after the forward pass.
    """
    queries: Dict[int, torch.Tensor] = field(default_factory=dict)
    keys: Dict[int, torch.Tensor] = field(default_factory=dict)
    values: Dict[int, torch.Tensor] = field(default_factory=dict)
    patterns: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    def clear(self):
        """Clear all captured data."""
        self.queries.clear()
        self.keys.clear()
        self.values.clear()
        self.patterns.clear()


class AttentionHook:
    """
    Manager for attention hook registration and data capture.
    
    Handles:
        - Forward hook registration at correct module points
        - Data capture with proper reshaping
        - Automatic detach and CPU offload
        - Token filtering for memory efficiency
        - Cleanup after capture
        
    Example:
        >>> hook = AttentionHook(model)
        >>> hook.register_capture_hooks(layers=[0, 1, 2])
        >>> output = model(input_ids)
        >>> capture = hook.collect_capture(config)
        >>> hook.remove_all_hooks()
    """
    
    def __init__(
        self,
        model,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int
    ):
        """
        Initialize hook manager.
        
        Args:
            model: The transformer model
            n_heads: Number of query heads
            n_kv_heads: Number of KV heads (for GQA)
            head_dim: Dimension per head
        """
        self.model = model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.gqa_ratio = n_heads // n_kv_heads
        
        self._hooks: List[HookHandle] = []
        self._captured = CapturedData()
        self._hook_counter = 0
    
    def register_capture_hooks(
        self,
        layers: List[int],
        capture_q: bool = True,
        capture_k: bool = True,
        capture_v: bool = True,
        token_filter: Optional[Callable[[int], bool]] = None
    ) -> None:
        """
        Register hooks for capturing Q/K/V at specified layers.
        
        Args:
            layers: Layer indices to capture
            capture_q: Whether to capture queries
            capture_k: Whether to capture keys
            capture_v: Whether to capture values
            token_filter: Optional function (token_idx) -> bool for filtering
        """
        for layer_idx in layers:
            layer_module = self._get_layer_module(layer_idx)
            
            if capture_q:
                self._register_projection_hook(
                    layer_idx, layer_module, 'q', token_filter
                )
            
            if capture_k:
                self._register_projection_hook(
                    layer_idx, layer_module, 'k', token_filter
                )
            
            if capture_v:
                self._register_projection_hook(
                    layer_idx, layer_module, 'v', token_filter
                )
    
    def _get_layer_module(self, layer_idx: int):
        """Get the layer module - override in model-specific subclasses."""
        # Default assumes HuggingFace LLaMA-style structure
        return self.model.model.layers[layer_idx]
    
    def _get_projection_module(self, layer_module, component: str):
        """Get Q/K/V projection module from layer."""
        if component == 'q':
            return layer_module.self_attn.q_proj
        elif component == 'k':
            return layer_module.self_attn.k_proj
        elif component == 'v':
            return layer_module.self_attn.v_proj
        else:
            raise ValueError(f"Unknown component: {component}")
    
    def _register_projection_hook(
        self,
        layer_idx: int,
        layer_module,
        component: str,
        token_filter: Optional[Callable] = None
    ) -> None:
        """Register a hook on a projection output."""
        proj_module = self._get_projection_module(layer_module, component)
        
        # Determine number of heads for this component
        if component == 'q':
            n_heads = self.n_heads
            storage = self._captured.queries
        else:
            n_heads = self.n_kv_heads
            storage = self._captured.keys if component == 'k' else self._captured.values
        
        def hook_fn(module, input, output):
            # output: [batch, seq, n_heads * head_dim]
            batch, seq, hidden = output.shape
            
            # Reshape to [batch, seq, n_heads, head_dim]
            reshaped = output.view(batch, seq, n_heads, self.head_dim)
            
            # Apply token filter if provided
            if token_filter is not None:
                mask = torch.tensor(
                    [token_filter(i) for i in range(seq)],
                    device=output.device
                )
                reshaped = reshaped[:, mask, :, :]
            
            # Detach and move to CPU
            captured = reshaped.detach().cpu()
            
            # Store (concatenate if already exists for batched processing)
            storage[layer_idx] = captured
        
        handle = proj_module.register_forward_hook(hook_fn)
        
        hook_handle = HookHandle(
            id=f"capture_{self._hook_counter}",
            layer=layer_idx,
            component=component,
            handle=handle
        )
        self._hooks.append(hook_handle)
        self._hook_counter += 1
    
    def collect_and_clear(self) -> CapturedData:
        """
        Collect captured data and clear internal storage.
        
        Returns:
            CapturedData with all captured tensors
        """
        data = self._captured
        self._captured = CapturedData()
        return data
    
    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit - cleanup hooks."""
        self.remove_all_hooks()
        self._captured.clear()


def assemble_capture_tensor(
    data: Dict[int, torch.Tensor],
    layers: List[int],
    n_heads: int,
    n_tokens: int,
    head_dim: int
) -> torch.Tensor:
    """
    Assemble captured layer data into a single tensor.
    
    Args:
        data: Dict mapping layer_idx -> [batch, seq, heads, dim]
        layers: List of layer indices (defines output order)
        n_heads: Number of heads
        n_tokens: Sequence length
        head_dim: Head dimension
        
    Returns:
        Tensor of shape [n_layers, n_heads, n_tokens, head_dim]
    """
    result = torch.zeros(len(layers), n_heads, n_tokens, head_dim)
    
    for i, layer_idx in enumerate(layers):
        if layer_idx in data:
            # data[layer_idx]: [batch, seq, heads, dim]
            # Take first batch, transpose to [heads, seq, dim]
            tensor = data[layer_idx][0]  # [seq, heads, dim]
            result[i] = tensor.permute(1, 0, 2)  # [heads, seq, dim]
    
    return result
