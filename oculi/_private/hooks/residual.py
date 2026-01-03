"""
Residual Stream Hooks
=====================

PyTorch hook implementations for capturing residual stream activations.

Hook Locations (LLaMA):
    - pre_attn: Block input (forward_pre hook on transformer block)
    - post_attn: After self_attn (forward hook on self_attn module)
    - pre_mlp: After post_attention_layernorm (forward hook on layernorm)
    - post_mlp: After mlp (forward hook on mlp module)
"""

from typing import Callable, Dict, List, Optional, Any
import torch
from torch.utils.hooks import RemovableHandle


def create_pre_attn_hook(
    storage: Dict[int, torch.Tensor],
    layer_idx: int,
    dtype: torch.dtype = torch.float32
) -> Callable:
    """
    Create a pre-hook that captures block input (pre-attention).
    
    This is a forward_pre hook on the transformer block.
    Input is the residual stream before input_layernorm.
    
    Args:
        storage: Dict to store captured tensors {layer_idx: tensor}
        layer_idx: Layer index for storage key
        dtype: Storage dtype for conversion
        
    Returns:
        Hook function for register_forward_pre_hook
    """
    def hook(module, args):
        # args[0] is hidden_states: [batch, seq, hidden]
        hidden_states = args[0]
        storage[layer_idx] = hidden_states.detach().to(dtype).cpu()
        # Must return None for forward_pre hooks (or modified args)
        return None
    
    return hook


def create_post_attn_hook(
    storage: Dict[int, torch.Tensor],
    layer_idx: int,
    dtype: torch.dtype = torch.float32
) -> Callable:
    """
    Create a hook that captures attention output (post-attention).
    
    This is a forward hook on the self_attn module.
    Output is the attention contribution before residual add.
    
    Args:
        storage: Dict to store captured tensors {layer_idx: tensor}
        layer_idx: Layer index for storage key
        dtype: Storage dtype for conversion
        
    Returns:
        Hook function for register_forward_hook
    """
    def hook(module, input, output):
        # output is (attn_output, attn_weights) or just attn_output
        # attn_output: [batch, seq, hidden]
        if isinstance(output, tuple):
            attn_output = output[0]
        else:
            attn_output = output
        storage[layer_idx] = attn_output.detach().to(dtype).cpu()
    
    return hook


def create_pre_mlp_hook(
    storage: Dict[int, torch.Tensor],
    layer_idx: int,
    dtype: torch.dtype = torch.float32
) -> Callable:
    """
    Create a hook that captures post_attention_layernorm output (pre-MLP).
    
    This is a forward hook on the post_attention_layernorm module.
    Output is the normalized residual before MLP processing.
    
    Args:
        storage: Dict to store captured tensors {layer_idx: tensor}
        layer_idx: Layer index for storage key
        dtype: Storage dtype for conversion
        
    Returns:
        Hook function for register_forward_hook
    """
    def hook(module, input, output):
        # output: [batch, seq, hidden]
        storage[layer_idx] = output.detach().to(dtype).cpu()
    
    return hook


def create_post_mlp_hook(
    storage: Dict[int, torch.Tensor],
    layer_idx: int,
    dtype: torch.dtype = torch.float32
) -> Callable:
    """
    Create a hook that captures MLP output (post-MLP).
    
    This is a forward hook on the mlp module.
    Output is the MLP contribution before final residual add.
    
    Args:
        storage: Dict to store captured tensors {layer_idx: tensor}
        layer_idx: Layer index for storage key
        dtype: Storage dtype for conversion
        
    Returns:
        Hook function for register_forward_hook
    """
    def hook(module, input, output):
        # output: [batch, seq, hidden]
        storage[layer_idx] = output.detach().to(dtype).cpu()
    
    return hook


def register_residual_hooks(
    model: Any,
    layers: List[int],
    storage: Dict[str, Dict[int, torch.Tensor]],
    capture_pre_attn: bool = True,
    capture_post_attn: bool = True,
    capture_pre_mlp: bool = True,
    capture_post_mlp: bool = True,
    dtype: torch.dtype = torch.float32
) -> List[RemovableHandle]:
    """
    Register all residual stream hooks on a model.
    
    Args:
        model: The LLaMA model (LlamaForCausalLM or similar)
        layers: Which layers to register hooks on
        storage: Dict with keys 'pre_attn', 'post_attn', 'pre_mlp', 'post_mlp'
                 Each value is a dict to store {layer_idx: tensor}
        capture_pre_attn: Whether to capture pre-attention
        capture_post_attn: Whether to capture post-attention
        capture_pre_mlp: Whether to capture pre-MLP
        capture_post_mlp: Whether to capture post-MLP
        dtype: Storage dtype for conversion
        
    Returns:
        List of RemovableHandle for cleanup
        
    Example:
        >>> storage = {'pre_attn': {}, 'post_attn': {}, 'pre_mlp': {}, 'post_mlp': {}}
        >>> handles = register_residual_hooks(model.model, [0, 1, 2], storage)
        >>> # Run forward pass
        >>> for h in handles: h.remove()
    """
    handles = []
    
    # Get the layers module list
    # Support both model.model.layers (full LlamaForCausalLM) and model.layers (LlamaModel)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model_layers = model.model.layers
    elif hasattr(model, 'layers'):
        model_layers = model.layers
    else:
        raise ValueError("Cannot find layers in model. Expected model.model.layers or model.layers")
    
    for layer_idx in layers:
        layer = model_layers[layer_idx]
        
        if capture_pre_attn:
            hook = create_pre_attn_hook(storage['pre_attn'], layer_idx, dtype)
            h = layer.register_forward_pre_hook(hook)
            handles.append(h)
        
        if capture_post_attn:
            hook = create_post_attn_hook(storage['post_attn'], layer_idx, dtype)
            h = layer.self_attn.register_forward_hook(hook)
            handles.append(h)
        
        if capture_pre_mlp:
            hook = create_pre_mlp_hook(storage['pre_mlp'], layer_idx, dtype)
            h = layer.post_attention_layernorm.register_forward_hook(hook)
            handles.append(h)
        
        if capture_post_mlp:
            hook = create_post_mlp_hook(storage['post_mlp'], layer_idx, dtype)
            h = layer.mlp.register_forward_hook(hook)
            handles.append(h)
    
    return handles


def assemble_residual_capture(
    storage: Dict[str, Dict[int, torch.Tensor]],
    layers: List[int],
    n_tokens: int,
    hidden_dim: int,
    model_name: str = ""
):
    """
    Assemble captured data into ResidualCapture structure.
    
    Args:
        storage: Dict with captured data from register_residual_hooks
        layers: List of captured layer indices
        n_tokens: Number of tokens in sequence
        hidden_dim: Model hidden dimension
        model_name: Name of the model
        
    Returns:
        ResidualCapture instance
    """
    from oculi.capture.structures import ResidualCapture
    
    def stack_storage(key: str) -> Optional[torch.Tensor]:
        """Stack tensors from storage into [L, T, H] tensor."""
        store = storage.get(key, {})
        if not store:
            return None
        
        # Stack in layer order
        tensors = []
        for layer_idx in layers:
            if layer_idx in store:
                # Remove batch dimension: [1, T, H] -> [T, H]
                t = store[layer_idx]
                if t.dim() == 3 and t.shape[0] == 1:
                    t = t.squeeze(0)
                tensors.append(t)
        
        if not tensors:
            return None
        
        return torch.stack(tensors, dim=0)  # [L, T, H]
    
    return ResidualCapture(
        pre_attn=stack_storage('pre_attn'),
        post_attn=stack_storage('post_attn'),
        pre_mlp=stack_storage('pre_mlp'),
        post_mlp=stack_storage('post_mlp'),
        n_layers=len(layers),
        n_tokens=n_tokens,
        hidden_dim=hidden_dim,
        model_name=model_name,
        captured_layers=tuple(layers),
    )
