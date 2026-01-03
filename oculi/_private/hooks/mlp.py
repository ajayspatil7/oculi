"""
MLP Capture Hooks
=================

PyTorch hook implementations for capturing MLP internals.

LLaMA MLP Structure:
    gate = gate_proj(x)           # [B, T, I]
    up = up_proj(x)               # [B, T, I]
    hidden = silu(gate) * up      # [B, T, I] - "neuron activations"
    output = down_proj(hidden)    # [B, T, H]

Hook Locations:
    - gate_proj: model.model.layers[i].mlp.gate_proj (output hook)
    - up_proj: model.model.layers[i].mlp.up_proj (output hook)
    - post_activation: model.model.layers[i].mlp (custom extraction)
    - mlp_output: model.model.layers[i].mlp (output hook)
"""

from typing import Callable, Dict, List, Optional, Any
import torch
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle


def create_gate_proj_hook(
    storage: Dict[int, torch.Tensor],
    layer_idx: int,
    dtype: torch.dtype = torch.float32
) -> Callable:
    """
    Create a hook that captures gate_proj output (pre-activation).
    
    Args:
        storage: Dict to store captured tensors {layer_idx: tensor}
        layer_idx: Layer index for storage key
        dtype: Storage dtype for conversion
        
    Returns:
        Hook function for register_forward_hook
    """
    def hook(module, input, output):
        # output: [batch, seq, intermediate_dim]
        storage[layer_idx] = output.detach().to(dtype).cpu()
    
    return hook


def create_up_proj_hook(
    storage: Dict[int, torch.Tensor],
    layer_idx: int,
    dtype: torch.dtype = torch.float32
) -> Callable:
    """
    Create a hook that captures up_proj output.
    
    Args:
        storage: Dict to store captured tensors {layer_idx: tensor}
        layer_idx: Layer index for storage key
        dtype: Storage dtype for conversion
        
    Returns:
        Hook function for register_forward_hook
    """
    def hook(module, input, output):
        # output: [batch, seq, intermediate_dim]
        storage[layer_idx] = output.detach().to(dtype).cpu()
    
    return hook


def create_mlp_output_hook(
    storage: Dict[int, torch.Tensor],
    layer_idx: int,
    dtype: torch.dtype = torch.float32
) -> Callable:
    """
    Create a hook that captures final MLP output.
    
    This captures the output of the entire MLP module.
    
    Args:
        storage: Dict to store captured tensors {layer_idx: tensor}
        layer_idx: Layer index for storage key
        dtype: Storage dtype for conversion
        
    Returns:
        Hook function for register_forward_hook
    """
    def hook(module, input, output):
        # output: [batch, seq, hidden_dim]
        storage[layer_idx] = output.detach().to(dtype).cpu()
    
    return hook


def create_post_activation_hook(
    gate_storage: Dict[int, torch.Tensor],
    up_storage: Dict[int, torch.Tensor],
    post_activation_storage: Dict[int, torch.Tensor],
    layer_idx: int,
    dtype: torch.dtype = torch.float32
) -> Callable:
    """
    Create a hook that computes post-activation from gate and up projections.
    
    Since silu(gate) * up happens inside the MLP forward, we need to
    capture gate and up separately, then compute the post-activation.
    
    This hook is registered on the MLP module and triggers after forward.
    
    Args:
        gate_storage: Dict with gate_proj outputs
        up_storage: Dict with up_proj outputs
        post_activation_storage: Dict to store computed post-activations
        layer_idx: Layer index for storage key
        dtype: Storage dtype for conversion
        
    Returns:
        Hook function for register_forward_hook
    """
    def hook(module, input, output):
        # Compute post_activation from stored gate and up
        if layer_idx in gate_storage and layer_idx in up_storage:
            gate = gate_storage[layer_idx].float()
            up = up_storage[layer_idx].float()
            # silu(gate) * up
            post_activation = F.silu(gate) * up
            post_activation_storage[layer_idx] = post_activation.to(dtype).cpu()
    
    return hook


def register_mlp_hooks(
    model: Any,
    layers: List[int],
    storage: Dict[str, Dict[int, torch.Tensor]],
    capture_pre_activation: bool = True,
    capture_post_activation: bool = True,
    capture_gate: bool = False,
    capture_output: bool = True,
    dtype: torch.dtype = torch.float32
) -> List[RemovableHandle]:
    """
    Register all MLP hooks on a model.
    
    Args:
        model: The LLaMA model (LlamaForCausalLM or similar)
        layers: Which layers to register hooks on
        storage: Dict with keys 'pre_activation', 'post_activation', 'gate', 'up', 'output'
        capture_pre_activation: Whether to capture gate_proj output (before SiLU)
        capture_post_activation: Whether to capture silu(gate)*up
        capture_gate: Whether to capture gate_proj output separately
        capture_output: Whether to capture final MLP output
        dtype: Storage dtype for conversion
        
    Returns:
        List of RemovableHandle for cleanup
    """
    handles = []
    
    # Get the layers module list
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model_layers = model.model.layers
    elif hasattr(model, 'layers'):
        model_layers = model.layers
    else:
        raise ValueError("Cannot find layers in model")
    
    # Temporary storage for computing post_activation
    gate_temp = {} if capture_post_activation else None
    up_temp = {} if capture_post_activation else None
    
    for layer_idx in layers:
        layer = model_layers[layer_idx]
        mlp = layer.mlp
        
        # Gate projection (pre-activation)
        if capture_pre_activation or capture_gate or capture_post_activation:
            target_storage = storage.get('pre_activation', {}) if capture_pre_activation else {}
            
            def make_gate_hook(storage_dict, gate_dict, lidx):
                def hook(module, input, output):
                    tensor = output.detach().to(dtype).cpu()
                    if storage_dict is not None:
                        storage_dict[lidx] = tensor
                    if gate_dict is not None:
                        gate_dict[lidx] = tensor
                return hook
            
            h = mlp.gate_proj.register_forward_hook(
                make_gate_hook(
                    storage['pre_activation'] if capture_pre_activation else None,
                    gate_temp,
                    layer_idx
                )
            )
            handles.append(h)
        
        # Up projection (needed for post_activation)
        if capture_post_activation:
            def make_up_hook(up_dict, lidx):
                def hook(module, input, output):
                    up_dict[lidx] = output.detach().to(dtype).cpu()
                return hook
            
            h = mlp.up_proj.register_forward_hook(make_up_hook(up_temp, layer_idx))
            handles.append(h)
        
        # Post-activation (silu(gate) * up) - computed after MLP forward
        if capture_post_activation:
            h = mlp.register_forward_hook(
                create_post_activation_hook(
                    gate_temp, up_temp, storage['post_activation'], layer_idx, dtype
                )
            )
            handles.append(h)
        
        # MLP output
        if capture_output:
            h = mlp.register_forward_hook(
                create_mlp_output_hook(storage['output'], layer_idx, dtype)
            )
            handles.append(h)
        
        # Gate output (separate from pre_activation if requested)
        if capture_gate and not capture_pre_activation:
            h = mlp.gate_proj.register_forward_hook(
                create_gate_proj_hook(storage['gate'], layer_idx, dtype)
            )
            handles.append(h)
    
    return handles


def assemble_mlp_capture(
    storage: Dict[str, Dict[int, torch.Tensor]],
    layers: List[int],
    n_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    model_name: str = ""
):
    """
    Assemble captured data into MLPCapture structure.
    
    Args:
        storage: Dict with captured data from register_mlp_hooks
        layers: List of captured layer indices
        n_tokens: Number of tokens in sequence
        hidden_dim: Model hidden dimension
        intermediate_dim: MLP intermediate dimension
        model_name: Name of the model
        
    Returns:
        MLPCapture instance
    """
    from oculi.capture.structures import MLPCapture
    
    def stack_storage(key: str) -> Optional[torch.Tensor]:
        """Stack tensors from storage into [L, T, D] tensor."""
        store = storage.get(key, {})
        if not store:
            return None
        
        tensors = []
        for layer_idx in layers:
            if layer_idx in store:
                t = store[layer_idx]
                if t.dim() == 3 and t.shape[0] == 1:
                    t = t.squeeze(0)
                tensors.append(t)
        
        if not tensors:
            return None
        
        return torch.stack(tensors, dim=0)
    
    return MLPCapture(
        pre_activation=stack_storage('pre_activation'),
        post_activation=stack_storage('post_activation'),
        gate_output=stack_storage('gate'),
        mlp_output=stack_storage('output'),
        n_layers=len(layers),
        n_tokens=n_tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        model_name=model_name,
        captured_layers=tuple(layers),
    )
