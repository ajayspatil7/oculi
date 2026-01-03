"""
LLaMA Model Anatomy
===================

Comprehensive documentation of LLaMA transformer architecture for instrumentation.

This module provides programmatic access to:
    - Hook points for residual stream, MLP, and attention
    - Tensor shapes at each location
    - Module paths for PyTorch hooks

Use this when:
    - Setting up capture hooks
    - Understanding where interventions should go
    - Debugging unexpected tensor shapes

Architecture Reference (LLaMA-3-8B):
    - 32 layers
    - 32 query heads, 8 KV heads (GQA 4:1)
    - 4096 hidden size, 128 head dim
    - 14336 intermediate size (MLP)

Module Structure:
    model.model.embed_tokens
    model.model.layers[i].input_layernorm
    model.model.layers[i].self_attn.{q,k,v,o}_proj
    model.model.layers[i].post_attention_layernorm
    model.model.layers[i].mlp.{gate,up,down}_proj
    model.model.norm
    model.lm_head
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


# =============================================================================
# HOOK POINT DEFINITIONS
# =============================================================================

@dataclass(frozen=True)
class HookPoint:
    """
    Describes a hook location in the LLaMA model.
    
    Attributes:
        name: Short identifier (e.g., "pre_attn")
        module_path: PyTorch module path template (use {i} for layer index)
        tensor_shape: Shape description using B=batch, T=tokens, H=hidden, D=head_dim
        description: Human-readable explanation
        hook_type: 'forward' or 'forward_pre' (pre-hook)
        
    Example:
        >>> hook = RESIDUAL_HOOKS[0]  # pre_attn
        >>> path = hook.module_path.format(i=5)
        >>> # "model.model.layers[5]"
    """
    name: str
    module_path: str
    tensor_shape: str
    description: str
    hook_type: str = "forward"  # "forward" or "forward_pre"


class HookLocation(Enum):
    """Semantic locations for hooks."""
    # Residual stream
    PRE_ATTN = "pre_attn"
    POST_ATTN = "post_attn"
    PRE_MLP = "pre_mlp"
    POST_MLP = "post_mlp"
    
    # Attention components
    Q_PROJ = "q_proj"
    K_PROJ = "k_proj"
    V_PROJ = "v_proj"
    O_PROJ = "o_proj"
    ATTN_OUTPUT = "attn_output"
    
    # MLP components
    GATE_PROJ = "gate_proj"
    UP_PROJ = "up_proj"
    POST_ACTIVATION = "post_activation"
    MLP_OUTPUT = "mlp_output"


# =============================================================================
# RESIDUAL STREAM HOOKS
# =============================================================================

RESIDUAL_HOOKS: List[HookPoint] = [
    HookPoint(
        name="pre_attn",
        module_path="model.model.layers[{i}]",
        tensor_shape="[B, T, H]",
        description="Input to transformer block (before input_layernorm). "
                    "This is the residual stream before attention processing.",
        hook_type="forward_pre",
    ),
    HookPoint(
        name="post_attn",
        module_path="model.model.layers[{i}].self_attn",
        tensor_shape="[B, T, H]",
        description="Output of self_attn module (before residual add). "
                    "The attention contribution to the residual stream.",
        hook_type="forward",
    ),
    HookPoint(
        name="pre_mlp",
        module_path="model.model.layers[{i}].post_attention_layernorm",
        tensor_shape="[B, T, H]",
        description="Output of post_attention_layernorm (before MLP). "
                    "Input to the MLP block.",
        hook_type="forward",
    ),
    HookPoint(
        name="post_mlp",
        module_path="model.model.layers[{i}].mlp",
        tensor_shape="[B, T, H]",
        description="Output of MLP module (before final residual add). "
                    "The MLP contribution to the residual stream.",
        hook_type="forward",
    ),
]


# =============================================================================
# ATTENTION COMPONENT HOOKS
# =============================================================================

ATTENTION_HOOKS: List[HookPoint] = [
    HookPoint(
        name="q_proj",
        module_path="model.model.layers[{i}].self_attn.q_proj",
        tensor_shape="[B, T, n_heads * head_dim]",
        description="Query projection output. For LLaMA-3-8B: [B, T, 4096]. "
                    "Reshape to [B, T, n_heads, head_dim] for per-head analysis.",
        hook_type="forward",
    ),
    HookPoint(
        name="k_proj",
        module_path="model.model.layers[{i}].self_attn.k_proj",
        tensor_shape="[B, T, n_kv_heads * head_dim]",
        description="Key projection output. For LLaMA-3-8B GQA: [B, T, 1024]. "
                    "Note: smaller than q_proj due to GQA!",
        hook_type="forward",
    ),
    HookPoint(
        name="v_proj",
        module_path="model.model.layers[{i}].self_attn.v_proj",
        tensor_shape="[B, T, n_kv_heads * head_dim]",
        description="Value projection output. Same shape as k_proj.",
        hook_type="forward",
    ),
    HookPoint(
        name="o_proj",
        module_path="model.model.layers[{i}].self_attn.o_proj",
        tensor_shape="[B, T, H]",
        description="Output projection. Projects attention output back to hidden dim.",
        hook_type="forward",
    ),
]


# =============================================================================
# MLP COMPONENT HOOKS
# =============================================================================

MLP_HOOKS: List[HookPoint] = [
    HookPoint(
        name="gate_proj",
        module_path="model.model.layers[{i}].mlp.gate_proj",
        tensor_shape="[B, T, I]",
        description="Gate projection output. I = intermediate_dim (14336 for LLaMA-3-8B). "
                    "This is passed through SiLU activation.",
        hook_type="forward",
    ),
    HookPoint(
        name="up_proj",
        module_path="model.model.layers[{i}].mlp.up_proj",
        tensor_shape="[B, T, I]",
        description="Up projection output. Element-wise multiplied with activated gate.",
        hook_type="forward",
    ),
    HookPoint(
        name="post_activation",
        module_path="model.model.layers[{i}].mlp",
        tensor_shape="[B, T, I]",
        description="After SiLU(gate) * up. The 'neuron activations' in interpretability terms. "
                    "NOTE: Requires forward hook with custom extraction (not a direct module output).",
        hook_type="forward",  # Special handling needed
    ),
    HookPoint(
        name="mlp_output",
        module_path="model.model.layers[{i}].mlp",
        tensor_shape="[B, T, H]",
        description="Full MLP output after down_proj. The MLP contribution to residual stream.",
        hook_type="forward",
    ),
]


# =============================================================================
# EMBEDDING AND OUTPUT HOOKS
# =============================================================================

EMBEDDING_HOOKS: List[HookPoint] = [
    HookPoint(
        name="embed_tokens",
        module_path="model.model.embed_tokens",
        tensor_shape="[B, T, H]",
        description="Token embeddings (layer 0 input).",
        hook_type="forward",
    ),
    HookPoint(
        name="final_norm",
        module_path="model.model.norm",
        tensor_shape="[B, T, H]",
        description="Output of final layer norm (before lm_head).",
        hook_type="forward",
    ),
    HookPoint(
        name="lm_head",
        module_path="model.lm_head",
        tensor_shape="[B, T, V]",
        description="Final logits. V = vocab_size.",
        hook_type="forward",
    ),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_hook_by_name(name: str) -> Optional[HookPoint]:
    """
    Get a HookPoint by its name.
    
    Args:
        name: Hook name (e.g., "pre_attn", "q_proj")
        
    Returns:
        HookPoint if found, None otherwise
    """
    all_hooks = RESIDUAL_HOOKS + ATTENTION_HOOKS + MLP_HOOKS + EMBEDDING_HOOKS
    for hook in all_hooks:
        if hook.name == name:
            return hook
    return None


def get_module_path(hook: HookPoint, layer: int) -> str:
    """
    Get the actual module path for a specific layer.
    
    Args:
        hook: HookPoint definition
        layer: Layer index
        
    Returns:
        Formatted module path string
        
    Example:
        >>> hook = get_hook_by_name("q_proj")
        >>> path = get_module_path(hook, 5)
        >>> # "model.model.layers[5].self_attn.q_proj"
    """
    return hook.module_path.format(i=layer)


def get_all_residual_paths(n_layers: int) -> Dict[str, List[str]]:
    """
    Get all residual stream hook paths for a model.
    
    Args:
        n_layers: Number of transformer layers
        
    Returns:
        Dict mapping hook name to list of module paths (one per layer)
    """
    result = {}
    for hook in RESIDUAL_HOOKS:
        result[hook.name] = [
            get_module_path(hook, i) for i in range(n_layers)
        ]
    return result


# =============================================================================
# MODEL DIMENSION REFERENCE
# =============================================================================

@dataclass(frozen=True)
class ModelDimensions:
    """Reference dimensions for different LLaMA variants."""
    name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    hidden_size: int
    intermediate_size: int
    head_dim: int
    vocab_size: int
    
    @property
    def is_gqa(self) -> bool:
        return self.n_heads != self.n_kv_heads
    
    @property
    def gqa_ratio(self) -> int:
        return self.n_heads // self.n_kv_heads


# Common LLaMA configurations
LLAMA_3_8B = ModelDimensions(
    name="Meta-Llama-3-8B",
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    hidden_size=4096,
    intermediate_size=14336,
    head_dim=128,
    vocab_size=128256,
)

LLAMA_3_70B = ModelDimensions(
    name="Meta-Llama-3-70B",
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    hidden_size=8192,
    intermediate_size=28672,
    head_dim=128,
    vocab_size=128256,
)

LLAMA_2_7B = ModelDimensions(
    name="Llama-2-7b",
    n_layers=32,
    n_heads=32,
    n_kv_heads=32,  # MHA, not GQA
    hidden_size=4096,
    intermediate_size=11008,
    head_dim=128,
    vocab_size=32000,
)

# Lookup table
MODEL_DIMENSIONS: Dict[str, ModelDimensions] = {
    "llama-3-8b": LLAMA_3_8B,
    "llama-3-70b": LLAMA_3_70B,
    "llama-2-7b": LLAMA_2_7B,
}


# =============================================================================
# TRANSFORMER BLOCK STRUCTURE (for reference)
# =============================================================================

"""
LLaMA Transformer Block Structure
=================================

For each layer i:

    Input: residual_stream[i]   # [B, T, H]
    
    # Attention sub-block
    1. x = input_layernorm(residual_stream[i])           # pre_attn after norm
    2. q = q_proj(x)                                      # [B, T, n_heads * d]
    3. k = k_proj(x)                                      # [B, T, n_kv_heads * d]
    4. v = v_proj(x)                                      # [B, T, n_kv_heads * d]
    5. q, k = apply_rope(q, k)                            # RoPE applied
    6. k, v = expand_kv_for_gqa(k, v)                     # GQA expansion
    7. attn_out = attention(q, k, v)                      # [B, T, n_heads * d]
    8. attn_out = o_proj(attn_out)                        # post_attn: [B, T, H]
    9. residual_stream += attn_out                        # Residual connection
    
    # MLP sub-block
    10. x = post_attention_layernorm(residual_stream)     # pre_mlp
    11. gate = gate_proj(x)                               # [B, T, I]
    12. up = up_proj(x)                                   # [B, T, I]
    13. hidden = silu(gate) * up                          # post_activation: [B, T, I]
    14. mlp_out = down_proj(hidden)                       # post_mlp: [B, T, H]
    15. residual_stream += mlp_out                        # Residual connection
    
    Output: residual_stream[i+1]   # [B, T, H]

Key Intervention Points:
    - pre_attn (step 1): Modify information before attention sees it
    - post_attn (step 8): Modify attention's contribution to residual
    - pre_mlp (step 10): Modify information before MLP sees it
    - post_mlp (step 14): Modify MLP's contribution to residual
    - post_activation (step 13): Modify neuron activations

Note on Residual Stream:
    post_mlp[i] ≈ pre_attn[i+1] (with layer norm)
    This allows tracking information flow across layers.
"""
