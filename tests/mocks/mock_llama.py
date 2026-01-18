"""
Mock LLaMA Model for Testing
============================

A minimal LLaMA-3 style model with tiny dimensions for local testing.
Mimics the exact architecture of Meta-Llama-3-8B but with:
- 4 layers instead of 32
- 4 query heads instead of 32  
- 2 KV heads instead of 8 (GQA ratio 2:1)
- 64 head dim instead of 128
- 256 hidden size instead of 4096

This allows testing Oculi on a MacBook without GPU or large memory.

Usage:
    from tests.mocks.mock_llama import MockLlamaForCausalLM, MockLlamaConfig
    
    config = MockLlamaConfig()
    model = MockLlamaForCausalLM(config)
    
    # Or use the adapter directly:
    from tests.mocks.mock_llama import MockLlamaAdapter
    adapter = MockLlamaAdapter()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Callable, Union

# Import Oculi interfaces
import sys
sys.path.insert(0, '/Users/ajaysp/oculi')
from oculi.models.base import AttentionAdapter
from oculi.capture.structures import (
    AttentionCapture,
    AttentionStructure,
    CaptureConfig,
    ResidualCapture,
    ResidualConfig,
    MLPCapture,
    MLPConfig,
    FullCapture,
)


@dataclass
class MockLlamaConfig:
    """
    Configuration mimicking LLaMA-3-8B architecture with tiny dimensions.
    
    Real LLaMA-3-8B:
        - num_hidden_layers: 32
        - num_attention_heads: 32
        - num_key_value_heads: 8
        - hidden_size: 4096
        - head_dim: 128
        - vocab_size: 128256
    
    This mock:
        - Preserves GQA ratio (4:1 -> 2:1)
        - Uses tiny dimensions for CPU testing
    """
    num_hidden_layers: int = 4
    num_attention_heads: int = 4      # Query heads
    num_key_value_heads: int = 2      # KV heads (GQA 2:1)
    hidden_size: int = 256
    intermediate_size: int = 512
    vocab_size: int = 1000
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        
        # Precompute cos/sin
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int):
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding to Q and K."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MockLlamaAttention(nn.Module):
    """
    LLaMA-style attention with GQA (Grouped Query Attention).
    
    Matches the exact module structure of real LLaMA for hook compatibility:
    - self.q_proj: Query projection
    - self.k_proj: Key projection  
    - self.v_proj: Value projection
    - self.o_proj: Output projection
    """
    
    def __init__(self, config: MockLlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        # Projections (matching LLaMA naming for hook compatibility)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(query_states, seq_len)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # GQA: Repeat KV heads
        key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Compute attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Apply external mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None


class MockLlamaMLP(nn.Module):
    """LLaMA-style MLP with SiLU activation."""
    
    def __init__(self, config: MockLlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MockLlamaDecoderLayer(nn.Module):
    """Single LLaMA decoder layer."""
    
    def __init__(self, config: MockLlamaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MockLlamaAttention(config, layer_idx)
        self.mlp = MockLlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, attn_weights


class MockLlamaModel(nn.Module):
    """
    Core LLaMA model (without LM head).
    
    Matches structure: model.layers[i].self_attn.{q,k,v,o}_proj
    """
    
    def __init__(self, config: MockLlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MockLlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        hidden_states = self.embed_tokens(input_ids)
        
        all_attentions = () if output_attentions else None
        
        for layer in self.layers:
            hidden_states, attn_weights = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, all_attentions


class MockLlamaForCausalLM(nn.Module):
    """
    LLaMA model with Language Modeling head.
    
    This is the full model that matches the structure expected by LlamaAdapter:
    - model.model.layers[i].self_attn.q_proj
    - model.model.layers[i].self_attn.k_proj
    - etc.
    """
    
    def __init__(self, config: MockLlamaConfig):
        super().__init__()
        self.config = config
        self.model = MockLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        return_dict: bool = True,
    ):
        hidden_states, attentions = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return MockCausalLMOutput(
                logits=logits,
                attentions=attentions,
            )
        return logits, attentions
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Simple autoregressive generation."""
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            outputs = self(generated)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Stop at EOS or pad
            if pad_token_id is not None and next_token.item() == pad_token_id:
                break
        
        return generated


@dataclass
class MockCausalLMOutput:
    """Output container matching HuggingFace format."""
    logits: torch.Tensor
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class MockTokenizer:
    """Simple tokenizer for testing."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.pad_token_id = 0
        
        # Simple char-to-id mapping
        self.vocab = {chr(i): i % vocab_size for i in range(256)}
    
    def encode(self, text: str, return_tensors: str = None) -> torch.Tensor:
        """Encode text to token IDs."""
        ids = [self.vocab.get(c, 0) for c in text]
        if return_tensors == 'pt':
            return torch.tensor([ids])
        return ids
    
    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text (simplified)."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        # Simple mock decode - just returns placeholder
        return f"[decoded {len(token_ids)} tokens]"


# =============================================================================
# MOCK LLAMA ADAPTER - Compatible with Oculi API
# =============================================================================

class MockLlamaAdapter(AttentionAdapter):
    """
    Oculi ModelAdapter for MockLlamaForCausalLM.
    
    Drop-in replacement for testing on CPU without loading full LLaMA.
    
    Usage:
        adapter = MockLlamaAdapter()
        capture = adapter.capture(adapter.tokenize("Hello world"))
    """
    
    def __init__(
        self,
        config: Optional[MockLlamaConfig] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self._config = config or MockLlamaConfig()
        self._device = torch.device(device)
        self._dtype = dtype
        self._model_name = "mock-llama-tiny"
        
        # Create model and tokenizer
        self._model = MockLlamaForCausalLM(self._config).to(self._device).to(self._dtype)
        self._model.eval()
        self._tokenizer = MockTokenizer(self._config.vocab_size)
        
        # Cache architecture info
        self._n_layers = self._config.num_hidden_layers
        self._n_heads = self._config.num_attention_heads
        self._n_kv_heads = self._config.num_key_value_heads
        self._head_dim = self._config.head_dim
        
        # Hook management
        self._hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._hook_counter = 0
    
    # =========================================================================
    # ARCHITECTURE INTROSPECTION
    # =========================================================================
    
    def num_layers(self) -> int:
        return self._n_layers
    
    def num_heads(self, layer: int = 0) -> int:
        return self._n_heads
    
    def num_kv_heads(self, layer: int = 0) -> int:
        return self._n_kv_heads
    
    def head_dim(self, layer: int = 0) -> int:
        return self._head_dim
    
    def attention_structure(self, layer: int = 0) -> AttentionStructure:
        return AttentionStructure(
            n_query_heads=self._n_heads,
            n_kv_heads=self._n_kv_heads,
            head_dim=self._head_dim
        )
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    # =========================================================================
    # CAPTURE API
    # =========================================================================
    
    def capture(
        self,
        input_ids: torch.Tensor,
        config: Optional[CaptureConfig] = None
    ) -> AttentionCapture:
        """Capture attention data from forward pass."""
        if config is None:
            config = CaptureConfig()
        
        config.validate(self._n_layers)
        
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must be shape [1, seq_len], got {input_ids.shape}"
            )
        
        n_tokens = input_ids.shape[1]
        layers_to_capture = config.get_layers(self._n_layers)
        
        # Storage
        captured_q = {}
        captured_k = {}
        captured_v = {}
        captured_patterns = {}
        
        handles = []
        
        for layer_idx in layers_to_capture:
            layer = self._model.model.layers[layer_idx]
            
            if config.capture_queries:
                h = layer.self_attn.q_proj.register_forward_hook(
                    self._make_capture_hook(captured_q, layer_idx, 'q')
                )
                handles.append(h)
            
            if config.capture_keys:
                h = layer.self_attn.k_proj.register_forward_hook(
                    self._make_capture_hook(captured_k, layer_idx, 'k')
                )
                handles.append(h)
            
            if config.capture_values:
                h = layer.self_attn.v_proj.register_forward_hook(
                    self._make_capture_hook(captured_v, layer_idx, 'v')
                )
                handles.append(h)
        
        try:
            input_ids = input_ids.to(self._device)
            with torch.no_grad():
                outputs = self._model(
                    input_ids,
                    output_attentions=config.capture_patterns,
                    return_dict=True
                )
            
            if config.capture_patterns and outputs.attentions is not None:
                for layer_idx in layers_to_capture:
                    captured_patterns[layer_idx] = outputs.attentions[layer_idx].cpu()
        
        finally:
            for h in handles:
                h.remove()
        
        # Assemble tensors
        queries = self._assemble_tensor(
            captured_q, layers_to_capture, self._n_heads, n_tokens, self._head_dim
        ) if config.capture_queries else None
        
        keys = self._assemble_tensor(
            captured_k, layers_to_capture, self._n_kv_heads, n_tokens, self._head_dim
        ) if config.capture_keys else None
        
        values = self._assemble_tensor(
            captured_v, layers_to_capture, self._n_kv_heads, n_tokens, self._head_dim
        ) if config.capture_values else None
        
        patterns = self._assemble_patterns(
            captured_patterns, layers_to_capture
        ) if config.capture_patterns else None
        
        return AttentionCapture(
            queries=queries,
            keys=keys,
            values=values,
            patterns=patterns,
            n_layers=len(layers_to_capture),
            n_heads=self._n_heads,
            n_kv_heads=self._n_kv_heads,
            n_tokens=n_tokens,
            head_dim=self._head_dim,
            model_name=self._model_name,
            qk_stage=config.qk_stage,
            captured_layers=tuple(layers_to_capture),
        )
    
    def _make_capture_hook(
        self,
        storage: dict,
        layer_idx: int,
        component: str
    ) -> Callable:
        def hook(module, input, output):
            batch, seq, hidden = output.shape
            n_heads = self._n_heads if component == 'q' else self._n_kv_heads
            reshaped = output.view(batch, seq, n_heads, self._head_dim)
            storage[layer_idx] = reshaped.detach().cpu()
        return hook
    
    def _assemble_tensor(
        self,
        storage: dict,
        layers: List[int],
        n_heads: int,
        n_tokens: int,
        head_dim: int
    ) -> torch.Tensor:
        result = torch.zeros(len(layers), n_heads, n_tokens, head_dim)
        for i, layer_idx in enumerate(layers):
            if layer_idx in storage:
                # [batch, seq, heads, dim] -> [heads, seq, dim]
                result[i] = storage[layer_idx][0].permute(1, 0, 2)
        return result
    
    def _assemble_patterns(
        self,
        storage: dict,
        layers: List[int]
    ) -> Optional[torch.Tensor]:
        if not storage:
            return None
        
        first = next(iter(storage.values()))
        _, n_heads, seq, _ = first.shape
        
        result = torch.zeros(len(layers), n_heads, seq, seq)
        for i, layer_idx in enumerate(layers):
            if layer_idx in storage:
                result[i] = storage[layer_idx][0]
        return result
    
    # =========================================================================
    # RESIDUAL STREAM CAPTURE API (Phase 1)
    # =========================================================================
    
    def capture_residual(
        self,
        input_ids: torch.Tensor,
        config: Optional[ResidualConfig] = None
    ) -> ResidualCapture:
        """Capture residual stream activations during forward pass."""
        from oculi._private.hooks.residual import (
            register_residual_hooks,
            assemble_residual_capture,
        )
        
        if config is None:
            config = ResidualConfig()
        
        config.validate(self._n_layers)
        
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"input_ids must be shape [1, seq_len], got {input_ids.shape}")
        
        n_tokens = input_ids.shape[1]
        layers_to_capture = config.get_layers(self._n_layers)
        hidden_dim = self._config.hidden_size
        
        storage = {
            'pre_attn': {},
            'post_attn': {},
            'pre_mlp': {},
            'post_mlp': {},
        }
        
        handles = register_residual_hooks(
            self._model,
            layers_to_capture,
            storage,
            capture_pre_attn=config.capture_pre_attn,
            capture_post_attn=config.capture_post_attn,
            capture_pre_mlp=config.capture_pre_mlp,
            capture_post_mlp=config.capture_post_mlp,
            dtype=config.storage_dtype,
        )
        
        try:
            input_ids = input_ids.to(self._device)
            with torch.no_grad():
                self._model(input_ids, return_dict=True)
        finally:
            for h in handles:
                h.remove()
        
        return assemble_residual_capture(
            storage,
            layers_to_capture,
            n_tokens,
            hidden_dim,
            self._model_name,
        )
    
    # =========================================================================
    # MLP CAPTURE API (Phase 1)
    # =========================================================================
    
    def capture_mlp(
        self,
        input_ids: torch.Tensor,
        config: Optional[MLPConfig] = None
    ) -> MLPCapture:
        """Capture MLP internals during forward pass."""
        from oculi._private.hooks.mlp import (
            register_mlp_hooks,
            assemble_mlp_capture,
        )
        
        if config is None:
            config = MLPConfig()
        
        config.validate(self._n_layers)
        
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"input_ids must be shape [1, seq_len], got {input_ids.shape}")
        
        n_tokens = input_ids.shape[1]
        layers_to_capture = config.get_layers(self._n_layers)
        hidden_dim = self._config.hidden_size
        intermediate_dim = self._config.intermediate_size
        
        storage = {
            'pre_activation': {},
            'post_activation': {},
            'gate': {},
            'output': {},
        }
        
        handles = register_mlp_hooks(
            self._model,
            layers_to_capture,
            storage,
            capture_pre_activation=config.capture_pre_activation,
            capture_post_activation=config.capture_post_activation,
            capture_gate=config.capture_gate,
            capture_output=config.capture_output,
            dtype=config.storage_dtype,
        )
        
        try:
            input_ids = input_ids.to(self._device)
            with torch.no_grad():
                self._model(input_ids, return_dict=True)
        finally:
            for h in handles:
                h.remove()
        
        return assemble_mlp_capture(
            storage,
            layers_to_capture,
            n_tokens,
            hidden_dim,
            intermediate_dim,
            self._model_name,
        )
    
    # =========================================================================
    # FULL CAPTURE API (Phase 1)
    # =========================================================================
    
    def capture_full(
        self,
        input_ids: torch.Tensor,
        attention_config: Optional[CaptureConfig] = None,
        residual_config: Optional[ResidualConfig] = None,
        mlp_config: Optional[MLPConfig] = None
    ) -> FullCapture:
        """Capture attention, residual, and MLP in a single forward pass."""
        from oculi._private.hooks.residual import (
            register_residual_hooks,
            assemble_residual_capture,
        )
        from oculi._private.hooks.mlp import (
            register_mlp_hooks,
            assemble_mlp_capture,
        )
        
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"input_ids must be shape [1, seq_len], got {input_ids.shape}")
        
        n_tokens = input_ids.shape[1]
        hidden_dim = self._config.hidden_size
        intermediate_dim = self._config.intermediate_size
        
        all_handles = []
        
        # Attention hooks setup
        captured_q = {}
        captured_k = {}
        captured_v = {}
        attn_layers = []
        
        if attention_config is not None:
            attention_config.validate(self._n_layers)
            attn_layers = attention_config.get_layers(self._n_layers)
            
            for layer_idx in attn_layers:
                layer = self._model.model.layers[layer_idx]
                
                if attention_config.capture_queries:
                    def make_qkv_hook(storage, lidx, n_h, h_d):
                        def hook(module, input, output):
                            # output: [batch, seq, n_heads * head_dim]
                            reshaped = output.view(output.shape[0], output.shape[1], n_h, h_d)
                            storage[lidx] = reshaped.detach().cpu()
                        return hook
                    h = layer.self_attn.q_proj.register_forward_hook(
                        make_qkv_hook(captured_q, layer_idx, self._n_heads, self._head_dim)
                    )
                    all_handles.append(h)
                
                if attention_config.capture_keys:
                    def make_kv_hook(storage, lidx, n_h, h_d):
                        def hook(module, input, output):
                            reshaped = output.view(output.shape[0], output.shape[1], n_h, h_d)
                            storage[lidx] = reshaped.detach().cpu()
                        return hook
                    h = layer.self_attn.k_proj.register_forward_hook(
                        make_kv_hook(captured_k, layer_idx, self._n_kv_heads, self._head_dim)
                    )
                    all_handles.append(h)
                
                if attention_config.capture_values:
                    h = layer.self_attn.v_proj.register_forward_hook(
                        make_kv_hook(captured_v, layer_idx, self._n_kv_heads, self._head_dim)
                    )
                    all_handles.append(h)
        
        # Residual hooks
        residual_storage = {
            'pre_attn': {},
            'post_attn': {},
            'pre_mlp': {},
            'post_mlp': {},
        }
        residual_layers = []
        
        if residual_config is not None:
            residual_config.validate(self._n_layers)
            residual_layers = residual_config.get_layers(self._n_layers)
            
            handles = register_residual_hooks(
                self._model,
                residual_layers,
                residual_storage,
                capture_pre_attn=residual_config.capture_pre_attn,
                capture_post_attn=residual_config.capture_post_attn,
                capture_pre_mlp=residual_config.capture_pre_mlp,
                capture_post_mlp=residual_config.capture_post_mlp,
                dtype=residual_config.storage_dtype,
            )
            all_handles.extend(handles)
        
        # MLP hooks
        mlp_storage = {
            'pre_activation': {},
            'post_activation': {},
            'gate': {},
            'output': {},
        }
        mlp_layers = []
        
        if mlp_config is not None:
            mlp_config.validate(self._n_layers)
            mlp_layers = mlp_config.get_layers(self._n_layers)
            
            handles = register_mlp_hooks(
                self._model,
                mlp_layers,
                mlp_storage,
                capture_pre_activation=mlp_config.capture_pre_activation,
                capture_post_activation=mlp_config.capture_post_activation,
                capture_gate=mlp_config.capture_gate,
                capture_output=mlp_config.capture_output,
                dtype=mlp_config.storage_dtype,
            )
            all_handles.extend(handles)
        
        # Run forward pass
        try:
            input_ids = input_ids.to(self._device)
            with torch.no_grad():
                outputs = self._model(
                    input_ids,
                    output_attentions=attention_config.capture_patterns if attention_config else False,
                    return_dict=True
                )
            
            # Handle attention patterns
            captured_patterns = {}
            if attention_config and attention_config.capture_patterns:
                if outputs.attentions is not None:
                    for layer_idx in attn_layers:
                        captured_patterns[layer_idx] = outputs.attentions[layer_idx].cpu()
                else:
                    captured_patterns = self._compute_attention_patterns(
                        captured_q, captured_k, attn_layers, n_tokens
                    )
        finally:
            for h in all_handles:
                h.remove()
        
        # Assemble captures
        attention_capture = None
        if attention_config is not None:
            queries = self._assemble_tensor(
                captured_q, attn_layers, self._n_heads, n_tokens, self._head_dim
            ) if attention_config.capture_queries else None
            
            keys = self._assemble_tensor(
                captured_k, attn_layers, self._n_kv_heads, n_tokens, self._head_dim
            ) if attention_config.capture_keys else None
            
            values = self._assemble_tensor(
                captured_v, attn_layers, self._n_kv_heads, n_tokens, self._head_dim
            ) if attention_config.capture_values else None
            
            patterns = self._assemble_patterns(
                captured_patterns, attn_layers
            ) if attention_config.capture_patterns else None
            
            attention_capture = AttentionCapture(
                queries=queries,
                keys=keys,
                values=values,
                patterns=patterns,
                n_layers=len(attn_layers),
                n_heads=self._n_heads,
                n_kv_heads=self._n_kv_heads,
                n_tokens=n_tokens,
                head_dim=self._head_dim,
                model_name=self._model_name,
                qk_stage=attention_config.qk_stage,
                captured_layers=tuple(attn_layers),
            )
        
        residual_capture = None
        if residual_config is not None:
            residual_capture = assemble_residual_capture(
                residual_storage,
                residual_layers,
                n_tokens,
                hidden_dim,
                self._model_name,
            )
        
        mlp_capture = None
        if mlp_config is not None:
            mlp_capture = assemble_mlp_capture(
                mlp_storage,
                mlp_layers,
                n_tokens,
                hidden_dim,
                intermediate_dim,
                self._model_name,
            )
        
        return FullCapture(
            attention=attention_capture,
            residual=residual_capture,
            mlp=mlp_capture,
        )
    
    # =========================================================================
    # GENERATION API
    # =========================================================================
    
    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        if isinstance(prompt, str):
            input_ids = self.tokenize(prompt)
        else:
            input_ids = prompt
        
        input_ids = input_ids.to(self._device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
                **kwargs
            )
        
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def tokenize(self, text: str) -> torch.Tensor:
        return self._tokenizer.encode(text, return_tensors='pt')
    
    def decode(self, token_ids: torch.Tensor) -> str:
        return self._tokenizer.decode(token_ids.squeeze(), skip_special_tokens=True)
    
    # =========================================================================
    # HOOK MANAGEMENT
    # =========================================================================
    
    def add_hook(
        self,
        hook_fn,
        layer: int,
        component: str,
        stage: str = "post_proj"
    ) -> str:
        module = self._get_hook_module(layer, component, stage)
        handle = module.register_forward_hook(hook_fn)
        
        handle_id = f"hook_{self._hook_counter}"
        self._hook_counter += 1
        self._hooks[handle_id] = handle
        
        return handle_id
    
    def remove_hook(self, handle_id: str) -> None:
        if handle_id in self._hooks:
            self._hooks[handle_id].remove()
            del self._hooks[handle_id]
    
    def reset_hooks(self) -> None:
        for handle in self._hooks.values():
            handle.remove()
        self._hooks.clear()
    
    def _get_hook_module(self, layer: int, component: str, stage: str):
        layer_module = self._model.model.layers[layer]
        
        if component == 'q':
            return layer_module.self_attn.q_proj
        elif component == 'k':
            return layer_module.self_attn.k_proj
        elif component == 'v':
            return layer_module.self_attn.v_proj
        elif component == 'attn_out':
            return layer_module.self_attn.o_proj
        elif component == 'pattern':
            return layer_module.self_attn
        else:
            raise ValueError(f"Unknown component: {component}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_mock_adapter(
    n_layers: int = 4,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    hidden_size: int = 256,
    device: str = "cpu",
) -> MockLlamaAdapter:
    """
    Create a MockLlamaAdapter with custom dimensions.
    
    Args:
        n_layers: Number of transformer layers
        n_heads: Number of query heads
        n_kv_heads: Number of KV heads (for GQA)
        hidden_size: Model hidden dimension
        device: Device to load model on
        
    Returns:
        MockLlamaAdapter ready for testing
        
    Example:
        >>> adapter = create_mock_adapter(n_layers=8, n_heads=8)
        >>> capture = adapter.capture(adapter.tokenize("test"))
    """
    config = MockLlamaConfig(
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        hidden_size=hidden_size,
    )
    return MockLlamaAdapter(config=config, device=device)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print(" Creating MockLlamaAdapter...")
    adapter = MockLlamaAdapter()
    
    print(f"\n Model Architecture:")
    print(f"   Layers: {adapter.num_layers()}")
    print(f"   Query Heads: {adapter.num_heads()}")
    print(f"   KV Heads: {adapter.num_kv_heads()}")
    print(f"   Head Dim: {adapter.head_dim()}")
    print(f"   Attention Type: {adapter.attention_structure().attention_type}")
    print(f"   GQA Ratio: {adapter.attention_structure().gqa_ratio}:1")
    
    print("\n Testing capture...")
    input_ids = adapter.tokenize("Hello, this is a test!")
    print(f"   Input shape: {input_ids.shape}")
    
    capture = adapter.capture(input_ids)
    print(f"   Queries shape: {capture.queries.shape}")
    print(f"   Keys shape: {capture.keys.shape}")
    print(f"   Values shape: {capture.values.shape}")
    print(f"   Patterns shape: {capture.patterns.shape}")
    
    print("\n Mock LLaMA adapter working correctly!")
