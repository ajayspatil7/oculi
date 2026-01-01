"""
LLaMA Attention Adapter
=======================

Public adapter for LLaMA model family (LLaMA 2, LLaMA 3).

This adapter provides:
    - Explicit model loading (no magic auto-detection)
    - Attention capture with GQA handling
    - Hook management for interventions
    - Generation utilities

Supported Models:
    - meta-llama/Meta-Llama-3-8B (GQA: 32 Q heads, 8 KV heads)
    - meta-llama/Meta-Llama-3-70B
    - meta-llama/Llama-2-7b-hf (MHA: 32 Q heads, 32 KV heads)
    - meta-llama/Llama-2-13b-hf
    - meta-llama/Llama-2-70b-hf

Usage:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from oculi.models.llama import LlamaAttentionAdapter
    
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    adapter = LlamaAttentionAdapter(model, tokenizer)
    capture = adapter.capture(input_ids)
"""

from typing import Optional, Dict, List, Callable, Union
import torch

from oculi.models.base import AttentionAdapter, CaptureError
from oculi.capture.structures import (
    AttentionCapture,
    AttentionStructure,
    CaptureConfig,
)

# Import attention anatomy (the "where things are" documentation)
from oculi.models.llama.attention import (
    get_q_proj,
    get_k_proj,
    get_v_proj,
    get_o_proj,
    get_attention_module,
    expand_kv_for_gqa,
    create_capture_hook,
)


class LlamaAttentionAdapter(AttentionAdapter):
    """
    Attention adapter for LLaMA model family.
    
    This class wraps a pre-loaded LLaMA model and provides the Oculi
    capture/intervention interface. You must load the model yourself
    (no magic, no hidden state).
    
    Architecture (LLaMA-3-8B):
        - 32 layers
        - 32 query heads, 8 KV heads (GQA 4:1)
        - 128 head dimension
        - 4096 hidden size
        
    Args:
        model: Pre-loaded LlamaForCausalLM
        tokenizer: Pre-loaded tokenizer
        device: Override device (default: model's device)
        
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        >>> adapter = LlamaAttentionAdapter(model, tokenizer)
        >>> print(adapter.attention_structure())
        AttentionStructure(n_query_heads=32, n_kv_heads=8, head_dim=128)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[str] = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        
        # Determine device from model
        if device is not None:
            self._device = torch.device(device)
        else:
            self._device = next(model.parameters()).device
        
        # Cache architecture info from model config
        config = model.config
        self._model_name = getattr(config, '_name_or_path', 'llama')
        self._n_layers = config.num_hidden_layers
        self._n_heads = config.num_attention_heads
        self._n_kv_heads = getattr(config, 'num_key_value_heads', self._n_heads)
        self._head_dim = config.hidden_size // self._n_heads
        
        # Hook management
        self._hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._hook_counter = 0
        
        # Ensure model is in eval mode
        self._model.eval()
    
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
    # DIRECT MODEL ACCESS (for learning/debugging)
    # =========================================================================
    
    @property
    def model(self):
        """Direct access to underlying model (for learning/debugging)."""
        return self._model
    
    @property
    def tokenizer(self):
        """Direct access to tokenizer."""
        return self._tokenizer
    
    def get_attention_module(self, layer: int):
        """
        Get the attention module for a layer.
        
        Returns: LlamaAttention (or LlamaSdpaAttention/LlamaFlashAttention2)
        
        Useful for understanding the exact implementation:
            >>> attn = adapter.get_attention_module(0)
            >>> print(type(attn))  # See which implementation is used
            >>> print(attn.q_proj.weight.shape)  # Inspect weights
        """
        return get_attention_module(self._model, layer)
    
    def get_q_proj(self, layer: int):
        """Get Q projection module. See models/llama/attention.py for details."""
        return get_q_proj(self._model, layer)
    
    def get_k_proj(self, layer: int):
        """Get K projection module. See models/llama/attention.py for details."""
        return get_k_proj(self._model, layer)
    
    def get_v_proj(self, layer: int):
        """Get V projection module. See models/llama/attention.py for details."""
        return get_v_proj(self._model, layer)
    
    # =========================================================================
    # CAPTURE API
    # =========================================================================
    
    def capture(
        self,
        input_ids: torch.Tensor,
        config: Optional[CaptureConfig] = None
    ) -> AttentionCapture:
        """
        Capture attention data from a forward pass.
        
        This registers temporary hooks, runs the forward pass, and 
        assembles the captured data into an AttentionCapture.
        
        Args:
            input_ids: [1, seq_len] token IDs
            config: What to capture (default: all components, all layers)
            
        Returns:
            AttentionCapture with Q, K, V, and/or attention patterns
        """
        if config is None:
            config = CaptureConfig()
        
        config.validate(self._n_layers)
        
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must be shape [1, seq_len], got {input_ids.shape}"
            )
        
        n_tokens = input_ids.shape[1]
        layers_to_capture = config.get_layers(self._n_layers)
        
        # Storage for captured tensors
        captured_q = {}
        captured_k = {}
        captured_v = {}
        captured_patterns = {}
        
        # Register hooks
        handles = []
        
        for layer_idx in layers_to_capture:
            if config.capture_queries:
                hook = create_capture_hook(
                    captured_q, layer_idx, 'q', self._n_heads, self._head_dim
                )
                h = get_q_proj(self._model, layer_idx).register_forward_hook(hook)
                handles.append(h)
            
            if config.capture_keys:
                hook = create_capture_hook(
                    captured_k, layer_idx, 'k', self._n_kv_heads, self._head_dim
                )
                h = get_k_proj(self._model, layer_idx).register_forward_hook(hook)
                handles.append(h)
            
            if config.capture_values:
                hook = create_capture_hook(
                    captured_v, layer_idx, 'v', self._n_kv_heads, self._head_dim
                )
                h = get_v_proj(self._model, layer_idx).register_forward_hook(hook)
                handles.append(h)
        
        try:
            input_ids = input_ids.to(self._device)
            with torch.no_grad():
                outputs = self._model(
                    input_ids,
                    output_attentions=config.capture_patterns,
                    return_dict=True
                )
            
            # Extract attention patterns if requested
            if config.capture_patterns and outputs.attentions is not None:
                for layer_idx in layers_to_capture:
                    captured_patterns[layer_idx] = outputs.attentions[layer_idx].cpu()
        
        finally:
            for h in handles:
                h.remove()
        
        # Assemble tensors
        queries = self._assemble_tensor(
            captured_q, layers_to_capture, self._n_heads, n_tokens
        ) if config.capture_queries else None
        
        keys = self._assemble_tensor(
            captured_k, layers_to_capture, self._n_kv_heads, n_tokens
        ) if config.capture_keys else None
        
        values = self._assemble_tensor(
            captured_v, layers_to_capture, self._n_kv_heads, n_tokens
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
    
    def _assemble_tensor(
        self,
        storage: dict,
        layers: List[int],
        n_heads: int,
        n_tokens: int
    ) -> torch.Tensor:
        """Assemble captured data into [L, H, T, D] tensor."""
        result = torch.zeros(len(layers), n_heads, n_tokens, self._head_dim)
        for i, layer_idx in enumerate(layers):
            if layer_idx in storage:
                # storage is [batch, seq, heads, dim] -> [heads, seq, dim]
                result[i] = storage[layer_idx][0].permute(1, 0, 2)
        return result
    
    def _assemble_patterns(
        self,
        storage: dict,
        layers: List[int]
    ) -> Optional[torch.Tensor]:
        """Assemble attention patterns into [L, H, T, T] tensor."""
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
    # GENERATION API
    # =========================================================================
    
    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int = 100,
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
                temperature=temperature if do_sample else None,
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
        hook_fn: Callable,
        layer: int,
        component: str,
        stage: str = "post_proj"
    ) -> str:
        """
        Add a forward hook at specified location.
        
        Components:
            - 'q': After Q projection (model.layers[l].self_attn.q_proj)
            - 'k': After K projection (model.layers[l].self_attn.k_proj)
            - 'v': After V projection (model.layers[l].self_attn.v_proj)
            - 'attn_out': After output projection (model.layers[l].self_attn.o_proj)
            - 'pattern': Full attention module (for patterns)
        """
        if component == 'q':
            module = get_q_proj(self._model, layer)
        elif component == 'k':
            module = get_k_proj(self._model, layer)
        elif component == 'v':
            module = get_v_proj(self._model, layer)
        elif component == 'attn_out':
            module = get_o_proj(self._model, layer)
        elif component == 'pattern':
            module = get_attention_module(self._model, layer)
        else:
            raise ValueError(
                f"Unknown component: {component}. "
                f"Must be one of: 'q', 'k', 'v', 'attn_out', 'pattern'"
            )
        
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
