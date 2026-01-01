"""
LLaMA Model Adapter
===================

Private implementation for LLaMA model family (LLaMA 2, LLaMA 3).

Supports:
- LLaMA 2: 7B, 13B, 70B
- LLaMA 3: 8B, 70B (with GQA)

Hook Points:
- model.layers.{layer}.self_attn.q_proj (Q projection output)
- model.layers.{layer}.self_attn.k_proj (K projection output)
- model.layers.{layer}.self_attn.v_proj (V projection output)
- model.layers.{layer}.self_attn (attention output)
"""

from typing import Optional, Dict, List, Callable
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oculi.capture.adapter import ModelAdapter, CaptureError
from oculi.capture.structures import (
    AttentionCapture,
    AttentionStructure,
    CaptureConfig,
)
from oculi.capture.loader import register_adapter


class LlamaAdapter(ModelAdapter):
    """
    ModelAdapter implementation for LLaMA model family.
    
    Private class — instantiated via oculi.load().
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        self._model_name = model_name
        self._device = torch.device(device)
        self._dtype = dtype
        
        # Load model and tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            **kwargs
        )
        self._model.eval()
        
        # Cache architecture info
        config = self._model.config
        self._n_layers = config.num_hidden_layers
        self._n_heads = config.num_attention_heads
        self._n_kv_heads = getattr(config, 'num_key_value_heads', self._n_heads)
        self._head_dim = config.hidden_size // self._n_heads
        
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
        """
        Capture attention data from a forward pass.
        
        Implements the public ModelAdapter.capture() interface.
        """
        # Default config
        if config is None:
            config = CaptureConfig()
        
        # Validate
        config.validate(self._n_layers)
        
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must be shape [1, seq_len], got {input_ids.shape}"
            )
        
        n_tokens = input_ids.shape[1]
        layers_to_capture = config.get_layers(self._n_layers)
        
        # Storage for captured data
        captured_q = {}
        captured_k = {}
        captured_v = {}
        captured_patterns = {}
        
        # Register hooks
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
            # Forward pass
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
                    # attentions is tuple of [batch, heads, seq, seq]
                    captured_patterns[layer_idx] = outputs.attentions[layer_idx].cpu()
            
        finally:
            # Always remove hooks
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
        """Create a hook that captures output to storage."""
        def hook(module, input, output):
            # Output is [batch, seq, hidden]
            # Need to reshape to [batch, seq, n_heads, head_dim]
            batch, seq, hidden = output.shape
            
            if component == 'q':
                n_heads = self._n_heads
            else:  # k or v
                n_heads = self._n_kv_heads
            
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
        """Assemble captured data into [L, H, T, D] tensor."""
        result = torch.zeros(len(layers), n_heads, n_tokens, head_dim)
        for i, layer_idx in enumerate(layers):
            if layer_idx in storage:
                # storage[layer_idx] is [batch, seq, heads, dim]
                result[i] = storage[layer_idx][0].permute(1, 0, 2)  # [heads, seq, dim] -> needs fix
        return result
    
    def _assemble_patterns(
        self,
        storage: dict,
        layers: List[int]
    ) -> torch.Tensor:
        """Assemble attention patterns into [L, H, T, T] tensor."""
        if not storage:
            return None
        
        first = next(iter(storage.values()))
        _, n_heads, seq, _ = first.shape
        
        result = torch.zeros(len(layers), n_heads, seq, seq)
        for i, layer_idx in enumerate(layers):
            if layer_idx in storage:
                result[i] = storage[layer_idx][0]  # Remove batch dim
        return result
    
    # =========================================================================
    # GENERATION API
    # =========================================================================
    
    def generate(
        self,
        prompt,
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
        hook_fn,
        layer: int,
        component: str,
        stage: str = "post_proj"
    ) -> str:
        """Add a forward hook at specified location."""
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
    
    def _get_hook_module(
        self,
        layer: int,
        component: str,
        stage: str
    ):
        """Get the module to hook based on component and stage."""
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


# Register this adapter with the loader
register_adapter("meta-llama/*", LlamaAdapter)
register_adapter("Meta-Llama/*", LlamaAdapter)
