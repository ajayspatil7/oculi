"""
Spectra Model Adapters
======================

Provides a unified interface for different model architectures.
Dynamically adapts to model properties (layers, heads, GQA, etc).
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


@dataclass
class ModelInfo:
    """Model architecture information extracted dynamically."""
    name: str
    family: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    max_position_embeddings: int
    is_gqa: bool  # Grouped Query Attention
    
    @property
    def total_heads(self) -> int:
        return self.n_layers * self.n_heads
    
    def __repr__(self) -> str:
        return (
            f"ModelInfo({self.name})\n"
            f"  Layers: {self.n_layers}\n"
            f"  Heads: {self.n_heads} (KV: {self.n_kv_heads})\n"
            f"  GQA: {self.is_gqa}\n"
            f"  Head dim: {self.head_dim}\n"
            f"  Max context: {self.max_position_embeddings}"
        )


class ModelAdapter:
    """
    Unified adapter for different transformer architectures.
    
    Handles:
    - Model loading
    - Architecture detection
    - Layer/head access
    - GQA vs MHA differences
    """
    
    # Known model family mappings
    FAMILY_MAP = {
        'llama': ['llama', 'meta-llama'],
        'mistral': ['mistral'],
        'qwen': ['qwen'],
        'gemma': ['gemma'],
    }
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
        family: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.family = family or self._detect_family(model_name)
        
        self.model = None
        self.tokenizer = None
        self.info: Optional[ModelInfo] = None
    
    def _detect_family(self, model_name: str) -> str:
        """Auto-detect model family from name."""
        name_lower = model_name.lower()
        for family, keywords in self.FAMILY_MAP.items():
            if any(kw in name_lower for kw in keywords):
                return family
        return "unknown"
    
    def load(self) -> "ModelAdapter":
        """Load model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device
        )
        self.model.eval()
        
        # Extract architecture info
        self.info = self._extract_info()
        
        print(self.info)
        return self
    
    def _extract_info(self) -> ModelInfo:
        """Extract model architecture info dynamically."""
        config = self.model.config
        
        # Get basic dimensions
        n_layers = config.num_hidden_layers
        n_heads = config.num_attention_heads
        hidden_size = config.hidden_size
        head_dim = hidden_size // n_heads
        
        # Get KV heads (for GQA)
        n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
        is_gqa = n_kv_heads != n_heads
        
        # Get max context
        max_pos = getattr(config, 'max_position_embeddings', 4096)
        
        return ModelInfo(
            name=self.model_name,
            family=self.family,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            max_position_embeddings=max_pos,
            is_gqa=is_gqa
        )
    
    def get_layer(self, layer_idx: int):
        """Get a specific transformer layer."""
        if self.family in ['llama', 'mistral', 'qwen']:
            return self.model.model.layers[layer_idx]
        elif self.family == 'gemma':
            return self.model.model.layers[layer_idx]
        else:
            # Generic fallback
            return self.model.model.layers[layer_idx]
    
    def get_attention(self, layer_idx: int):
        """Get the attention module for a layer."""
        layer = self.get_layer(layer_idx)
        return layer.self_attn
    
    def get_input_layernorm(self, layer_idx: int):
        """Get input LayerNorm for a layer."""
        layer = self.get_layer(layer_idx)
        return layer.input_layernorm
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> Any:
        """Run forward pass."""
        with torch.no_grad():
            return self.model(input_ids, use_cache=False, **kwargs)
    
    def to_dict(self) -> Dict:
        """Export model info as dict for metadata."""
        if self.info is None:
            return {"name": self.model_name, "loaded": False}
        return {
            "name": self.info.name,
            "family": self.info.family,
            "n_layers": self.info.n_layers,
            "n_heads": self.info.n_heads,
            "n_kv_heads": self.info.n_kv_heads,
            "head_dim": self.info.head_dim,
            "hidden_size": self.info.hidden_size,
            "max_position_embeddings": self.info.max_position_embeddings,
            "is_gqa": self.info.is_gqa,
        }


def load_model(model_name: str, family: str = None) -> ModelAdapter:
    """
    Convenience function to load and return a model adapter.
    
    Usage:
        adapter = load_model("meta-llama/Meta-Llama-3-8B")
        print(adapter.info)
    """
    adapter = ModelAdapter(model_name, family=family)
    adapter.load()
    return adapter
