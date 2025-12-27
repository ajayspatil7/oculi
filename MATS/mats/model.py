"""
Model Loading for MATS 10.0
============================

Loads Qwen2.5-7B-Instruct via TransformerLens with GQA-aware configuration.

TransformerLens provides:
- hook_rot_q / hook_rot_k: Post-RoPE Q/K tensors (critical for accurate intervention)
- Automatic GQA handling with correct head dimension broadcasting
- Pattern caching for efficient entropy computation
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import torch

try:
    from transformer_lens import HookedTransformer
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False
    HookedTransformer = None


@dataclass
class ModelInfo:
    """Model architecture information for Qwen2.5-7B-Instruct."""
    name: str
    n_layers: int
    n_q_heads: int
    n_kv_heads: int
    gqa_ratio: int
    head_dim: int
    d_model: int
    
    @property
    def is_gqa(self) -> bool:
        return self.n_q_heads != self.n_kv_heads
    
    def __repr__(self) -> str:
        return (
            f"ModelInfo({self.name})\n"
            f"  Layers: {self.n_layers}\n"
            f"  Q-Heads: {self.n_q_heads}, KV-Heads: {self.n_kv_heads}\n"
            f"  GQA Ratio: {self.gqa_ratio}:1\n"
            f"  Head Dim: {self.head_dim}, D-Model: {self.d_model}"
        )


# Qwen2.5-7B-Instruct architecture (LOCKED)
QWEN_7B_INFO = ModelInfo(
    name="Qwen/Qwen2.5-7B-Instruct",
    n_layers=28,
    n_q_heads=28,
    n_kv_heads=4,
    gqa_ratio=7,  # 28 / 4 = 7 Q-heads per KV-head
    head_dim=128,
    d_model=3584,
)


class ModelWrapper:
    """
    Wrapper for TransformerLens HookedTransformer.
    
    Provides consistent interface and GQA-aware hook management.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.model: Optional[HookedTransformer] = None
        self.info: ModelInfo = QWEN_7B_INFO
        
    def load(self) -> "ModelWrapper":
        """Load model via TransformerLens."""
        if not TRANSFORMER_LENS_AVAILABLE:
            raise ImportError(
                "TransformerLens not installed. Run: pip install transformer_lens"
            )
        
        print(f"Loading model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {self.dtype}")
        
        self.model = HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Verify architecture matches expected
        self._verify_architecture()
        
        print(f"\n{self.info}")
        print("✅ Model loaded successfully")
        
        return self
    
    def _verify_architecture(self):
        """Verify loaded model matches expected Qwen2.5-7B architecture."""
        cfg = self.model.cfg
        
        assert cfg.n_layers == self.info.n_layers, \
            f"Expected {self.info.n_layers} layers, got {cfg.n_layers}"
        assert cfg.n_heads == self.info.n_q_heads, \
            f"Expected {self.info.n_q_heads} heads, got {cfg.n_heads}"
        
        print(f"  ✓ Architecture verified: {cfg.n_layers}L x {cfg.n_heads}H")
    
    def run_with_cache(
        self,
        prompt: str,
        prepend_bos: bool = True,
        return_type: str = "logits",
    ):
        """
        Run forward pass and return logits + activation cache.
        
        Args:
            prompt: Input text
            prepend_bos: Whether to prepend BOS token
            return_type: What to return ("logits", "loss", etc.)
            
        Returns:
            Tuple of (output, cache)
            - cache contains attention patterns at cache["pattern", layer]
        """
        return self.model.run_with_cache(
            prompt,
            prepend_bos=prepend_bos,
            return_type=return_type,
        )
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate text continuation.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample (vs greedy)
            
        Returns:
            Generated text (including prompt)
        """
        return self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            **kwargs,
        )
    
    def add_hook(self, hook_name: str, hook_fn, level: str = None):
        """Add a hook to the model."""
        self.model.add_hook(hook_name, hook_fn, level=level)
        
    def reset_hooks(self):
        """Remove all hooks."""
        self.model.reset_hooks()
        
    @property
    def tokenizer(self):
        """Access tokenizer."""
        return self.model.tokenizer


def load_model(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> ModelWrapper:
    """
    Load model and return wrapper.
    
    Usage:
        model = load_model()
        logits, cache = model.run_with_cache("Hello, world!")
        pattern = cache["pattern", 15]  # Layer 15 attention pattern
    """
    wrapper = ModelWrapper(model_name, device, dtype)
    wrapper.load()
    return wrapper


def get_model_info() -> ModelInfo:
    """Get Qwen2.5-7B-Instruct architecture info without loading."""
    return QWEN_7B_INFO
