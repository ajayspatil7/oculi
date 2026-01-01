"""
Model Adapters (Private)
========================

Model-specific implementations of the ModelAdapter interface.

Each adapter handles:
- Model loading with correct configuration
- Hook point discovery for that model family
- GQA/MQA mapping
- Capture implementation

To add a new model:
1. Create new adapter file (e.g., phi_adapter.py)
2. Subclass ModelAdapter
3. Call register_adapter() at module level
"""

from oculi._private.adapters.llama import LlamaAdapter
# Future: from oculi._private.adapters.mistral import MistralAdapter
# Future: from oculi._private.adapters.qwen import QwenAdapter

__all__ = ["LlamaAdapter"]
