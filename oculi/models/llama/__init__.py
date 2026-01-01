"""
LLaMA Model Adapter
===================

Public adapter for LLaMA model family.

Exports:
    LlamaAttentionAdapter: Main adapter class
    
For attention anatomy details, see:
    oculi/models/llama/attention.py
"""

from oculi.models.llama.adapter import LlamaAttentionAdapter

__all__ = ["LlamaAttentionAdapter"]
