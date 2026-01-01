"""
Mock Models for Testing
=======================

Lightweight model implementations for testing Oculi without GPU/large memory.
"""

from tests.mocks.mock_llama import (
    MockLlamaConfig,
    MockLlamaForCausalLM,
    MockLlamaAdapter,
    MockTokenizer,
    create_mock_adapter,
)

__all__ = [
    "MockLlamaConfig",
    "MockLlamaForCausalLM", 
    "MockLlamaAdapter",
    "MockTokenizer",
    "create_mock_adapter",
]
