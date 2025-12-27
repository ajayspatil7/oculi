"""
Data Module for MATS 10.0
==========================

Handles GSM8K loading and sycophancy prompt generation.
"""

from .gsm8k import (
    load_gsm8k,
    filter_multistep_problems,
    generate_wrong_answer,
    prepare_problem,
)
from .prompts import (
    format_control_prompt,
    format_sycophancy_prompt,
    CONTROL_TEMPLATE,
    SYCOPHANCY_TEMPLATE,
    NONSENSE_REASONS,
)

__all__ = [
    "load_gsm8k",
    "filter_multistep_problems",
    "generate_wrong_answer",
    "prepare_problem",
    "format_control_prompt",
    "format_sycophancy_prompt",
    "CONTROL_TEMPLATE",
    "SYCOPHANCY_TEMPLATE",
    "NONSENSE_REASONS",
]
