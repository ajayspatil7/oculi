#!/usr/bin/env python3
"""
Experiment Configuration Info Extractor
========================================

Extracts and displays configuration details about the attention metrics experiment.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import compute_attention_entropy, compute_effective_attention_span


def extract_config_info():
    """Extract configuration information from the codebase."""
    
    print("=" * 70)
    print("EXPERIMENT CONFIGURATION INFORMATION")
    print("=" * 70)
    
    # Model
    print("\n📦 MODEL")
    print("-" * 70)
    print("  Name: meta-llama/Meta-Llama-3-8B")
    print("  Type: Causal Language Model")
    print("  Architecture: Llama-3 (Grouped Query Attention)")
    print("  Layers: 32")
    print("  Attention Heads: 32 query heads, 8 key-value heads (GQA)")
    print("  Head Dimension: 128")
    print("  Precision: float16")
    
    # Context length
    print("\n📏 CONTEXT LENGTH")
    print("-" * 70)
    print("  Default: 512 tokens")
    print("  Configurable via: --context-length argument")
    print("  Note: Must match preprocessed shard sequence length")
    
    # Attention implementation
    print("\n🔍 ATTENTION IMPLEMENTATION")
    print("-" * 70)
    print("  Method: Manual recomputation (not FlashAttention)")
    print("  Formula: softmax(Q @ K^T / sqrt(d_k))")
    print("  Pre/Post-RoPE: PRE-RoPE (before rotary position embeddings)")
    print("  Rationale:")
    print("    - FlashAttention doesn't expose attention weights")
    print("    - Pre-RoPE captures position-agnostic query geometry")
    print("    - RoPE preserves norms (rotation), so ||Q|| unchanged")
    print("  Masking: Causal (autoregressive)")
    
    # Entropy computation
    print("\n🎲 ENTROPY COMPUTATION")
    print("-" * 70)
    
    # Read entropy function to get ignore_first_n default
    import inspect
    entropy_sig = inspect.signature(compute_attention_entropy)
    ignore_first_n = entropy_sig.parameters['ignore_first_n'].default
    eps = entropy_sig.parameters['eps'].default
    
    print(f"  Formula: H = -Σ p_i · log(p_i)")
    print(f"  Computed over: Valid attention weights (causal mask applied)")
    print(f"  Precision: float32 (for numerical stability)")
    print(f"  Epsilon: {eps} (for log stability)")
    print(f"  Ignored tokens: First {ignore_first_n} tokens (insufficient context)")
    print(f"  NaN handling: Early tokens set to NaN")
    
    # Query normalization
    print("\n📐 QUERY NORMALIZATION")
    print("-" * 70)
    print("  Norm type: L2 norm (Euclidean)")
    print("  Formula: ||Q|| = sqrt(Σ Q_i²)")
    print("  Applied to: Query vectors per (layer, head, token)")
    print("  Dimension: Computed along head_dim (128)")
    print("  Space: Pre-RoPE query space")
    
    # k_eff threshold
    print("\n📊 EFFECTIVE ATTENTION SPAN (k_eff)")
    print("-" * 70)
    
    # Read k_eff function to get threshold default
    keff_sig = inspect.signature(compute_effective_attention_span)
    threshold = keff_sig.parameters['threshold'].default
    
    print(f"  Threshold: {threshold * 100:.0f}% cumulative attention mass")
    print(f"  Definition: Minimum number of top-weighted keys to reach threshold")
    print(f"  Method:")
    print(f"    1. Sort attention weights (descending)")
    print(f"    2. Compute cumulative sum")
    print(f"    3. Find first position where cumsum >= {threshold}")
    print(f"  Interpretation: How many tokens receive 90% of attention")
    
    # Permutation control (removed)
    print("\n🎯 PERMUTATION CONTROL")
    print("-" * 70)
    print("  Status: REMOVED from current implementation")
    print("  Reason: Script focuses on data capture, not statistical testing")
    print("  Previous implementation:")
    print("    - Shuffled entropy values randomly")
    print("    - Recomputed correlations on shuffled data")
    print("    - Default: 100 permutations")
    print("    - Purpose: Verify correlations weren't spurious")
    print("  Note: Not needed for raw metrics capture")
    
    # Data capture specifics
    print("\n💾 DATA CAPTURE DETAILS")
    print("-" * 70)
    print("  Metrics per token:")
    print("    - query_norm: L2 norm of Q vector")
    print("    - entropy: Attention distribution entropy")
    print("    - max_attn: Maximum attention weight")
    print("    - k_eff: Effective attention span (90% threshold)")
    print("  Granularity: Per (sample, layer, head, token)")
    print("  Output: CSV format (incremental writes)")
    print("  Expected rows for 64 samples: 33,554,432")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("This experiment captures PRE-RoPE attention metrics from Llama-3-8B")
    print("using manually recomputed attention (not FlashAttention) to access")
    print("intermediate attention weights. All metrics are computed at per-token")
    print("granularity and saved to CSV for downstream analysis.")
    print("=" * 70)


if __name__ == "__main__":
    extract_config_info()
