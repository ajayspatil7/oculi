#!/usr/bin/env python3
"""
EXP2b — Global Gain Analysis
=============================

Compute gain (sensitivity to Q scaling) for ALL heads across ALL layers.

This produces a complete gain map of the model, enabling:
- Stratification of heads into high/medium/low gain categories
- Principled selection of heads for subsequent experiments

Output:
- gain_summary_global.csv: One row per (layer, head) with gain metrics
- gain_heatmap.png: Layer × Head heatmap of gain values
- gain_distribution.png: Distribution of gain values

Usage:
    python experiments/exp2b_gain_global/compute_global_gain.py

Runtime estimate: ~30-60 minutes on A10G GPU
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_loader import load_long_context, load_from_shards
from src.metrics import (
    compute_attention_entropy,
    compute_max_attention_weight,
)


# Output directory - ALL results go here
OUTPUT_DIR = Path("experiments/exp2b_gain_global")


@dataclass
class HeadGainResult:
    """Gain result for a single head."""
    layer: int
    head: int
    gain_entropy: float
    gain_max_attn: float
    r_squared_entropy: float
    r_squared_max_attn: float
    is_monotonic_entropy: bool
    is_monotonic_max_attn: bool
    baseline_entropy: float
    baseline_max_attn: float


class GlobalGainProfiler:
    """
    Computes gain for all heads across all layers.
    
    For each (layer, head):
    - Scales Q by alpha values [0.5, 0.75, 1.0, 1.25, 1.5]
    - Measures entropy and max_attn
    - Computes gain = slope vs log(alpha)
    """
    
    def __init__(self, model):
        self.model = model
        self.config = model.config
        
        self.n_layers = self.config.num_hidden_layers
        self.n_heads = self.config.num_attention_heads
        self.n_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.hidden_size // self.n_heads
        
        self._hidden_states: Dict[int, torch.Tensor] = {}
        self.hooks: List = []
    
    def _get_hidden_hook(self, layer_idx: int):
        """
        Hook to capture pre-attention hidden states.
        
        We capture after input LayerNorm, matching inputs to QKV projection.
        """
        def hook_fn(module, input, output):
            self._hidden_states[layer_idx] = output.detach()
        return hook_fn
    
    def _compute_qkv_with_scaling(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        target_head: int,
        q_scale: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q, K with Q scaling for target head only.
        V is NEVER modified.
        """
        layer = self.model.model.layers[layer_idx]
        attn = layer.self_attn
        
        batch_size, seq_len, _ = hidden_states.shape
        
        with torch.no_grad():
            Q = attn.q_proj(hidden_states)
            K = attn.k_proj(hidden_states)
            
            Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
            
            # Scale only the target head's Q
            if q_scale != 1.0:
                Q[:, target_head, :, :] = Q[:, target_head, :, :] * q_scale
        
        return Q, K
    
    def _compute_attention_probs(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Compute attention probabilities with causal mask."""
        batch_size, n_heads, seq_len, head_dim = Q.shape
        _, n_kv_heads, _, _ = K.shape
        
        # Expand K for GQA
        n_rep = n_heads // n_kv_heads
        if n_rep > 1:
            K = K.repeat_interleave(n_rep, dim=1)
        
        scale = head_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        return torch.softmax(scores, dim=-1)
    
    def register_hooks(self):
        """Register hooks on all layers."""
        self.remove_hooks()
        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]
            hook = layer.input_layernorm.register_forward_hook(
                self._get_hidden_hook(layer_idx)
            )
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._hidden_states = {}
    
    def compute_metrics_for_head(
        self,
        input_ids: torch.Tensor,
        layer: int,
        head: int,
        alpha: float
    ) -> Tuple[float, float]:
        """
        Compute entropy and max_attn for a single (layer, head, alpha) combination.
        
        Returns:
            (mean_entropy, mean_max_attn)
        """
        # Get hidden states for this layer
        hidden_states = self._hidden_states[layer]
        
        # Compute scaled Q, K
        Q, K = self._compute_qkv_with_scaling(layer, hidden_states, head, alpha)
        
        # Compute attention probs
        attn_probs = self._compute_attention_probs(Q, K)
        
        # Extract target head
        target_attn = attn_probs[:, head:head+1, :, :]
        
        # Compute metrics
        entropy = compute_attention_entropy(target_attn, ignore_first_n=2)
        max_attn = compute_max_attention_weight(target_attn)
        
        # Aggregate
        entropy_np = entropy.squeeze().cpu().numpy()
        max_attn_np = max_attn.squeeze().cpu().numpy()
        
        valid = ~np.isnan(entropy_np)
        mean_entropy = float(np.mean(entropy_np[valid]))
        mean_max_attn = float(np.mean(max_attn_np[valid]))
        
        return mean_entropy, mean_max_attn
    
    def compute_gain_for_head(
        self,
        input_ids: torch.Tensor,
        layer: int,
        head: int,
        alphas: List[float]
    ) -> HeadGainResult:
        """
        Compute gain for a single head.
        
        Runs through all alpha values and fits linear regression.
        """
        entropies = []
        max_attns = []
        
        for alpha in alphas:
            entropy, max_attn = self.compute_metrics_for_head(input_ids, layer, head, alpha)
            entropies.append(entropy)
            max_attns.append(max_attn)
        
        # Convert to numpy
        entropies = np.array(entropies)
        max_attns = np.array(max_attns)
        log_alphas = np.log(np.array(alphas))
        
        # Linear regression for entropy
        slope_entropy, _, r_entropy, _, _ = stats.linregress(log_alphas, entropies)
        r_sq_entropy = r_entropy ** 2
        
        # Linear regression for max_attn
        slope_max_attn, _, r_max_attn, _, _ = stats.linregress(log_alphas, max_attns)
        r_sq_max_attn = r_max_attn ** 2
        
        # Check monotonicity
        # Entropy should decrease with scale (negative gain)
        is_mono_entropy = all(entropies[i] >= entropies[i+1] for i in range(len(entropies)-1))
        # Max attn should increase with scale (positive gain)
        is_mono_max_attn = all(max_attns[i] <= max_attns[i+1] for i in range(len(max_attns)-1))
        
        # Baseline = value at alpha=1.0
        baseline_idx = alphas.index(1.0) if 1.0 in alphas else len(alphas) // 2
        
        return HeadGainResult(
            layer=layer,
            head=head,
            gain_entropy=slope_entropy,
            gain_max_attn=slope_max_attn,
            r_squared_entropy=r_sq_entropy,
            r_squared_max_attn=r_sq_max_attn,
            is_monotonic_entropy=is_mono_entropy,
            is_monotonic_max_attn=is_mono_max_attn,
            baseline_entropy=entropies[baseline_idx],
            baseline_max_attn=max_attns[baseline_idx]
        )
    
    def run_global_gain_analysis(
        self,
        input_ids: torch.Tensor,
        alphas: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5],
        verbose: bool = True
    ) -> List[HeadGainResult]:
        """
        Run gain analysis for ALL heads across ALL layers.
        
        Args:
            input_ids: Input token IDs
            alphas: Scale factors
            verbose: Print progress
            
        Returns:
            List of HeadGainResult, one per (layer, head)
        """
        results = []
        
        # First, capture all hidden states with a single forward pass
        self.register_hooks()
        try:
            with torch.no_grad():
                _ = self.model(input_ids, use_cache=False)
            
            if verbose:
                print(f"\n{'='*70}")
                print("GLOBAL GAIN ANALYSIS")
                print(f"{'='*70}")
                print(f"Layers: {self.n_layers}")
                print(f"Heads per layer: {self.n_heads}")
                print(f"Total heads: {self.n_layers * self.n_heads}")
                print(f"Alphas: {alphas}")
                print(f"{'='*70}")
            
            # Progress bar for all heads
            total_heads = self.n_layers * self.n_heads
            pbar = tqdm(total=total_heads, desc="Computing gain", disable=not verbose)
            
            for layer in range(self.n_layers):
                for head in range(self.n_heads):
                    result = self.compute_gain_for_head(input_ids, layer, head, alphas)
                    results.append(result)
                    pbar.update(1)
                
                # Clear cache periodically
                if layer % 4 == 0:
                    torch.cuda.empty_cache()
            
            pbar.close()
            
        finally:
            self.remove_hooks()
        
        return results


def results_to_dataframe(results: List[HeadGainResult]) -> pd.DataFrame:
    """Convert results to DataFrame."""
    data = []
    for r in results:
        data.append({
            'layer': r.layer,
            'head': r.head,
            'gain_entropy': r.gain_entropy,
            'gain_max_attn': r.gain_max_attn,
            'r_squared_entropy': r.r_squared_entropy,
            'r_squared_max_attn': r.r_squared_max_attn,
            'is_monotonic_entropy': r.is_monotonic_entropy,
            'is_monotonic_max_attn': r.is_monotonic_max_attn,
            'baseline_entropy': r.baseline_entropy,
            'baseline_max_attn': r.baseline_max_attn
        })
    return pd.DataFrame(data)


def plot_gain_heatmap(df: pd.DataFrame, output_path: Path, metric: str = 'gain_max_attn'):
    """
    Plot gain as a layer × head heatmap.
    """
    n_layers = df['layer'].max() + 1
    n_heads = df['head'].max() + 1
    
    # Pivot to 2D array
    heatmap_data = df.pivot(index='layer', columns='head', values=metric).values
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto')
    
    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(f'Gain Heatmap: {metric}\n(Positive = increases with Q scaling)', 
                 fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{metric}', fontsize=11)
    
    # Tick labels
    ax.set_xticks(range(0, n_heads, 4))
    ax.set_yticks(range(0, n_layers, 4))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_gain_distribution(df: pd.DataFrame, output_path: Path):
    """
    Plot distribution of gain values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Entropy gain distribution
    ax1 = axes[0]
    ax1.hist(df['gain_entropy'], bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=df['gain_entropy'].median(), color='green', linestyle=':', linewidth=2, 
                label=f'Median: {df["gain_entropy"].median():.4f}')
    ax1.set_xlabel('Gain (Entropy)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Entropy Gain', fontsize=14)
    ax1.legend()
    
    # Max attn gain distribution
    ax2 = axes[1]
    ax2.hist(df['gain_max_attn'], bins=50, color='#e74c3c', edgecolor='white', alpha=0.8)
    ax2.axvline(x=0, color='blue', linestyle='--', linewidth=2)
    ax2.axvline(x=df['gain_max_attn'].median(), color='green', linestyle=':', linewidth=2,
                label=f'Median: {df["gain_max_attn"].median():.4f}')
    ax2.set_xlabel('Gain (Max Attn)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Max Attention Gain', fontsize=14)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_layer_summary(df: pd.DataFrame, output_path: Path):
    """
    Plot mean gain per layer.
    """
    layer_means = df.groupby('layer').agg({
        'gain_entropy': 'mean',
        'gain_max_attn': 'mean'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = layer_means['layer']
    ax.plot(x, layer_means['gain_entropy'], 'o-', color='#3498db', linewidth=2, 
            markersize=8, label='Entropy Gain')
    ax.plot(x, layer_means['gain_max_attn'], 's-', color='#e74c3c', linewidth=2,
            markersize=8, label='Max Attn Gain')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Gain', fontsize=12)
    ax.set_title('Mean Gain per Layer', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def identify_head_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize heads into high/medium/low gain.
    
    Uses absolute value of gain_max_attn for categorization.
    """
    df = df.copy()
    
    # Use absolute gain for ranking (we care about magnitude)
    df['abs_gain_max_attn'] = df['gain_max_attn'].abs()
    
    # Percentile thresholds
    p90 = df['abs_gain_max_attn'].quantile(0.90)
    p10 = df['abs_gain_max_attn'].quantile(0.10)
    
    def categorize(row):
        if row['abs_gain_max_attn'] >= p90:
            return 'high'
        elif row['abs_gain_max_attn'] <= p10:
            return 'low'
        else:
            return 'medium'
    
    df['gain_category'] = df.apply(categorize, axis=1)
    
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="EXP2b: Global Gain Analysis across all layers and heads"
    )
    parser.add_argument("--alphas", type=str, default="0.5,0.75,1.0,1.25,1.5",
                        help="Comma-separated scale factors")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Sample index to use")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Data directory")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Model name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("EXP2b — GLOBAL GAIN ANALYSIS")
    print("=" * 70)
    print("\nPurpose: Compute gain for ALL heads across ALL layers")
    print("         to enable principled head selection for subsequent experiments.")
    print("=" * 70)
    
    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    
    print(f"\n--- GPU ---")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print(f"\n--- Loading Model ---")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    load_time = time.time() - start_time
    print(f"  Model: {args.model}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Heads: {model.config.num_attention_heads}")
    print(f"  Load time: {load_time:.1f}s")
    
    # Load sample
    print(f"\n--- Loading Sample ---")
    samples = load_from_shards(args.data_dir, n_samples=args.sample_idx + 1, device="cuda")
    if samples and len(samples) > args.sample_idx:
        input_ids = samples[args.sample_idx]["input_ids"]
        print(f"  Source: Preprocessed shards")
    else:
        sample = load_long_context(tokenizer, target_length=512)
        input_ids = sample["input_ids"]
        print(f"  Source: Generated sample")
    print(f"  Tokens: {input_ids.shape[1]}")
    
    # Parse alphas
    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    print(f"  Alphas: {alphas}")
    
    # Run global gain analysis
    profiler = GlobalGainProfiler(model)
    
    analysis_start = time.time()
    results = profiler.run_global_gain_analysis(input_ids, alphas, verbose=True)
    analysis_time = time.time() - analysis_start
    
    print(f"\n  Analysis time: {analysis_time:.1f}s ({analysis_time/60:.1f} min)")
    
    # Convert to DataFrame
    df = results_to_dataframe(results)
    
    # Categorize heads
    df = identify_head_categories(df)
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    csv_path = OUTPUT_DIR / "gain_summary_global.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    # Generate plots
    print("\n--- Generating Plots ---")
    
    # Heatmap
    heatmap_path = OUTPUT_DIR / "gain_heatmap_max_attn.png"
    plot_gain_heatmap(df, heatmap_path, 'gain_max_attn')
    
    heatmap_entropy_path = OUTPUT_DIR / "gain_heatmap_entropy.png"
    plot_gain_heatmap(df, heatmap_entropy_path, 'gain_entropy')
    
    # Distribution
    dist_path = OUTPUT_DIR / "gain_distribution.png"
    plot_gain_distribution(df, dist_path)
    
    # Layer summary
    layer_path = OUTPUT_DIR / "gain_by_layer.png"
    plot_layer_summary(df, layer_path)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\n--- Gain (Max Attn) ---")
    print(f"  Mean: {df['gain_max_attn'].mean():.4f}")
    print(f"  Std: {df['gain_max_attn'].std():.4f}")
    print(f"  Min: {df['gain_max_attn'].min():.4f}")
    print(f"  Max: {df['gain_max_attn'].max():.4f}")
    
    print(f"\n--- Gain (Entropy) ---")
    print(f"  Mean: {df['gain_entropy'].mean():.4f}")
    print(f"  Std: {df['gain_entropy'].std():.4f}")
    print(f"  Min: {df['gain_entropy'].min():.4f}")
    print(f"  Max: {df['gain_entropy'].max():.4f}")
    
    print(f"\n--- Monotonicity ---")
    mono_entropy = df['is_monotonic_entropy'].sum()
    mono_max_attn = df['is_monotonic_max_attn'].sum()
    total = len(df)
    print(f"  Entropy monotonic: {mono_entropy}/{total} ({100*mono_entropy/total:.1f}%)")
    print(f"  Max attn monotonic: {mono_max_attn}/{total} ({100*mono_max_attn/total:.1f}%)")
    
    print(f"\n--- Head Categories ---")
    for cat in ['high', 'medium', 'low']:
        count = (df['gain_category'] == cat).sum()
        print(f"  {cat}: {count} heads")
    
    # Show top and bottom heads
    print("\n" + "=" * 70)
    print("TOP 10 HIGH-GAIN HEADS")
    print("=" * 70)
    top_heads = df.nlargest(10, 'abs_gain_max_attn')[['layer', 'head', 'gain_max_attn', 'gain_entropy']]
    print(top_heads.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("TOP 10 LOW-GAIN HEADS")
    print("=" * 70)
    bottom_heads = df.nsmallest(10, 'abs_gain_max_attn')[['layer', 'head', 'gain_max_attn', 'gain_entropy']]
    print(bottom_heads.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        if f.is_file():
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
