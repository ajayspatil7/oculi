"""
Visualization Script for Phase 1 Results
==========================================

Generates visualizations from Phase 1 experiment results:
1. Correlation heatmap (layers × heads)
2. Scatter plots (query norm vs entropy)
3. Distribution histograms
4. Null distribution comparison

Usage:
    python scripts/visualize.py --results-dir results --timestamp 20251222_103000
    python scripts/visualize.py --results-dir results --latest
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Phase 1 results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing experiment results")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Specific experiment timestamp")
    parser.add_argument("--latest", action="store_true",
                        help="Use the latest experiment results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures (default: results/figures)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Figure DPI")
    return parser.parse_args()


def find_latest_results(results_dir: Path) -> str:
    """Find the timestamp of the latest experiment."""
    summaries = list(results_dir.glob("summary_*.json"))
    if not summaries:
        raise FileNotFoundError(f"No summary files found in {results_dir}")
    
    # Sort by modification time
    latest = max(summaries, key=lambda p: p.stat().st_mtime)
    timestamp = latest.stem.replace("summary_", "")
    return timestamp


def load_results(results_dir: Path, timestamp: str):
    """Load all results for a given timestamp."""
    correlations_path = results_dir / f"correlations_{timestamp}.csv"
    matrices_path = results_dir / f"heatmap_matrices_{timestamp}.npz"
    raw_path = results_dir / f"raw_data_{timestamp}.pkl"
    summary_path = results_dir / f"summary_{timestamp}.json"
    
    df = pd.read_csv(correlations_path)
    matrices = np.load(matrices_path)
    
    with open(raw_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    return {
        'df': df,
        'pearson_matrix': matrices['pearson'],
        'spearman_matrix': matrices['spearman'],
        'q_norms': raw_data['q_norms'],
        'entropy': raw_data['entropy'],
        'summary': summary,
        'config': raw_data['config'],
    }


def plot_correlation_heatmap(
    matrix: np.ndarray,
    title: str,
    output_path: Path,
    dpi: int = 150
):
    """Plot correlation heatmap for layers vs heads."""
    n_layers, n_heads = matrix.shape
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use diverging colormap centered at 0
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    
    im = ax.imshow(
        matrix,
        cmap='RdBu_r',
        aspect='auto',
        vmin=-vmax,
        vmax=vmax
    )
    
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Correlation (r)', shrink=0.8)
    
    # Set ticks
    ax.set_xticks(np.arange(0, n_heads, 4))
    ax.set_yticks(np.arange(0, n_layers, 4))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_scatter_samples(
    q_norms: np.ndarray,
    entropy: np.ndarray,
    layer_head_pairs: list,
    output_path: Path,
    dpi: int = 150
):
    """Plot scatter plots for selected (layer, head) pairs."""
    n_pairs = len(layer_head_pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (layer, head) in enumerate(layer_head_pairs):
        ax = axes[idx]
        
        norms = q_norms[layer, head]
        ent = entropy[layer, head]
        
        # Remove NaN values
        valid = ~np.isnan(ent)
        norms = norms[valid]
        ent = ent[valid]
        
        ax.scatter(norms, ent, alpha=0.5, s=10)
        
        # Add trend line
        if len(norms) > 2:
            z = np.polyfit(norms, ent, 1)
            p = np.poly1d(z)
            x_line = np.linspace(norms.min(), norms.max(), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2)
        
        # Compute correlation for title
        from scipy import stats
        if len(norms) >= 4:
            r, p_val = stats.pearsonr(norms, ent)
            ax.set_title(f"Layer {layer}, Head {head}\nr = {r:.3f}", fontsize=11)
        else:
            ax.set_title(f"Layer {layer}, Head {head}", fontsize=11)
        
        ax.set_xlabel('Query Norm (‖Q‖₂)')
        ax.set_ylabel('Attention Entropy (H)')
    
    # Hide empty axes
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Query Norm vs Attention Entropy', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_distributions(
    q_norms: np.ndarray,
    entropy: np.ndarray,
    output_path: Path,
    dpi: int = 150
):
    """Plot distributions of query norms and entropy values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Flatten and remove NaN
    norms_flat = q_norms.flatten()
    entropy_flat = entropy.flatten()
    entropy_flat = entropy_flat[~np.isnan(entropy_flat)]
    
    # Query norm distribution
    ax = axes[0]
    ax.hist(norms_flat, bins=100, density=True, alpha=0.7, color='steelblue')
    ax.set_xlabel('Query Norm (‖Q‖₂)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Query Norm Distribution', fontsize=12, fontweight='bold')
    ax.axvline(norms_flat.mean(), color='red', linestyle='--', 
               label=f'Mean: {norms_flat.mean():.2f}')
    ax.legend()
    
    # Entropy distribution
    ax = axes[1]
    ax.hist(entropy_flat, bins=100, density=True, alpha=0.7, color='darkorange')
    ax.set_xlabel('Attention Entropy (H)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Attention Entropy Distribution', fontsize=12, fontweight='bold')
    ax.axvline(entropy_flat.mean(), color='red', linestyle='--',
               label=f'Mean: {entropy_flat.mean():.2f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_layer_summary(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150
):
    """Plot mean correlation by layer."""
    layer_stats = df.groupby('layer').agg({
        'pearson_r': ['mean', 'std'],
        'significant': 'sum'
    }).reset_index()
    layer_stats.columns = ['layer', 'mean_r', 'std_r', 'n_significant']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Mean correlation by layer
    ax = axes[0]
    ax.bar(layer_stats['layer'], layer_stats['mean_r'], 
           yerr=layer_stats['std_r'], alpha=0.7, capsize=2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='r = -0.5')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Mean Pearson r', fontsize=11)
    ax.set_title('Mean Correlation by Layer', fontsize=12, fontweight='bold')
    ax.legend()
    
    # Number of significant heads by layer
    ax = axes[1]
    n_heads = df.groupby('layer').size().iloc[0]
    ax.bar(layer_stats['layer'], layer_stats['n_significant'], alpha=0.7, color='green')
    ax.axhline(y=n_heads * 0.5, color='red', linestyle='--', alpha=0.5, 
               label=f'50% of heads ({n_heads//2})')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('# Significant Heads', fontsize=11)
    ax.set_title('Significant Correlations by Layer', fontsize=12, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_null_comparison(
    df: pd.DataFrame,
    summary: dict,
    output_path: Path,
    dpi: int = 150
):
    """Compare observed correlations with null distribution."""
    if 'randomization_control' not in summary:
        print("  Skipping null comparison (no control data)")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Observed |r| distribution
    observed_abs_r = df['pearson_r'].abs().dropna()
    ax.hist(observed_abs_r, bins=50, density=True, alpha=0.7, 
            color='steelblue', label='Observed |r|')
    
    # Null distribution (simulated from stored stats)
    null_stats = summary['randomization_control']
    null_mean = null_stats['abs_mean_shuffled_r']
    null_std = null_stats['std_shuffled_r']
    
    # Draw null distribution as a line
    x = np.linspace(0, 1, 100)
    null_dist = np.exp(-0.5 * ((x - null_mean) / null_std) ** 2) / (null_std * np.sqrt(2 * np.pi))
    ax.plot(x, null_dist, 'r-', linewidth=2, label=f'Null distribution (shuffled)')
    
    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Threshold (|r| = 0.5)')
    
    ax.set_xlabel('|Correlation|', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Observed vs Null Correlation Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def select_representative_heads(df: pd.DataFrame, n: int = 6) -> list:
    """Select representative (layer, head) pairs for scatter plots."""
    # Get pairs with strong correlations
    df_sorted = df.dropna().sort_values('pearson_r')
    
    # Mix of strongly negative, neutral, and strongly positive
    pairs = []
    
    # Top 2 most negative
    for _, row in df_sorted.head(2).iterrows():
        pairs.append((int(row['layer']), int(row['head'])))
    
    # Top 2 most positive
    for _, row in df_sorted.tail(2).iterrows():
        pairs.append((int(row['layer']), int(row['head'])))
    
    # 2 from middle (neutral)
    mid = len(df_sorted) // 2
    for _, row in df_sorted.iloc[mid:mid+2].iterrows():
        pairs.append((int(row['layer']), int(row['head'])))
    
    return pairs[:n]


def main():
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Determine timestamp
    if args.latest:
        timestamp = find_latest_results(results_dir)
        print(f"Using latest results: {timestamp}")
    elif args.timestamp:
        timestamp = args.timestamp
    else:
        timestamp = find_latest_results(results_dir)
        print(f"Using latest results: {timestamp}")
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading results from: {results_dir}")
    print(f"Saving figures to: {output_dir}")
    
    # Load results
    results = load_results(results_dir, timestamp)
    
    print("\nGenerating visualizations...")
    
    # 1. Correlation heatmaps
    print("\n1. Correlation heatmaps")
    plot_correlation_heatmap(
        results['pearson_matrix'],
        'Pearson Correlation: Query Norm vs Attention Entropy',
        output_dir / f"heatmap_pearson_{timestamp}.png",
        dpi=args.dpi
    )
    plot_correlation_heatmap(
        results['spearman_matrix'],
        'Spearman Correlation: Query Norm vs Attention Entropy',
        output_dir / f"heatmap_spearman_{timestamp}.png",
        dpi=args.dpi
    )
    
    # 2. Scatter plots
    print("\n2. Scatter plots")
    pairs = select_representative_heads(results['df'])
    plot_scatter_samples(
        results['q_norms'],
        results['entropy'],
        pairs,
        output_dir / f"scatter_samples_{timestamp}.png",
        dpi=args.dpi
    )
    
    # 3. Distributions
    print("\n3. Distribution histograms")
    plot_distributions(
        results['q_norms'],
        results['entropy'],
        output_dir / f"distributions_{timestamp}.png",
        dpi=args.dpi
    )
    
    # 4. Layer summary
    print("\n4. Layer summary")
    plot_layer_summary(
        results['df'],
        output_dir / f"layer_summary_{timestamp}.png",
        dpi=args.dpi
    )
    
    # 5. Null comparison
    print("\n5. Null distribution comparison")
    plot_null_comparison(
        results['df'],
        results['summary'],
        output_dir / f"null_comparison_{timestamp}.png",
        dpi=args.dpi
    )
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    main()
