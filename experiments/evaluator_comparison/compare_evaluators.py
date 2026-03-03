"""Visual comparison of different evaluators on synthetic distributions.

Compares KL divergence and Hybrid evaluators on various test distributions
to understand their behavior and sensitivity.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pycmaetad.evaluator.KLDiv import UniformKLEvaluator1D, UniformKLEvaluator2D
from pycmaetad.evaluator.hybrid_uniform import HybridUniformEvaluator


def generate_1d_distributions(n_samples=5000):
    """Generate various 1D test distributions.
    
    Creates distributions with different coverage and uniformity properties
    to test evaluator sensitivity.
    
    Returns:
        dict: {name: samples} mapping
    """
    distributions = {}
    
    # Domain: [-π, π] (like dihedral angles)
    domain = (-np.pi, np.pi)
    
    # 1. Uniform (ideal)
    distributions['Uniform'] = np.random.uniform(domain[0], domain[1], n_samples)
    
    # 2. Gaussian (concentrated in middle)
    distributions['Gaussian'] = np.random.normal(0, 1.0, n_samples)
    distributions['Gaussian'] = np.clip(distributions['Gaussian'], domain[0], domain[1])
    
    # 3. Bimodal (two peaks)
    mode1 = np.random.normal(-1.5, 0.4, n_samples // 2)
    mode2 = np.random.normal(1.5, 0.4, n_samples // 2)
    distributions['Bimodal'] = np.concatenate([mode1, mode2])
    distributions['Bimodal'] = np.clip(distributions['Bimodal'], domain[0], domain[1])
    
    # 4. Sparse uniform (samples only from 70% of space)
    # Sample only from 70% of the domain
    domain_width = domain[1] - domain[0]
    sparse_width = domain_width * 0.7
    sparse_min = domain[0]
    sparse_max = sparse_min + sparse_width
    distributions['Sparse (70%)'] = np.random.uniform(sparse_min, sparse_max, n_samples)
    
    # 5. Edge-biased (more samples at boundaries)
    distributions['Edge-biased'] = np.concatenate([
        np.random.uniform(domain[0], -2, n_samples // 3),
        np.random.uniform(-1, 1, n_samples // 3),
        np.random.uniform(2, domain[1], n_samples // 3)
    ])
    
    # 6. Very sparse (samples only from 40% of space)
    domain_width = domain[1] - domain[0]
    sparse_width = domain_width * 0.4
    sparse_min = domain[0]
    sparse_max = sparse_min + sparse_width
    distributions['Very Sparse (40%)'] = np.random.uniform(sparse_min, sparse_max, n_samples)
    
    return distributions, domain


def generate_2d_distributions(n_samples=5000):
    """Generate various 2D test distributions.
    
    Returns:
        dict: {name: (x, y) samples} mapping
    """
    distributions = {}
    
    # Domain: [-1.5, 1.5] x [-0.5, 2.5] (like Muller-Brown)
    x_range = (-1.5, 1.5)
    y_range = (-0.5, 2.5)
    
    # 1. Uniform (ideal)
    x_uniform = np.random.uniform(x_range[0], x_range[1], n_samples)
    y_uniform = np.random.uniform(y_range[0], y_range[1], n_samples)
    distributions['Uniform'] = np.column_stack([x_uniform, y_uniform])
    
    # 2. Gaussian cluster (concentrated in center)
    x_gauss = np.random.normal(0, 0.5, n_samples)
    y_gauss = np.random.normal(1.0, 0.5, n_samples)
    distributions['Gaussian Cluster'] = np.column_stack([x_gauss, y_gauss])
    distributions['Gaussian Cluster'] = np.clip(
        distributions['Gaussian Cluster'],
        [x_range[0], y_range[0]],
        [x_range[1], y_range[1]]
    )
    
    # 3. Three wells (three Gaussian clusters)
    x_well1 = np.random.normal(-0.8, 0.3, n_samples // 3)
    y_well1 = np.random.normal(0.5, 0.3, n_samples // 3)
    x_well2 = np.random.normal(0.8, 0.3, n_samples // 3)
    y_well2 = np.random.normal(0.5, 0.3, n_samples // 3)
    x_well3 = np.random.normal(0, 0.3, n_samples // 3)
    y_well3 = np.random.normal(1.8, 0.3, n_samples // 3)
    distributions['Three Wells'] = np.column_stack([
        np.concatenate([x_well1, x_well2, x_well3]),
        np.concatenate([y_well1, y_well2, y_well3])
    ])
    
    # 4. Sparse uniform (70% of space covered)
    # Sample only from 70% of each dimension
    x_width = x_range[1] - x_range[0]
    y_width = y_range[1] - y_range[0]
    x_sparse = np.random.uniform(x_range[0], x_range[0] + 0.7 * x_width, n_samples)
    y_sparse = np.random.uniform(y_range[0], y_range[0] + 0.7 * y_width, n_samples)
    distributions['Sparse (70%)'] = np.column_stack([x_sparse, y_sparse])
    
    # 5. Corner-biased (more samples in corners)
    corners = []
    for _ in range(n_samples // 4):
        corner = np.random.choice(4)
        if corner == 0:  # Bottom-left
            x = np.random.uniform(x_range[0], x_range[0] + 0.8, 1)
            y = np.random.uniform(y_range[0], y_range[0] + 0.8, 1)
        elif corner == 1:  # Bottom-right
            x = np.random.uniform(x_range[1] - 0.8, x_range[1], 1)
            y = np.random.uniform(y_range[0], y_range[0] + 0.8, 1)
        elif corner == 2:  # Top-left
            x = np.random.uniform(x_range[0], x_range[0] + 0.8, 1)
            y = np.random.uniform(y_range[1] - 0.8, y_range[1], 1)
        else:  # Top-right
            x = np.random.uniform(x_range[1] - 0.8, x_range[1], 1)
            y = np.random.uniform(y_range[1] - 0.8, y_range[1], 1)
        corners.append([x[0], y[0]])
    distributions['Corner-biased'] = np.array(corners)
    
    # 6. Very sparse (40% of space covered)
    x_width = x_range[1] - x_range[0]
    y_width = y_range[1] - y_range[0]
    x_vsparse = np.random.uniform(x_range[0], x_range[0] + 0.4 * x_width, n_samples)
    y_vsparse = np.random.uniform(y_range[0], y_range[0] + 0.4 * y_width, n_samples)
    distributions['Very Sparse (40%)'] = np.column_stack([x_vsparse, y_vsparse])
    
    return distributions, x_range, y_range


def evaluate_1d_distributions():
    """Compare 1D evaluators on test distributions."""
    print("=" * 70)
    print("1D EVALUATOR COMPARISON")
    print("=" * 70)
    
    # Generate distributions
    distributions, domain = generate_1d_distributions()
    
    # Create evaluators
    n_bins = 50
    bin_edges = np.linspace(domain[0], domain[1], n_bins + 1)
    
    kl_eval = UniformKLEvaluator1D(bin_edges)
    hybrid_light = HybridUniformEvaluator.from_ranges(
        domain, n_bins=n_bins,
        kl_weight=1.0, coverage_weight=1.0, entropy_weight=0.0
    )
    hybrid_medium = HybridUniformEvaluator.from_ranges(
        domain, n_bins=n_bins,
        kl_weight=1.0, coverage_weight=5.0, entropy_weight=0.0
    )
    hybrid_heavy = HybridUniformEvaluator.from_ranges(
        domain, n_bins=n_bins,
        kl_weight=1.0, coverage_weight=20.0, entropy_weight=0.0  # Penalize gaps heavily
    )
    
    # Evaluate
    results = {}
    for name, samples in distributions.items():
        kl_score = kl_eval.evaluate(samples)
        hybrid_light_score = hybrid_light.evaluate(samples)
        hybrid_medium_score = hybrid_medium.evaluate(samples)
        hybrid_heavy_score = hybrid_heavy.evaluate(samples)
        results[name] = {
            'KL': kl_score,
            'Hybrid (1.0)': hybrid_light_score,
            'Hybrid (5.0)': hybrid_medium_score,
            'Hybrid (20.0)': hybrid_heavy_score,
            'samples': samples
        }
    
    # Print results
    print(f"\n{'Distribution':<20} {'KL Div':<12} {'Hybrid(1.0)':<12} {'Hybrid(5.0)':<12} {'Hybrid(20.0)':<12}")
    print("-" * 80)
    for name, scores in results.items():
        print(f"{name:<20} {scores['KL']:>11.4f} {scores['Hybrid (1.0)']:>11.4f} {scores['Hybrid (5.0)']:>11.4f} {scores['Hybrid (20.0)']:>11.4f}")
    
    # Plot
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, len(distributions), hspace=0.3, wspace=0.3)
    
    for i, (name, data) in enumerate(results.items()):
        samples = data['samples']
        
        # Histogram
        ax_hist = fig.add_subplot(gs[0, i])
        ax_hist.hist(samples, bins=bin_edges, density=True, alpha=0.7, edgecolor='black')
        ax_hist.axhline(1/(domain[1]-domain[0]), color='r', linestyle='--', 
                        label='Ideal uniform', linewidth=2)
        ax_hist.set_title(name, fontsize=10, fontweight='bold')
        ax_hist.set_ylabel('Density' if i == 0 else '')
        if i == 0:
            ax_hist.legend(fontsize=8)
        
        # Scores comparison
        ax_scores = fig.add_subplot(gs[1, i])
        evaluators = ['KL', 'H(1.0)', 'H(5.0)', 'H(20.0)']
        scores_list = [data['KL'], data['Hybrid (1.0)'], data['Hybrid (5.0)'], data['Hybrid (20.0)']]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax_scores.bar(range(len(evaluators)), scores_list, color=colors, alpha=0.7)
        ax_scores.set_xticks(range(len(evaluators)))
        ax_scores.set_xticklabels(evaluators, rotation=45, ha='right', fontsize=7)
        ax_scores.set_ylabel('Score' if i == 0 else '')
        ax_scores.set_title(f'Scores', fontsize=9)
        
        # Add score values on bars
        for bar, score in zip(bars, scores_list):
            height = bar.get_height()
            ax_scores.text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.2f}', ha='center', va='bottom', fontsize=7)
        
        # Coverage analysis
        ax_cov = fig.add_subplot(gs[2, i])
        hist_counts, _ = np.histogram(samples, bins=bin_edges)
        occupied_bins = np.sum(hist_counts > 0)
        coverage = occupied_bins / n_bins
        
        ax_cov.bar([0, 1], [coverage, 1-coverage], color=['green', 'red'], alpha=0.7)
        ax_cov.set_xticks([0, 1])
        ax_cov.set_xticklabels(['Sampled', 'Empty'], fontsize=8)
        ax_cov.set_ylabel('Fraction' if i == 0 else '')
        ax_cov.set_ylim([0, 1])
        ax_cov.set_title(f'Coverage: {coverage:.1%}', fontsize=9)
        # Position label inside bar if coverage is high to avoid overlap with title
        if coverage > 0.9:
            ax_cov.text(0, coverage - 0.08, f'{coverage:.1%}', ha='center', va='top', fontsize=8, color='white', fontweight='bold')
        else:
            ax_cov.text(0, coverage + 0.05, f'{coverage:.1%}', ha='center', fontsize=8)
    
    plt.suptitle('1D Evaluator Comparison', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig('evaluator_comparison_1d.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: evaluator_comparison_1d.png")
    
    return results


def evaluate_2d_distributions():
    """Compare 2D evaluators on test distributions."""
    print("\n" + "=" * 70)
    print("2D EVALUATOR COMPARISON")
    print("=" * 70)
    
    # Generate distributions
    distributions, x_range, y_range = generate_2d_distributions()
    
    # Create evaluators
    n_bins = 30
    x_edges = np.linspace(x_range[0], x_range[1], n_bins + 1)
    y_edges = np.linspace(y_range[0], y_range[1], n_bins + 1)
    
    kl_eval = UniformKLEvaluator2D((x_edges, y_edges))
    hybrid_eval = HybridUniformEvaluator.from_ranges(
        (x_range, y_range), n_bins=n_bins,
        kl_weight=1.0, coverage_weight=0.5, entropy_weight=0.0
    )
    hybrid_heavy = HybridUniformEvaluator.from_ranges(
        (x_range, y_range), n_bins=n_bins,
        kl_weight=1.0, coverage_weight=2.0, entropy_weight=0.0
    )
    
    # Evaluate
    results = {}
    for name, samples in distributions.items():
        kl_score = kl_eval.evaluate(samples)
        hybrid_score = hybrid_eval.evaluate(samples)
        hybrid_heavy_score = hybrid_heavy.evaluate(samples)
        results[name] = {
            'KL': kl_score,
            'Hybrid (0.5)': hybrid_score,
            'Hybrid (2.0)': hybrid_heavy_score,
            'samples': samples
        }
    
    # Print results
    print(f"\n{'Distribution':<20} {'KL Div':<12} {'Hybrid(0.5)':<12} {'Hybrid(2.0)':<12}")
    print("-" * 70)
    for name, scores in results.items():
        print(f"{name:<20} {scores['KL']:>11.4f} {scores['Hybrid (0.5)']:>11.4f} {scores['Hybrid (2.0)']:>11.4f}")
    
    # Plot
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(3, len(distributions), hspace=0.35, wspace=0.3)
    
    for i, (name, data) in enumerate(results.items()):
        samples = data['samples']
        
        # 2D histogram
        ax_hist = fig.add_subplot(gs[0, i])
        hist, xedges, yedges = np.histogram2d(
            samples[:, 0], samples[:, 1],
            bins=(x_edges, y_edges)
        )
        im = ax_hist.imshow(
            hist.T, origin='lower', aspect='auto',
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            cmap='viridis', interpolation='nearest'
        )
        ax_hist.set_title(name, fontsize=10, fontweight='bold')
        ax_hist.set_xlabel('x' if i == len(distributions)//2 else '')
        ax_hist.set_ylabel('y' if i == 0 else '')
        plt.colorbar(im, ax=ax_hist, fraction=0.046, pad=0.04)
        
        # Scores comparison
        ax_scores = fig.add_subplot(gs[1, i])
        evaluators = ['KL', 'Hybrid (0.5)', 'Hybrid (2.0)']
        scores_list = [data['KL'], data['Hybrid (0.5)'], data['Hybrid (2.0)']]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax_scores.bar(range(len(evaluators)), scores_list, color=colors, alpha=0.7)
        ax_scores.set_xticks(range(len(evaluators)))
        ax_scores.set_xticklabels(evaluators, rotation=45, ha='right', fontsize=8)
        ax_scores.set_ylabel('Score' if i == 0 else '')
        ax_scores.set_title(f'Scores', fontsize=9)
        
        # Add score values on bars
        for bar, score in zip(bars, scores_list):
            height = bar.get_height()
            ax_scores.text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.2f}', ha='center', va='bottom', fontsize=7)
        
        # Coverage analysis
        ax_cov = fig.add_subplot(gs[2, i])
        hist_2d, _, _ = np.histogram2d(
            samples[:, 0], samples[:, 1],
            bins=(x_edges, y_edges)
        )
        total_bins = n_bins * n_bins
        occupied_bins = np.sum(hist_2d > 0)
        coverage = occupied_bins / total_bins
        
        ax_cov.bar([0, 1], [coverage, 1-coverage], color=['green', 'red'], alpha=0.7)
        ax_cov.set_xticks([0, 1])
        ax_cov.set_xticklabels(['Sampled', 'Empty'], fontsize=8)
        ax_cov.set_ylabel('Fraction' if i == 0 else '')
        ax_cov.set_ylim([0, 1])
        ax_cov.set_title(f'Coverage: {coverage:.1%}', fontsize=9)
        # Position label inside bar if coverage is high to avoid overlap with title
        if coverage > 0.9:
            ax_cov.text(0, coverage - 0.08, f'{coverage:.1%}', ha='center', va='top', fontsize=8, color='white', fontweight='bold')
        else:
            ax_cov.text(0, coverage + 0.05, f'{coverage:.1%}', ha='center', fontsize=8)
    
    plt.suptitle('2D Evaluator Comparison', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig('evaluator_comparison_2d.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: evaluator_comparison_2d.png")
    
    return results


def main():
    """Run full evaluator comparison."""
    print("\n" + "=" * 70)
    print("EVALUATOR COMPARISON TOOL")
    print("Comparing KL Divergence and Hybrid evaluators on synthetic distributions")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run comparisons
    results_1d = evaluate_1d_distributions()
    results_2d = evaluate_2d_distributions()
    
    print("\n" + "=" * 70)
    print("INSIGHTS")
    print("=" * 70)
    print("""
Key Observations:

1. KL Divergence:
   - Focuses ONLY on evenness of sampled bins
   - Does NOT penalize missing/empty bins
   - Sparse but uniform sampling can score well
   - Good for detecting uneven sampling

2. Hybrid (coverage_weight=1.0):
   - Light coverage penalty
   - Minimal impact on well-sampled distributions
   - Sparse 40%: +1.5% penalty

3. Hybrid (coverage_weight=5.0):
   - Moderate coverage penalty
   - Balances evenness + coverage
   - Sparse 40%: +7.4% penalty

4. Hybrid (coverage_weight=20.0):
   - Heavy coverage penalty
   - Strongly penalizes incomplete sampling
   - Sparse 40%: +30% penalty!
   - Best for ensuring thorough exploration

Coverage Penalty Formula:
   penalty = weight × (target_coverage - actual_coverage)²
   
   For Sparse 40% with 90% target:
   - Gap: 0.90 - 0.40 = 0.50
   - Base penalty: (0.50)² = 0.25
   - With weight=1.0:  0.25 × 1  = 0.25  → ~1.5% of KL score
   - With weight=5.0:  0.25 × 5  = 1.25  → ~7.4% of KL score 
   - With weight=20.0: 0.25 × 20 = 5.00  → ~30% of KL score

2D shows even larger effects (more bins → larger gaps).

Recommendation:
- For quick optimization: Use weight=1.0-5.0
- For balanced results: Use weight=5.0-10.0 
- For thorough exploration: Use weight=15.0-20.0+
- Adjust based on number of bins and coverage target
""")
    
    print("\n✓ Analysis complete! Check the generated PNG files.")


if __name__ == '__main__':
    main()
