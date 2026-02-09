"""Convergence and optimization progress visualization."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_convergence(result, output_path=None, show=False):
    """Plot CMA-ES optimization convergence.
    
    Args:
        result: Optimizer result dictionary with 'history' key
        output_path: Path to save plot (if None, returns figure)
        show: Whether to display plot interactively
        
    Returns:
        matplotlib Figure object
    """
    history = result['history']
    gens = [h['generation'] for h in history]
    best = np.array([h['best_score'] for h in history])
    mean = np.array([h['mean_score'] for h in history])
    std = np.array([h['std_score'] for h in history])
    
    # Determine cap value for display (handle outliers/penalties)
    valid_scores = best[np.isfinite(best) & (best < 1e4)]
    if len(valid_scores) > 3:
        cap_value = np.percentile(valid_scores, 95) * 1.5
    else:
        cap_value = np.max(best[np.isfinite(best)]) if np.any(np.isfinite(best)) else 10.0
    
    # Apply cap
    best_capped = np.clip(best, None, cap_value)
    mean_capped = np.clip(mean, None, cap_value)
    std_capped = np.clip(std, None, cap_value)
    
    lower_bound = np.clip(mean_capped - std_capped, 0, cap_value)
    upper_bound = np.clip(mean_capped + std_capped, 0, cap_value)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(gens, best_capped, 'b-o', label='Best', linewidth=2, markersize=6)
    ax.plot(gens, mean_capped, 'r--s', label='Mean', linewidth=2, markersize=5)
    ax.fill_between(gens, lower_bound, upper_bound,
                     alpha=0.2, color='red', label='±1 std')
    
    # Mark capped values
    n_capped = np.sum((best > cap_value) | (mean > cap_value))
    if n_capped > 0:
        ax.axhline(cap_value, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(0.02, 0.98, f'Display capped at {cap_value:.1f}\n({n_capped} outliers hidden)', 
                transform=ax.transAxes, verticalalignment='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Score (lower = better)', fontsize=12)
    ax.set_title('CMA-ES Optimization Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Use log scale if scores vary by orders of magnitude
    if np.max(best_capped) / np.min(best_capped[best_capped > 0]) > 10:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {Path(output_path).name}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_sigma_evolution(result, output_path=None, show=False):
    """Plot CMA-ES sigma (step size) evolution over generations.
    
    Sigma represents the step size or exploration radius of CMA-ES.
    As the algorithm converges, sigma decreases, indicating the search
    is focusing on a smaller region.
    
    Args:
        result: Optimizer result dictionary with 'history' key
        output_path: Path to save plot (if None, returns figure)
        show: Whether to display plot interactively
        
    Returns:
        matplotlib Figure object
    """
    history = result['history']
    
    # Extract sigma values
    gens = [h['generation'] for h in history]
    sigmas = [h.get('cma_sigma', None) for h in history]
    
    # Filter out None values (in case old checkpoints don't have sigma)
    valid_data = [(g, s) for g, s in zip(gens, sigmas) if s is not None]
    
    if not valid_data:
        print("Warning: No sigma data found in history")
        return None
    
    gens, sigmas = zip(*valid_data)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(gens, sigmas, 'g-o', label='Sigma (step size)', linewidth=2, markersize=6)
    
    # Add horizontal line at initial sigma for reference
    if len(sigmas) > 0:
        ax.axhline(sigmas[0], color='gray', linestyle=':', linewidth=1.5, alpha=0.5,
                   label=f'Initial σ = {sigmas[0]:.4f}')
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Sigma (σ)', fontsize=12)
    ax.set_title('CMA-ES Step Size Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Use log scale if sigma changes significantly
    if len(sigmas) > 1 and max(sigmas) / min(sigmas) > 5:
        ax.set_yscale('log')
    
    # Add annotation about convergence
    final_sigma = sigmas[-1]
    initial_sigma = sigmas[0]
    reduction = (1 - final_sigma / initial_sigma) * 100
    
    ax.text(0.98, 0.98, 
            f'Sigma reduction: {reduction:.1f}%\n'
            f'Initial: {initial_sigma:.4f}\n'
            f'Final: {final_sigma:.4f}',
            transform=ax.transAxes, 
            verticalalignment='top',
            horizontalalignment='right',
            fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {Path(output_path).name}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_parameter_evolution(result, bias, output_path=None, show=False):
    """Plot evolution of bias parameters over generations.
    
    Args:
        result: Optimizer result dictionary
        bias: Bias object (for parameter names/bounds)
        output_path: Path to save plot
        show: Whether to display plot
        
    Returns:
        matplotlib Figure object
    """
    history = result['history']
    n_params = bias.get_parameter_space_size()
    
    # Extract parameter evolution
    gens = [h['generation'] for h in history]
    best_params = np.array([h['best_solution'] for h in history])
    
    # Create subplots
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    bounds = bias.get_parameter_bounds()
    param_names = [f'param_{i}' for i in range(n_params)]
    
    for i in range(n_params):
        ax = axes[i]
        ax.plot(gens, best_params[:, i], 'b-o', linewidth=2, markersize=5)
        ax.axhline(bounds[i, 0], color='red', linestyle='--', alpha=0.5, label='Bounds')
        ax.axhline(bounds[i, 1], color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Value')
        ax.set_title(param_names[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    
    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle('Parameter Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {Path(output_path).name}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_convergence_diagnostics(result, output_path=None, show=False):
    """Plot diagnostic view combining convergence, sigma, and score variance.
    
    This helps diagnose premature convergence issues by showing:
    - Score improvement over time
    - Sigma reduction rate
    - Score variance (noise level)
    
    Args:
        result: Optimizer result dictionary with 'history' key
        output_path: Path to save plot
        show: Whether to display plot
        
    Returns:
        matplotlib Figure object
    """
    history = result['history']
    gens = [h['generation'] for h in history]
    best = np.array([h['best_score'] for h in history])
    mean = np.array([h['mean_score'] for h in history])
    std = np.array([h['std_score'] for h in history])
    sigmas = [h.get('cma_sigma', None) for h in history]
    
    # Filter valid sigma data
    valid_sigma = [(g, s) for g, s in zip(gens, sigmas) if s is not None]
    if valid_sigma:
        sigma_gens, sigma_vals = zip(*valid_sigma)
    else:
        sigma_gens, sigma_vals = [], []
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Score evolution
    ax1.plot(gens, best, 'b-o', label='Best', linewidth=2, markersize=5)
    ax1.plot(gens, mean, 'r--s', label='Mean', linewidth=2, markersize=4)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Convergence Diagnostics', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sigma evolution
    if sigma_vals:
        ax2.plot(sigma_gens, sigma_vals, 'g-o', linewidth=2, markersize=5)
        ax2.axhline(sigma_vals[0], color='gray', linestyle=':', alpha=0.5)
        ax2.set_ylabel('Sigma (step size)', fontsize=11, color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Add reduction info
        reduction = (1 - sigma_vals[-1] / sigma_vals[0]) * 100
        ax2.text(0.02, 0.98, f'σ reduction: {reduction:.1f}%', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                fontsize=9)
    
    # Plot 3: Score variance (noise indicator)
    ax3.plot(gens, std, 'purple', linewidth=2, marker='s', markersize=4)
    ax3.set_xlabel('Generation', fontsize=11)
    ax3.set_ylabel('Score Std Dev\n(noise level)', fontsize=11, color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3.grid(True, alpha=0.3)
    
    # Add coefficient of variation (CV) - relative noise
    cv = std / (np.abs(mean) + 1e-10) * 100
    mean_cv = np.mean(cv[np.isfinite(cv)])
    ax3.text(0.02, 0.98, f'Mean CV: {mean_cv:.1f}%\n(noise/signal)', 
            transform=ax3.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7),
            fontsize=9)
    
    # Add warning if sigma drops while noise is high
    if sigma_vals and len(sigma_vals) > 5:
        final_sigma_ratio = sigma_vals[-1] / sigma_vals[0]
        if final_sigma_ratio < 0.1 and mean_cv > 10:
            fig.text(0.5, 0.02, 
                    '⚠️  WARNING: Sigma reduced significantly while noise remains high.\n'
                    'Consider: Increasing N_REPLICAS, larger initial SIGMA, or larger POPULATION_SIZE',
                    ha='center', fontsize=10, color='red',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    if sigma_vals and len(sigma_vals) > 5 and sigma_vals[-1] / sigma_vals[0] < 0.1 and mean_cv > 10:
        plt.subplots_adjust(bottom=0.1)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {Path(output_path).name}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig
