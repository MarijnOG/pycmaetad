"""Muller-Brown CMA-ES optimization example with multiple Gaussians."""

import sys
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Ellipse

from pycmaetad.sampler import MullerBrownSampler
from pycmaetad.bias import MultiGaussian2DForceBias
from pycmaetad.evaluator import UniformKLEvaluator2D
from pycmaetad.optimizer import CMAESWorkflow


# ============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# ============================================================================

# Bias parameters (matching Alberto's scaling)
NUM_GAUSSIANS = 3
HEIGHT_RANGE = (0.0, 300.0)  # Alberto: x*1000 -> [0, 1000]
CENTER_X_RANGE = (-1.5, 1.5)  # Alberto: x*3.0-1.5 -> [-1.5, 1.5]
CENTER_Y_RANGE = (-0.5, 2.5)  # Alberto: x*3.0-0.5 -> [-0.5, 2.5]
# Alberto: sigma in [0, 0.5] -> variance [0, 0.25] -> log(variance) [log(0.0001), log(0.25)]
LOG_VARIANCE_X_RANGE = (np.log(0.001), np.log(0.5))  # sigma: ~0.01 to 0.5 nm
LOG_VARIANCE_Y_RANGE = (np.log(0.001), np.log(0.5))  # sigma: ~0.01 to 0.5 nm

# Sampler parameters
TEMPERATURE = 300.0
TIME_STEP = 0.0005 # 0.5 fs
FRICTION = 5.0 # x/ps
SIMULATION_STEPS = 25000
REPORT_INTERVAL = 100
INITIAL_POSITION = (-0.5, 1.5)

# Evaluator parameters
ENERGY_CUTOFF = 50.0  # kJ/mol above minimum for accessible region
N_BINS = 50  # Reduced from 50 for better statistics per bin (900 bins vs 2500)

# Optimizer parameters
INITIAL_MEAN = None
SIGMA = 0.5
POPULATION_SIZE = 20
MAX_GENERATIONS = 100
N_WORKERS = 20


def run_optimization():
    """Run the CMA-ES optimization and save results."""
    SCRIPT_DIR = Path(__file__).parent.resolve()
    
    print("\n" + "="*60)
    print("MULLER-BROWN CMA-ES OPTIMIZATION")
    print("Optimizing multi-Gaussian bias for uniform sampling")
    print("="*60)
    print(f"Working directory: {SCRIPT_DIR}")
    
    # Compute accessible region at 300K (kT ≈ 2.5 kJ/mol)
    accessible_ranges = get_accessible_region(temperature=TEMPERATURE, energy_cutoff=ENERGY_CUTOFF)
    
    # 1. Setup bias with 3 Gaussians
    bias = MultiGaussian2DForceBias(
        n_gaussians=NUM_GAUSSIANS,
        height_range=HEIGHT_RANGE,
        center_x_range=CENTER_X_RANGE,
        center_y_range=CENTER_Y_RANGE,
        log_variance_x_range=LOG_VARIANCE_X_RANGE,
        log_variance_y_range=LOG_VARIANCE_Y_RANGE,
    )
       
    # 2. Setup sampler
    sampler = MullerBrownSampler(
        temperature=TEMPERATURE,
        time_step=TIME_STEP,
        friction=FRICTION,
        simulation_steps=SIMULATION_STEPS,
        report_interval=REPORT_INTERVAL,
        initial_position=INITIAL_POSITION
    )
    
    # 3. Setup evaluator - only evaluate in accessible region (2D version)
    evaluator = UniformKLEvaluator2D.from_ranges(
        ranges=accessible_ranges,
        n_bins=N_BINS
    )
    
    # 4. Run optimization
    optimizer = CMAESWorkflow(
        bias=bias,
        sampler=sampler,
        evaluator=evaluator,
        initial_mean=INITIAL_MEAN,
        sigma=SIGMA,
        population_size=POPULATION_SIZE,
        max_generations=MAX_GENERATIONS,
        n_workers=N_WORKERS
    )
    
    output_dir = SCRIPT_DIR / "output_muller_brown"
    result = optimizer.optimize(str(output_dir))
    
    # Check if optimization was interrupted
    if result is None:
        print("\n⚠️  Optimization was interrupted before completing any generations")
        return None, None, None
    
    # Print results
    print("\n" + "="*60)
    if result.get('interrupted', False):
        print("OPTIMIZATION INTERRUPTED")
    else:
        print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nBest score: {result['best_score']:.6f}")
    print(f"Best generation: {result['best_generation']}")
    
    print(f"\nResults saved to: {output_dir}")
    
    # Save result dictionary for later plotting
    result_file = output_dir / "optimization_result.pkl"
    with open(result_file, 'wb') as f:
        pickle.dump({
            'result': result, 
            'bias': bias,
            'accessible_ranges': accessible_ranges
        }, f)
    print(f"✓ Result saved to: {result_file}")
    
    # Note checkpoint file location
    checkpoint_file = output_dir / "optimization_checkpoint.pkl"
    if checkpoint_file.exists():
        print(f"✓ Checkpoint file: {checkpoint_file}")
    
    return result, bias, accessible_ranges, output_dir


def resume_optimization(checkpoint_file=None):
    """Resume optimization from a checkpoint file."""
    SCRIPT_DIR = Path(__file__).parent.resolve()
    
    if checkpoint_file is None:
        checkpoint_file = SCRIPT_DIR / "output_muller_brown" / "optimization_checkpoint.pkl"
    
    checkpoint_file = Path(checkpoint_file)
    
    if not checkpoint_file.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_file}")
        print("Run optimization first with: python muller_brown_example.py run")
        return None, None, None, None
    
    print("\n" + "="*60)
    print("RESUMING MULLER-BROWN CMA-ES OPTIMIZATION")
    print("="*60)
    print(f"Checkpoint: {checkpoint_file}")
    
    # Compute accessible region
    accessible_ranges = get_accessible_region(temperature=TEMPERATURE, energy_cutoff=ENERGY_CUTOFF)
    
    # Recreate the same setup as original optimization
    bias = MultiGaussian2DForceBias(
        n_gaussians=NUM_GAUSSIANS,
        height_range=HEIGHT_RANGE,
        center_x_range=CENTER_X_RANGE,
        center_y_range=CENTER_Y_RANGE,
        log_variance_x_range=LOG_VARIANCE_X_RANGE,
        log_variance_y_range=LOG_VARIANCE_Y_RANGE,
    )
    
    sampler = MullerBrownSampler(
        temperature=TEMPERATURE,
        time_step=TIME_STEP,
        friction=FRICTION,
        simulation_steps=SIMULATION_STEPS,
        report_interval=REPORT_INTERVAL,
        initial_position=INITIAL_POSITION
    )
    
    evaluator = UniformKLEvaluator2D.from_ranges(
        ranges=accessible_ranges,
        n_bins=N_BINS
    )
    
    # Load checkpoint to get original settings
    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    original_pop_size = checkpoint_data.get('population_size', POPULATION_SIZE)
    original_n_workers = checkpoint_data.get('n_workers', N_WORKERS)
    original_sigma = checkpoint_data.get('sigma', SIGMA)
    original_max_gen = checkpoint_data.get('max_generations', MAX_GENERATIONS)
    
    print(f"\nRestoring original settings from checkpoint:")
    print(f"  Population size: {original_pop_size}")
    print(f"  Workers: {original_n_workers}")
    print(f"  Sigma: {original_sigma}")
    print(f"  Max generations: {original_max_gen}")
    
    # Create optimizer with SAME settings as original
    optimizer = CMAESWorkflow(
        bias=bias,
        sampler=sampler,
        evaluator=evaluator,
        initial_mean=None,
        sigma=original_sigma,
        population_size=original_pop_size,
        max_generations=original_max_gen,
        n_workers=original_n_workers
    )
    
    # Resume from checkpoint
    output_dir = SCRIPT_DIR / "output_muller_brown"
    result = optimizer.optimize(str(output_dir), resume_from=str(checkpoint_file))
    
    if result is None:
        print("\n⚠️  Optimization was interrupted before completing any new generations")
        return None, None, None, None
    
    # Print results
    print("\n" + "="*60)
    if result.get('interrupted', False):
        print("OPTIMIZATION INTERRUPTED AGAIN")
    else:
        print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nBest score: {result['best_score']:.6f}")
    print(f"Best generation: {result['best_generation']}")
    
    print(f"\nResults saved to: {output_dir}")
    
    # Save final result
    result_file = output_dir / "optimization_result.pkl"
    with open(result_file, 'wb') as f:
        pickle.dump({
            'result': result,
            'bias': bias,
            'accessible_ranges': accessible_ranges
        }, f)
    print(f"✓ Result saved to: {result_file}")
    
    return result, bias, accessible_ranges, output_dir


def generate_plots(result_file=None):
    """Generate plots from saved optimization results."""
    SCRIPT_DIR = Path(__file__).parent.resolve()
    
    if result_file is None:
        # Try checkpoint file first, then final result file
        checkpoint_file = SCRIPT_DIR / "output_muller_brown" / "optimization_checkpoint.pkl"
        result_file_path = SCRIPT_DIR / "output_muller_brown" / "optimization_result.pkl"
        
        if result_file_path.exists():
            result_file = result_file_path
            print("Using final result file (optimization completed)")
        elif checkpoint_file.exists():
            result_file = checkpoint_file
            print("Using checkpoint file (optimization may have been interrupted)")
        else:
            print(f"Error: No result files found in {SCRIPT_DIR / 'output_muller_brown'}")
            print("Run optimization first with: python muller_brown_example.py run")
            return
    
    result_file = Path(result_file)
    
    if not result_file.exists():
        print(f"Error: Result file not found: {result_file}")
        print("Run optimization first with: python muller_brown_example.py run")
        return
    
    # Load saved results
    print(f"\nLoading results from: {result_file}")
    with open(result_file, 'rb') as f:
        data = pickle.load(f)
    
    result = data['result']
    bias = data['bias']
    accessible_ranges = data.get('accessible_ranges')
    
    # Compute accessible ranges if not saved
    if accessible_ranges is None:
        accessible_ranges = get_accessible_region(temperature=TEMPERATURE, energy_cutoff=ENERGY_CUTOFF)
    
    output_dir = result_file.parent
    
    # Use the centralized plotting function
    create_all_plots(result, bias, accessible_ranges, output_dir)


def create_all_plots(result, bias, accessible_ranges, output_dir):
    """Create all visualization plots from optimization results.
    
    Args:
        result: Optimization result dictionary
        bias: MultiGaussian2DForceBias object
        accessible_ranges: Accessible region bounds ((x_min, x_max), (y_min, y_max))
        output_dir: Directory to save plots (Path object)
    """
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    # Generate Muller-Brown specific plots
    plot_convergence_detailed(result, output_dir)
    plot_best_bias_multi_improved(bias, result['best_parameters'], output_dir, accessible_ranges)
    plot_sampling_coverage(result, output_dir)
    plot_trajectory_density_2d(result, output_dir, accessible_ranges, n_bins=N_BINS)
    plot_cmaes_exploration(result, bias, output_dir, save_frequency=10)
    
    # Print diagnostics
    print_diagnostics(result)
    
    print(f"\n✅ All plots saved to: {output_dir}")


def get_accessible_region(temperature=300.0, energy_cutoff=0):
    """Determine the accessible region based on energy threshold.
    
    Args:
        temperature: Temperature in K
        energy_cutoff: Energy above minimum that's considered accessible (kJ/mol)
    
    Returns:
        ((x_min, x_max), (y_min, y_max))
    """
    # Create fine grid to find accessible region
    x = np.linspace(-1.5, 1.5, 500)
    y = np.linspace(-0.5, 2.5, 500)
    X, Y = np.meshgrid(x, y)
    
    Z = muller_brown_potential(X, Y)
    min_energy = Z.min()
    
    # Find regions within energy cutoff
    accessible_mask = (Z - min_energy) < energy_cutoff
    
    # Get bounding box of accessible region
    accessible_y, accessible_x = np.where(accessible_mask)
    
    x_min = x[accessible_x.min()]
    x_max = x[accessible_x.max()]
    y_min = y[accessible_y.min()]
    y_max = y[accessible_y.max()]
    
    # Add small margin
    margin_x = 0.1 * (x_max - x_min)
    margin_y = 0.1 * (y_max - y_min)
    
    return (
        (x_min - margin_x, x_max + margin_x),
        (y_min - margin_y, y_max + margin_y)
    )


def plot_convergence_detailed(result, output_dir):
    """Plot CMA-ES convergence with more detail."""
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
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top: Convergence
    axes[0].plot(gens, best_capped, 'b-o', label='Best', linewidth=2, markersize=6)
    axes[0].plot(gens, mean_capped, 'r--s', label='Mean', linewidth=2, markersize=5)
    axes[0].fill_between(gens, lower_bound, upper_bound,
                          alpha=0.2, color='red', label='±1 std')
    
    # Mark capped values
    n_capped = np.sum((best > cap_value) | (mean > cap_value))
    if n_capped > 0:
        axes[0].axhline(cap_value, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        axes[0].text(0.02, 0.98, f'Display capped at {cap_value:.1f}\n({n_capped} outliers hidden)', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    axes[0].set_xlabel('Generation', fontsize=12)
    axes[0].set_ylabel('KL Divergence', fontsize=12)
    axes[0].set_title('CMA-ES Convergence (lower = more uniform sampling)', 
                      fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Use log scale if scores vary by orders of magnitude
    if np.max(best_capped) / np.min(best_capped[best_capped > 0]) > 10:
        axes[0].set_yscale('log')
    
    # Bottom: Score distribution per generation (violin plot)
    all_scores = []
    positions = []
    for h in history[::max(1, len(history)//10)]:  # Sample every 10th gen
        all_scores.append(h['all_scores'])
        positions.append(h['generation'])
    
    parts = axes[1].violinplot(all_scores, positions=positions, widths=2, 
                                showmeans=True, showmedians=True)
    axes[1].set_xlabel('Generation', fontsize=12)
    axes[1].set_ylabel('KL Divergence', fontsize=12)
    axes[1].set_title('Score Distribution (violin plot)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / "convergence_detailed.png", dpi=150)
    plt.close()
    print(f"📊 Saved: convergence_detailed.png")


def plot_best_bias_multi_improved(bias, best_params, output_dir, accessible_ranges):
    """Visualize with MUCH better colormaps and scaling."""
    bias.set_parameters(best_params)
    
    # Create grid
    x = np.linspace(-1.5, 1.5, 300)
    y = np.linspace(-0.5, 2.5, 300)
    X, Y = np.meshgrid(x, y)
    
    # Compute MB potential
    Z_mb = muller_brown_potential(X, Y)
    
    # Compute total bias
    Z_bias = np.zeros_like(X)
    
    gaussians_info = []
    for i, gauss in enumerate(bias.get_all_gaussians()):
        params = gauss.get_parameters()
        height, cx, cy, log_var_x, rho, log_var_y = params
        
        var_x = np.exp(log_var_x)
        var_y = np.exp(log_var_y)
        cov_xy = rho * np.sqrt(var_x * var_y)
        
        # Compute inverse
        det = var_x * var_y - cov_xy**2
        i11 = var_y / det
        i12 = -cov_xy / det
        i22 = var_x / det
        
        # Add this Gaussian
        dx = X - cx
        dy = Y - cy
        exponent = -0.5 * (dx*dx*i11 + 2*dx*dy*i12 + dy*dy*i22)
        gauss_contribution = height * np.exp(exponent)
        Z_bias += gauss_contribution
        
        # Compute rotation angle from covariance matrix
        cov_matrix = np.array([[var_x, cov_xy], [cov_xy, var_y]])
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Angle of first eigenvector (in degrees)
        angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        angle_deg = np.degrees(angle_rad)
        
        # Ellipse width/height are 2*sqrt(eigenvalue) for 2σ
        width_2sigma = 2 * np.sqrt(eigenvalues[0]) * 2  # 2σ = 95% contour
        height_2sigma = 2 * np.sqrt(eigenvalues[1]) * 2
        
        gaussians_info.append({
            'height': height,
            'center': (cx, cy),
            'sigma': (np.sqrt(var_x), np.sqrt(var_y)),
            'rho': rho,
            'contribution': gauss_contribution,
            'angle': angle_deg,
            'ellipse_width': width_2sigma,
            'ellipse_height': height_2sigma
        })
    
    Z_total = Z_mb + Z_bias

    # Energy ranges for visualization
    E_min = Z_mb.min()
    E_vis_min = -150
    E_vis_max = -50 

    # Log-scale transformation (same offset for both)
    E_offset = E_min - 10  # Offset to make all values positive
    Z_mb_log = np.log10(Z_mb - E_offset + 1)  # +1 to avoid log(0)
    Z_total_log = np.log10(Z_total - E_offset + 1)  # Same transformation
    
    # Create accessible region mask
    accessible_x = accessible_ranges[0]
    accessible_y = accessible_ranges[1]
    accessible_mask = (
        (X >= accessible_x[0]) & (X <= accessible_x[1]) &
        (Y >= accessible_y[0]) & (Y <= accessible_y[1])
    )
    
    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Row 1: Main plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Row 2: Individual Gaussians
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Row 3: Analysis
    ax7 = fig.add_subplot(gs[2, :2])
    ax8 = fig.add_subplot(gs[2, 2])
    
    # Bias colormap (symmetric around 0)
    bias_max = np.percentile(np.abs(Z_bias), 95)
    bias_norm = TwoSlopeNorm(vmin=-bias_max, vcenter=0, vmax=bias_max)
    
    # ========== 1. MB Potential (LOG SCALE) ==========
    cs1 = ax1.contourf(X, Y, Z_mb_log, levels=40, cmap='viridis')
    ax1.contour(X, Y, Z_mb, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    
    # Overlay accessible region
    ax1.contour(X, Y, accessible_mask.astype(float), levels=[0.5], 
               colors='red', linewidths=3, linestyles='--', alpha=0.8)
    
    for i, info in enumerate(gaussians_info):
        ax1.plot(info['center'][0], info['center'][1], 'r*', 
                markersize=15, markeredgecolor='white', markeredgewidth=1.5,
                label=f'G{i+1}')
    
    # Add labeled contour lines at key energies
    key_levels = np.array([-150, -130, -110, -90, -70, -50])
    contour_lines = ax1.contour(X, Y, Z_mb, levels=key_levels,
                                colors='white', linewidths=1.2, alpha=0.8)
    ax1.clabel(contour_lines, inline=True, fontsize=8, 
              fmt='%0.0f', inline_spacing=10)

    # Mark well positions with actual energy
    well_x = [-0.55, 0.62, -0.05]
    well_y = [1.45, 0.03, 0.47]
    well_names = ['Main', 'Right', 'Center']

    for name, wx, wy in zip(well_names, well_x, well_y):
        ix = np.argmin(np.abs(x - wx))
        iy = np.argmin(np.abs(y - wy))
        energy = Z_mb[iy, ix]
        
        ax1.plot(wx, wy, 'w*', markersize=12, markeredgecolor='black', markeredgewidth=1)
        ax1.text(wx, wy - 0.15, f'{name}\n{energy:.0f} kJ/mol',
                ha='center', va='top', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.85))
    
    ax1.set_title('Muller-Brown Potential\n(red dashed = evaluation region)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (nm)')
    ax1.set_ylabel('Y (nm)')
    ax1.legend(loc='upper right', fontsize=8)
    cb1 = plt.colorbar(cs1, ax=ax1)
    cb1.set_label('log₁₀(E - E_min + 1)', fontsize=9)
    
    # ========== 2. Total Bias ==========
    cs2 = ax2.contourf(X, Y, Z_bias, levels=40, cmap='RdBu_r', norm=bias_norm,
                      extend='both')
    ax2.contour(X, Y, Z_bias, levels=20, colors='black', alpha=0.2, linewidths=0.5)
    
    # Overlay accessible region
    ax2.contour(X, Y, accessible_mask.astype(float), levels=[0.5], 
               colors='green', linewidths=2, linestyles='--', alpha=0.8)
    
    from matplotlib.patches import Ellipse
    
    for i, info in enumerate(gaussians_info):
        ax2.plot(info['center'][0], info['center'][1], 'k*', 
                markersize=15, markeredgecolor='white', markeredgewidth=1.5)
        
        # Draw ROTATED ellipse showing Gaussian extent (2σ)
        ellipse = Ellipse(
            info['center'], 
            width=info['ellipse_width'],
            height=info['ellipse_height'],
            angle=info['angle'],
            fill=False, 
            edgecolor='black',
            linewidth=2,
            linestyle='--',
            alpha=0.7
        )
        ax2.add_patch(ellipse)
    
    ax2.set_title(f'Multi-Gaussian Bias\n(green dashed = evaluation region)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (nm)')
    ax2.set_ylabel('Y (nm)')
    cb2 = plt.colorbar(cs2, ax=ax2)
    cb2.set_label(f'Bias (kJ/mol)\n[±{bias_max:.0f}]', fontsize=9)
    
    # ========== 3. Biased Potential (LOG SCALE - MATCHING PLOT 1) ==========
    cs3 = ax3.contourf(X, Y, Z_total_log, levels=40, cmap='viridis')  # ← Changed to log scale
    ax3.contour(X, Y, Z_total, levels=30, colors='white', alpha=0.3, linewidths=0.5)
    
    # Overlay accessible region
    ax3.contour(X, Y, accessible_mask.astype(float), levels=[0.5], 
               colors='red', linewidths=2, linestyles='--', alpha=0.8)
    
    ax3.set_title('Biased Potential\n(ideally: flat in evaluation region)', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (nm)')
    ax3.set_ylabel('Y (nm)')
    cb3 = plt.colorbar(cs3, ax=ax3)
    cb3.set_label('log₁₀(E - E_min + 1)', fontsize=9) 
    
    # ========== 4-6. Individual Gaussians ==========
    for i, (ax, info) in enumerate(zip([ax4, ax5, ax6], gaussians_info)):
        contrib_levels = np.linspace(0, info['height'], 30)
        
        cs = ax.contourf(X, Y, info['contribution'], levels=contrib_levels, 
                        cmap='Reds', extend='max')
        ax.contour(X, Y, info['contribution'], levels=15, colors='black', 
                   alpha=0.2, linewidths=0.5)
        ax.contour(X, Y, accessible_mask.astype(float), levels=[0.5],
                  colors='green', linewidths=1.5, linestyles='--', alpha=0.6)
        ax.plot(info['center'][0], info['center'][1], 'k*', 
               markersize=12, markeredgecolor='white')
        
        # Add rotated ellipse
        ellipse = Ellipse(
            info['center'],
            width=info['ellipse_width'],
            height=info['ellipse_height'],
            angle=info['angle'],
            fill=False,
            edgecolor='black',
            linewidth=2,
            linestyle='--',
            alpha=0.8
        )
        ax.add_patch(ellipse)
        
        ax.set_title(f'Gaussian {i+1}\nh={info["height"]:.1f}, '
                    f'σ=({info["sigma"][0]:.2f}, {info["sigma"][1]:.2f})\n'
                    f'ρ={info["rho"]:.2f}, θ={info["angle"]:.1f}°',
                    fontsize=10)
        ax.set_xlabel('X (nm)', fontsize=9)
        ax.set_ylabel('Y (nm)', fontsize=9)
        plt.colorbar(cs, ax=ax, label='Bias (kJ/mol)')
    
    # ========== 7. Cross-sections ==========
    well_x = [-0.55, 0.62, -0.05]
    well_y = [1.45, 0.03, 0.47]
    colors = ['b', 'r', 'g']
    
    for i, (wx, wy, color) in enumerate(zip(well_x, well_y, colors)):
        ix = np.argmin(np.abs(x - wx))
        iy = np.argmin(np.abs(y - wy))
        
        ax7.plot(x, Z_mb[:, ix], f'{color}--', alpha=0.5, linewidth=1.5,
                label=f'Well {i+1} MB')
        ax7.plot(x, Z_total[:, ix], f'{color}-', linewidth=2.5,
                label=f'Well {i+1} Biased')
    
    ax7.axvspan(accessible_x[0], accessible_x[1], alpha=0.1, color='green',
               label='Evaluation region')
    ax7.axhline(E_vis_max, color='orange', linestyle=':', linewidth=2,
               label=f'Vis cutoff ({E_vis_max} kJ/mol)')
    
    ax7.set_xlabel('X position (nm)', fontsize=11)
    ax7.set_ylabel('Energy (kJ/mol)', fontsize=11)
    ax7.set_title('Energy Cross-Sections (Y at well centers)', 
                  fontsize=12, fontweight='bold')
    ax7.legend(fontsize=8, ncol=3)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([E_vis_min - 20, E_vis_max + 50])
    
    # ========== 8. Energy histogram ==========
    Z_accessible = Z_total[accessible_mask]
    
    ax8.hist(Z_accessible.flatten(), bins=50, alpha=0.7, color='steelblue', 
            edgecolor='black', density=True)
    ax8.axvline(np.mean(Z_accessible), color='red', linestyle='--', linewidth=2,
               label=f'Mean={np.mean(Z_accessible):.1f}')
    ax8.axvline(np.median(Z_accessible), color='orange', linestyle='--', linewidth=2,
               label=f'Median={np.median(Z_accessible):.1f}')
    
    ax8.set_xlabel('Energy (kJ/mol)', fontsize=11)
    ax8.set_ylabel('Probability Density', fontsize=11)
    ax8.set_title('Biased Energy Distribution\n(accessible region only)', 
                  fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(output_dir / "best_bias_multi_detailed.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Saved: best_bias_multi_detailed.png")
    
    # Print detailed stats
    print("\n" + "="*60)
    print("GAUSSIAN PLACEMENT ANALYSIS")
    print("="*60)
    for i, info in enumerate(gaussians_info):
        print(f"\nGaussian {i+1}:")
        print(f"  Height:      {info['height']:>7.2f} kJ/mol")
        print(f"  Center:      ({info['center'][0]:>6.3f}, {info['center'][1]:>6.3f}) nm")
        print(f"  Sigma:       ({info['sigma'][0]:>6.3f}, {info['sigma'][1]:>6.3f}) nm")
        print(f"  Correlation: {info['rho']:>6.3f}")
        print(f"  Rotation:    {info['angle']:>6.1f}°")
        print(f"  Coverage:    ~{4*info['sigma'][0]*4*info['sigma'][1]:.2f} nm² (4σ ellipse)")
    
    print(f"\nBiased potential statistics (ACCESSIBLE REGION ONLY):")
    print(f"  Range:  [{Z_accessible.min():.1f}, {Z_accessible.max():.1f}] kJ/mol")
    print(f"  Spread: {Z_accessible.max() - Z_accessible.min():.1f} kJ/mol")
    print(f"  Std:    {Z_accessible.std():.1f} kJ/mol")
    print(f"  → Goal: minimize spread & std for flat landscape")
    
    print(f"\nAccessible region:")
    print(f"  X: [{accessible_x[0]:.2f}, {accessible_x[1]:.2f}] nm")
    print(f"  Y: [{accessible_y[0]:.2f}, {accessible_y[1]:.2f}] nm")
    print(f"  Area: {(accessible_x[1]-accessible_x[0])*(accessible_y[1]-accessible_y[0]):.2f} nm²")
    
    print(f"\nVisualization range: [{E_vis_min}, {E_vis_max}] kJ/mol")


def plot_sampling_coverage(result, output_dir):
    """Show how sampling coverage evolves."""
    history = result['history']
    last_gen = history[-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scores = last_gen['all_scores']
    ax.bar(range(len(scores)), sorted(scores), alpha=0.7, color='steelblue',
          edgecolor='black')
    ax.axhline(last_gen['best_score'], color='green', linestyle='--', 
              linewidth=2, label=f'Best={last_gen["best_score"]:.3f}')
    ax.axhline(last_gen['mean_score'], color='red', linestyle='--',
              linewidth=2, label=f'Mean={last_gen["mean_score"]:.3f}')
    
    ax.set_xlabel('Individual (sorted)', fontsize=12)
    ax.set_ylabel('KL Divergence', fontsize=12)
    ax.set_title(f'Final Generation Scores (gen {last_gen["generation"]})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "final_generation_scores.png", dpi=150)
    plt.close()
    print(f"📊 Saved: final_generation_scores.png")


def print_diagnostics(result):
    """Print optimization diagnostics."""
    history = result['history']
    
    print("\n" + "="*60)
    print("OPTIMIZATION DIAGNOSTICS")
    print("="*60)
    
    # Convergence rate
    initial_best = history[0]['best_score']
    final_best = history[-1]['best_score']
    improvement = (initial_best - final_best) / initial_best * 100
    
    print(f"\n📈 Convergence:")
    print(f"  Initial best:  {initial_best:.4f}")
    print(f"  Final best:    {final_best:.4f}")
    print(f"  Improvement:   {improvement:.1f}%")
    
    if improvement < 10:
        print(f"  ⚠️  WARNING: Less than 10% improvement!")
        print(f"      → Try more generations or better initialization")
    
    # Stagnation check
    last_5 = [h['best_score'] for h in history[-5:]]
    if len(last_5) >= 5:
        stagnation = np.std(last_5) / np.mean(last_5) * 100
        print(f"\n📊 Stagnation (last 5 gens):")
        print(f"  Relative std: {stagnation:.2f}%")
        if stagnation < 1:
            print(f"  ⚠️  Likely converged (or stuck!)")
    
    # Noise estimate
    mean_stds = [h['std_score'] / h['mean_score'] for h in history]
    avg_noise = np.mean(mean_stds) * 100
    print(f"\n🔊 Noise level:")
    print(f"  Avg CV:  {avg_noise:.1f}%")
    if avg_noise > 30:
        print(f"  ⚠️  High noise! Consider:")
        print(f"      → Longer simulations (more steps)")
        print(f"      → More bins in evaluator")
        print(f"      → Averaging multiple runs per individual")
    
    # Efficiency
    total_time = sum(h['generation_time'] for h in history)
    total_evals = len(history) * len(history[0]['all_scores'])
    avg_time_per_eval = total_time / total_evals
    
    print(f"\n⏱️  Efficiency:")
    print(f"  Total time:       {total_time:.1f} s ({total_time/60:.1f} min)")
    print(f"  Avg per eval:     {avg_time_per_eval:.1f} s")
    print(f"  Total evals:      {total_evals}")


def plot_cmaes_exploration(result, bias, output_dir, save_frequency=10):
    """Visualize CMA-ES parameter exploration over generations.
    
    Shows where the optimizer is searching in parameter space, focusing on
    Gaussian center positions (cx, cy) and other key parameters.
    
    Args:
        result: Optimization result dictionary with history
        bias: MultiGaussian2DForceBias object (for parameter interpretation)
        output_dir: Directory to save plots
        save_frequency: Save plots every N generations (default: 10)
    """
    history = result['history']
    n_gaussians = bias.n_gaussians
    
    # Identify which generations to plot
    gens_to_plot = list(range(0, len(history), save_frequency))
    if (len(history) - 1) not in gens_to_plot:
        gens_to_plot.append(len(history) - 1)  # Always include final
    
    print(f"\n📊 Creating CMA-ES exploration plots for {len(gens_to_plot)} generations...")
    
    for gen_idx in gens_to_plot:
        gen_data = history[gen_idx]
        gen = gen_data['generation']
        population = np.array(gen_data['population'])  # Convert list to numpy array
        cma_mean = gen_data['cma_mean']
        scores = gen_data['all_scores']
        best_idx = np.argmin(scores)
        
        # Denormalize population for visualization
        population_denorm = np.array([bias.denormalize_parameters(p) for p in population])
        mean_denorm = bias.denormalize_parameters(cma_mean)
        
        # Create visualization - simpler layout
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Gaussian centers in 2D space with CMA-ES search ellipse
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Draw Muller-Brown wells for reference
        well_positions = [(-0.55, 1.45), (0.62, 0.03), (-0.05, 0.47)]
        for wx, wy in well_positions:
            ax1.scatter(wx, wy, c='red', marker='*', s=200, edgecolor='darkred', 
                       linewidth=1.5, zorder=10, alpha=0.7)
        
        # Plot each Gaussian's mean and best positions (not full population)
        colors = plt.cm.Set1(np.linspace(0, 1, n_gaussians))
        for g in range(n_gaussians):
            cx_idx = g * 6 + 1  # Parameter index for cx
            cy_idx = g * 6 + 2  # Parameter index for cy
            
            # Mark best solution only
            ax1.scatter(population_denorm[best_idx, cx_idx], 
                       population_denorm[best_idx, cy_idx],
                       color=colors[g], marker='o', s=150, edgecolor='black', 
                       linewidth=2, zorder=9, label=f'G{g+1} (best)')
            
            # Mark CMA mean
            ax1.scatter(mean_denorm[cx_idx], mean_denorm[cy_idx],
                       color=colors[g], marker='X', s=180, edgecolor='black',
                       linewidth=1.5, zorder=8, alpha=0.6)
            
            # Draw CMA-ES search ellipse for this Gaussian's position (2-sigma)
            if gen_data.get('cma_cov') is not None:
                cov_matrix = gen_data['cma_cov']
                
                # Check if covariance matrix is valid
                if isinstance(cov_matrix, np.ndarray) and cov_matrix.ndim == 2:
                    try:
                        # Extract 2x2 covariance for (cx, cy) of this Gaussian
                        cov_cx_cy = cov_matrix[np.ix_([cx_idx, cy_idx], [cx_idx, cy_idx])]
                        
                        # Denormalize covariance (scale by parameter ranges)
                        bounds = bias.get_parameter_bounds()
                        scale_x = bounds[cx_idx, 1] - bounds[cx_idx, 0]
                        scale_y = bounds[cy_idx, 1] - bounds[cy_idx, 0]
                        cov_cx_cy_denorm = cov_cx_cy * np.outer([scale_x, scale_y], [scale_x, scale_y])
                        
                        # Compute eigenvalues and eigenvectors for ellipse
                        eigenvalues, eigenvectors = np.linalg.eigh(cov_cx_cy_denorm)
                        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                        width = 2 * np.sqrt(eigenvalues[0]) * 2  # 2-sigma
                        height = 2 * np.sqrt(eigenvalues[1]) * 2
                        
                        ellipse = Ellipse(
                            (mean_denorm[cx_idx], mean_denorm[cy_idx]),
                            width=width, height=height, angle=angle,
                            fill=False, edgecolor=colors[g], linewidth=2,
                            linestyle='--', alpha=0.5
                        )
                        ax1.add_patch(ellipse)
                    except (IndexError, ValueError) as e:
                        # Skip ellipse if covariance extraction fails
                        pass
        
        ax1.set_xlim(-1.5, 1.0)
        ax1.set_ylim(-0.5, 2.0)
        ax1.set_xlabel('x (nm)', fontsize=12)
        ax1.set_ylabel('y (nm)', fontsize=12)
        ax1.set_title(f'Generation {gen}: Gaussian Centers\n(dashed ellipse = CMA-ES search region)', 
                     fontsize=12, weight='bold')
        ax1.legend(loc='upper right', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Parameter ranges (heights and variances as box plots)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Show distribution of heights and correlation coefficients
        param_data = []
        param_labels = []
        param_colors = []
        
        for g in range(n_gaussians):
            h_idx = g * 6
            h_vals = population_denorm[:, h_idx]
            param_data.append(h_vals)
            param_labels.append(f'G{g+1}\nHeight')
            param_colors.append(colors[g])
        
        bp = ax2.boxplot(param_data, tick_labels=param_labels, patch_artist=True,
                        showmeans=True, meanline=True,
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.5),
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(color='blue', linewidth=2, linestyle='--'))
        
        # Color boxes
        for patch, color in zip(bp['boxes'], param_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        # Mark best values
        for g in range(n_gaussians):
            h_idx = g * 6
            ax2.scatter(g + 1, population_denorm[best_idx, h_idx],
                       c='lime', marker='D', s=120, edgecolor='darkgreen',
                       linewidth=2, zorder=10)
        
        ax2.set_ylabel('Height (kJ/mol)', fontsize=12)
        ax2.set_title(f'Generation {gen}: Gaussian Heights\n(red=median, blue=mean, green=best)', 
                     fontsize=12, weight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Score distribution and improvement
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Show score range and statistics
        ax3.hist(scores, bins=12, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1.5)
        ax3.axvline(scores[best_idx], color='lime', linestyle='--', linewidth=3, 
                   label=f'Best: {scores[best_idx]:.3f}')
        ax3.axvline(np.mean(scores), color='orange', linestyle='--', linewidth=3,
                   label=f'Mean: {np.mean(scores):.3f}')
        ax3.axvline(np.median(scores), color='red', linestyle=':', linewidth=2,
                   label=f'Median: {np.median(scores):.3f}')
        
        ax3.set_xlabel('KL Divergence', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title(f'Generation {gen}: Score Distribution\nStd: {np.std(scores):.3f}', 
                     fontsize=12, weight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Overall title
        fig.suptitle(f'CMA-ES Exploration - Generation {gen}\n' + 
                    f'Best: {scores[best_idx]:.4f} | Mean: {np.mean(scores):.4f} | Std: {np.std(scores):.4f}',
                    fontsize=14, weight='bold', y=1.02)
        
        # Save
        output_path = Path(output_dir) / f'cmaes_gen{gen:03d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {output_path.name}")
    
    print(f"✅ CMA-ES exploration plots complete!\n")


def plot_trajectory_density_2d(result, output_dir, accessible_ranges, n_bins=50):
    """Plot 2D density heatmap of sampled trajectory positions.
    
    Shows the actual sampling distribution overlaid on the Muller-Brown potential,
    allowing visual comparison with the target uniform distribution.
    
    Args:
        result: Optimization result dictionary with 'best_generation' and 'history'
        output_dir: Directory to save plots
        accessible_ranges: ((x_min, x_max), (y_min, y_max)) evaluation region
        n_bins: Number of bins for the 2D histogram (default: 50)
    """
    # Find the best trajectory file from the best generation
    best_gen = result['best_generation']
    best_gen_data = result['history'][best_gen]
    
    # Find the individual with the best score in that generation
    best_idx = np.argmin(best_gen_data['all_scores'])
    
    # Construct path to trajectory file
    gen_dir = Path(output_dir) / f"gen{best_gen:03d}"
    ind_dir = gen_dir / f"ind{best_idx:03d}"
    best_traj_file = ind_dir / "output.pdb"
    
    if not best_traj_file.exists():
        print(f"⚠️  Trajectory file not found: {best_traj_file}")
        print(f"    (Expected from gen {best_gen}, individual {best_idx})")
        return
    
    # Parse PDB to get positions
    positions = []
    with open(best_traj_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # PDB format: positions in Angstroms starting at column 30
                x = float(line[30:38].strip()) / 10.0  # Convert Å to nm
                y = float(line[38:46].strip()) / 10.0
                positions.append([x, y])
    
    positions = np.array(positions)
    
    if len(positions) == 0:
        print(f"⚠️  No positions found in trajectory file: {best_traj_file}")
        return
    
    print(f"📊 Loaded {len(positions)} trajectory frames from {best_traj_file.name}")
    
    # Create 2D histogram
    x_range = accessible_ranges[0]
    y_range = accessible_ranges[1]
    
    hist, x_edges, y_edges = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=n_bins,
        range=[x_range, y_range],
        density=False  # Get counts
    )
    
    # Normalize to probabilities
    hist_prob = hist / np.sum(hist)
    
    # Create grid for Muller-Brown potential
    x = np.linspace(x_range[0], x_range[1], 300)
    y = np.linspace(y_range[0], y_range[1], 300)
    X, Y = np.meshgrid(x, y)
    Z = muller_brown_potential(X, Y)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Muller-Brown potential with trajectory overlay
    ax1.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
    ax1.contour(X, Y, Z, levels=15, colors='white', linewidths=0.5, alpha=0.5)
    
    # Overlay trajectory density as scatter plot with alpha
    scatter = ax1.scatter(positions[::10, 0], positions[::10, 1],  # Subsample for visibility
                         c='red', s=1, alpha=0.1, rasterized=True)
    
    ax1.set_xlabel('X (nm)', fontsize=12)
    ax1.set_ylabel('Y (nm)', fontsize=12)
    ax1.set_title(f'Muller-Brown Potential + Trajectory Points\n({len(positions)} frames, every 10th shown)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    
    # Plot 2: 2D density heatmap
    X_bins = (x_edges[:-1] + x_edges[1:]) / 2
    Y_bins = (y_edges[:-1] + y_edges[1:]) / 2
    X_bins, Y_bins = np.meshgrid(X_bins, Y_bins)
    
    im = ax2.contourf(X_bins, Y_bins, hist_prob.T, levels=30, cmap='hot')
    ax2.contour(X_bins, Y_bins, hist_prob.T, levels=10, colors='white', 
               linewidths=0.5, alpha=0.5)
    
    # Add colorbar
    cb = plt.colorbar(im, ax=ax2)
    cb.set_label('Probability Density', fontsize=11)
    
    ax2.set_xlabel('X (nm)', fontsize=12)
    ax2.set_ylabel('Y (nm)', fontsize=12)
    ax2.set_title(f'Sampling Density (2D Histogram)\nBins: {n_bins}×{n_bins}', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlim(x_range)
    ax2.set_ylim(y_range)
    
    # Add statistics
    uniformity = 1.0 / (n_bins * n_bins)  # Target uniform probability
    max_prob = np.max(hist_prob)
    min_prob = np.min(hist_prob[hist_prob > 0]) if np.any(hist_prob > 0) else 0
    
    stats_text = (f"Target uniform: {uniformity:.6f}\n"
                  f"Max observed: {max_prob:.6f}\n"
                  f"Min observed: {min_prob:.6f}\n"
                  f"Ratio: {max_prob/uniformity:.2f}×")
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'trajectory_density_2d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved trajectory density plot: {output_path.name}\n")


def muller_brown_potential(X, Y):
    """Vectorized Muller-Brown potential."""
    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])
    
    V = np.zeros_like(X)
    for i in range(4):
        V += A[i] * np.exp(
            a[i] * (X - x0[i])**2 + 
            b[i] * (X - x0[i]) * (Y - y0[i]) + 
            c[i] * (Y - y0[i])**2
        )
    return V


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "run":
            # Run optimization only
            run_optimization()
        elif command == "resume":
            # Resume optimization from checkpoint
            checkpoint_file = sys.argv[2] if len(sys.argv) > 2 else None
            resume_optimization(checkpoint_file)
        elif command == "plot":
            # Generate plots only
            result_file = sys.argv[2] if len(sys.argv) > 2 else None
            generate_plots(result_file)
        elif command == "both":
            # Run optimization and generate plots
            result, bias, accessible_ranges, output_dir = run_optimization()
            if result is not None:
                create_all_plots(result, bias, accessible_ranges, output_dir)
        else:
            print("Unknown command. Usage:")
            print("  python muller_brown_example.py run              # Run optimization from start")
            print("  python muller_brown_example.py resume [file]    # Resume from checkpoint")
            print("  python muller_brown_example.py plot [file]      # Generate plots from saved results")
            print("  python muller_brown_example.py both             # Run optimization and generate plots")
            print("\nFor testing evaluation with ideal bias, run:")
            print("  python test_ideal_bias.py")
    else:
        # Default: run both optimization and plotting
        result, bias, accessible_ranges, output_dir = run_optimization()
        if result is not None:
            create_all_plots(result, bias, accessible_ranges, output_dir)
