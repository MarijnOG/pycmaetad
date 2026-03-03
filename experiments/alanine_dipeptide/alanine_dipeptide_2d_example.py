"""Alanine Dipeptide 2D CMA-ES optimization example.

This example demonstrates:
- 2D collective variable optimization (phi and psi torsion angles)
- PLUMED integration with OpenMM for molecular dynamics
- 2D Ramachandran space sampling
- Uniform distribution optimization in 2D periodic space

SPECIAL CONSIDERATIONS FOR 2D:
1. Parameter space: Much larger than 1D
   - With hills_per_d=2 and 2 CVs (phi, psi): 2 × 2 = 4 hills total
   - Each hill has 6 parameters in 2D: (center_x, center_y, height, width_x, width_y, correlation)
   - Total: ~24 parameters vs 6 for 1D
   - Correlation (ρ): controls angle between phi and psi Gaussians
     * ρ = 0: independent (diagonal covariance)
     * ρ > 0: positive correlation (elongated along diagonal)
     * ρ < 0: negative correlation (elongated along anti-diagonal)
   
2. Computational cost:
   - 2D histograms are more expensive than 1D
   - More simulation time may be needed to adequately sample 2D space
   - Consider using fewer bins initially (e.g., 20x20 instead of 50x50)

3. Periodicity:
   - Both phi and psi are periodic: [-π, π]
   - Evaluator must handle periodic boundaries correctly
   - Hills near boundaries wrap around

4. Initial placement:
   - Hills should be distributed across 2D space
   - Consider known secondary structure regions (α-helix, β-sheet, etc.)
   - Or use uniform grid initialization

5. Visualization:
   - 2D contour plots (Ramachandran plots)
   - Free energy surfaces
   - Trajectory overlays on Ramachandran space
"""

import sys
import pickle
import importlib.util
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pycmaetad.bias import PlumedHillBias2D
from pycmaetad.sampler import OpenMMPlumedSampler
from pycmaetad.optimizer import CMAESWorkflow
from pycmaetad.evaluator import UniformKLEvaluator2D


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(config_path):
    """Load configuration from a Python file.
    
    Args:
        config_path: Path to config .py file
        
    Returns:
        Dictionary containing configuration parameters
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    if not hasattr(config_module, 'CONFIG'):
        raise ValueError(f"Config file must define a 'CONFIG' dictionary: {config_path}")
    
    return config_module.CONFIG


# Global config (loaded in main)
CONFIG = None


def create_2d_initial_mean(hills_per_d):
    """Create initial mean for 2D hills placement.
    
    Places hills distributed across phi-psi space in a 2D grid:
    - Centers distributed uniformly in a 2D grid
    - Heights at mid-range
    - Widths at mid-range
    - No correlation initially
    
    Args:
        hills_per_d: Number of hills per CV dimension
        
    Returns:
        Initial mean vector in normalized [0,1] space
        
    Note:
        For hills_per_d=2 in 2D (phi, psi): 2 × 2 = 4 hills total
        For hills_per_d=3 in 2D (phi, psi): 3 × 3 = 9 hills total
        Each hill: 6 params (center_phi, center_psi, height, width_phi, width_psi, correlation)
    """
    n_hills_total = hills_per_d ** 2  # 2D grid of hills
    n_params_total = n_hills_total * 6  # 6 params per 2D hill
    
    mean = np.ones(n_params_total) * 0.5  # Default to middle
    
    # Create 2D grid of positions distributed across [0, 1] × [0, 1] space
    grid_1d = np.linspace(0, 1, hills_per_d, endpoint=False) + 0.5/hills_per_d
    grid_phi, grid_psi = np.meshgrid(grid_1d, grid_1d)
    
    # Flatten grid to get list of hill centers (row-major order)
    centers_phi = grid_phi.flatten()
    centers_psi = grid_psi.flatten()
    
    # Assign center positions for each hill in the grid
    for hill_idx in range(n_hills_total):
        param_start = hill_idx * 6
        mean[param_start + 0] = centers_phi[hill_idx]  # phi center
        mean[param_start + 1] = centers_psi[hill_idx]  # psi center
        # Other parameters (height, widths, correlation) remain at default 0.5
    
    print("\n🎯 2D Initial hill placement (grid):")
    for i in range(n_hills_total):
        param_start = i * 6
        print(f"  Hill {i}: phi={mean[param_start]:.2f}, psi={mean[param_start+1]:.2f}")
    
    return mean


def run_optimization():
    """Run the CMA-ES optimization and save results."""
    SCRIPT_DIR = Path(__file__).parent.resolve()
    
    print("\n" + "="*80)
    print("ALANINE DIPEPTIDE 2D CMA-ES OPTIMIZATION")
    print("Optimizing 2D bias parameters for uniform Ramachandran sampling")
    print("="*80)
    print(f"Configuration: {CONFIG['name']}")
    print(f"Description: {CONFIG['description']}")
    print(f"Working directory: {SCRIPT_DIR}")
    print(f"⚠️  2D NOTE: This is computationally expensive!")
    print(f"   - 2D histograms: {CONFIG['n_bins']}x{CONFIG['n_bins']} = {CONFIG['n_bins']**2} bins")
    print(f"   - Simulation time: {CONFIG['simulation_steps']} steps ({CONFIG['simulation_steps']*CONFIG['time_step']:.1f} ps)")
    print("="*80 + "\n")
    
    # Check for PDB file
    pdb_file = SCRIPT_DIR / "alanine-dipeptide-nowater.pdb"
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    # Create bin edges from config
    bin_edges_phi = np.linspace(-np.pi, np.pi, CONFIG['n_bins'] + 1)
    bin_edges_psi = np.linspace(-np.pi, np.pi, CONFIG['n_bins'] + 1)
    
    # Create sampler
    sampler = OpenMMPlumedSampler(
        pdb_file=str(pdb_file),
        forcefield_files=["amber14-all.xml", "amber14/tip3pfb.xml"],
        temperature=CONFIG['temperature'],
        time_step=CONFIG['time_step'],
        friction=CONFIG['friction'],
        simulation_steps=CONFIG['simulation_steps'],
        report_interval=CONFIG['report_interval']
    )
    
    # Create 2D bias
    bias = PlumedHillBias2D(
        plumed_template=str(SCRIPT_DIR / "plumed_template_2d.dat"),
        hills_per_d=CONFIG['hills_per_d'],
        hills_space=CONFIG['hills_space'],
        hills_height=CONFIG['hills_height'],
        hills_width=CONFIG['hills_width'],
        multivariate=CONFIG['multivariate']
    )
    
    # Create 2D evaluator
    evaluator = UniformKLEvaluator2D(
        bin_edges=(bin_edges_phi, bin_edges_psi)
    )
    
    # Create initial mean for 2D
    initial_mean = create_2d_initial_mean(CONFIG['hills_per_d'])
    
    # Debug: Show initial parameters
    print("\n" + "="*80)
    print("INITIAL PARAMETERS (2D)")
    print("="*80)
    bounds = bias.get_parameter_bounds()
    initial_params_denorm = bounds[:, 0] + initial_mean * (bounds[:, 1] - bounds[:, 0])
    
    n_hills = CONFIG['hills_per_d'] * 2  # hills per dimension × 2 CVs
    for i in range(n_hills):
        param_start = i * 6
        phi_c = initial_params_denorm[param_start]
        psi_c = initial_params_denorm[param_start + 1]
        height = initial_params_denorm[param_start + 2]
        width_phi = initial_params_denorm[param_start + 3]
        width_psi = initial_params_denorm[param_start + 4]
        corr = initial_params_denorm[param_start + 5]
        print(f"  Hill {i}:")
        print(f"    Center: (φ={phi_c:.2f}, ψ={psi_c:.2f}) rad")
        print(f"    Height: {height:.1f} kJ/mol")
        print(f"    Widths: (σ_φ={width_phi:.2f}, σ_ψ={width_psi:.2f}) rad")
        print(f"    Correlation: {corr:.2f}")
    print("="*80 + "\n")
    
    # Auto-calculate population size if not set
    n_params = len(initial_mean)
    calc_pop_size = max(16, int(4 + 3 * np.log(n_params)))
    pop_size = CONFIG['population_size'] if CONFIG['population_size'] is not None else calc_pop_size
    
    print(f"📊 CMA-ES Configuration:")
    print(f"   Parameters: {n_params}")
    print(f"   Population size: {pop_size} {'(auto-calculated from config)' if CONFIG['population_size'] is None else ''}")
    print(f"   Max generations: {CONFIG['max_generations']}")
    print(f"   Workers: {CONFIG['n_workers']}")
    print(f"   Replicas per eval: {CONFIG['n_replicas']}")
    print()
    
    # Override parameter bounds to enforce minimum widths (prevent degenerate narrow Gaussians)
    custom_bounds = bias.get_parameter_bounds()
    n_hills = CONFIG['hills_per_d'] * 2  # hills per dimension × 2 CVs
    for i in range(n_hills):
        idx = i * 6
        # Enforce minimum widths in both dimensions
        custom_bounds[idx + 3] = [CONFIG['min_width'], CONFIG['hills_width'][0]]  # width_x
        custom_bounds[idx + 4] = [CONFIG['min_width'], CONFIG['hills_width'][1]]  # width_y
    
    print(f"🔧 Custom parameter bounds:")
    print(f"   Heights: [0, {CONFIG['hills_height']}] kJ/mol")
    print(f"   Widths:  [{CONFIG['min_width']}, {CONFIG['hills_width'][0]}] rad (enforcing minimum to prevent pathological narrow hills)")
    print(f"   Correlation: [-1, 1]")
    print()
    
    # Create workflow
    workflow = CMAESWorkflow(
        bias=bias,
        sampler=sampler,
        evaluator=evaluator,
        initial_mean=initial_mean,
        sigma=CONFIG['sigma'],
        population_size=pop_size,
        bounds=custom_bounds,  # Use custom bounds with minimum width constraint
        max_generations=CONFIG['max_generations'],
        n_workers=CONFIG['n_workers'],
        n_replicas=CONFIG['n_replicas'],
        early_stop_patience=CONFIG['early_stop_patience']
    )
    
    # Run optimization
    output_dir = SCRIPT_DIR / "output_alanine_2d"
    result = workflow.optimize(str(output_dir))
    
    # Check if optimization was interrupted
    if result is None:
        print("\n⚠️  Optimization was interrupted before completing any generations")
        return None
    
    # Print results
    print("\n" + "="*80)
    if result.get('interrupted', False):
        print("OPTIMIZATION INTERRUPTED")
    else:
        print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nBest score: {result['best_score']:.6f}")
    print(f"Best generation: {result['best_generation']}")
    
    print(f"\nResults saved to: {output_dir}")
    
    # Save result dictionary for later plotting
    result_file = output_dir / "optimization_result.pkl"
    with open(result_file, 'wb') as f:
        pickle.dump({'result': result, 'bias': bias}, f)
    print(f"✓ Result saved to: {result_file}")
    
    checkpoint_file = output_dir / "optimization_checkpoint.pkl"
    if checkpoint_file.exists():
        print(f"✓ Checkpoint file: {checkpoint_file}")
    
    return result


def resume_optimization(checkpoint_file=None):
    """Resume optimization from a checkpoint file."""
    SCRIPT_DIR = Path(__file__).parent.resolve()
    
    if checkpoint_file is None:
        checkpoint_file = SCRIPT_DIR / "output_alanine_2d" / "optimization_checkpoint.pkl"
    
    checkpoint_file = Path(checkpoint_file)
    
    if not checkpoint_file.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_file}")
        print("Run optimization first with: python alanine_dipeptide_2d_example.py run")
        return None
    
    print("\n" + "="*80)
    print("RESUMING ALANINE DIPEPTIDE 2D CMA-ES OPTIMIZATION")
    print("="*80)
    print(f"Configuration: {CONFIG['name']}")
    print(f"Checkpoint: {checkpoint_file}")
    
    # Check for PDB file
    pdb_file = SCRIPT_DIR / "alanine-dipeptide-nowater.pdb"
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    # Create bin edges from config
    bin_edges_phi = np.linspace(-np.pi, np.pi, CONFIG['n_bins'] + 1)
    bin_edges_psi = np.linspace(-np.pi, np.pi, CONFIG['n_bins'] + 1)
    
    # Load checkpoint to get original settings
    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    original_pop_size = checkpoint_data.get('population_size', 16)
    original_n_workers = checkpoint_data.get('n_workers', CONFIG['n_workers'])
    original_n_replicas = checkpoint_data.get('n_replicas', CONFIG['n_replicas'])
    original_sigma = checkpoint_data.get('sigma', CONFIG['sigma'])
    original_max_gen = checkpoint_data.get('max_generations', CONFIG['max_generations'])
    
    print(f"\nRestoring original settings from checkpoint:")
    print(f"  Population size: {original_pop_size}")
    print(f"  Workers: {original_n_workers}")
    print(f"  Replicas: {original_n_replicas}")
    print(f"  Sigma: {original_sigma}")
    print(f"  Max generations: {original_max_gen}")
    
    # Recreate the same setup
    sampler = OpenMMPlumedSampler(
        pdb_file=str(pdb_file),
        forcefield_files=["amber14-all.xml", "amber14/tip3pfb.xml"],
        temperature=CONFIG['temperature'],
        time_step=CONFIG['time_step'],
        friction=CONFIG['friction'],
        simulation_steps=CONFIG['simulation_steps'],
        report_interval=CONFIG['report_interval']
    )
    
    bias = PlumedHillBias2D(
        plumed_template=str(SCRIPT_DIR / "plumed_template_2d.dat"),
        hills_per_d=CONFIG['hills_per_d'],
        hills_space=CONFIG['hills_space'],
        hills_height=CONFIG['hills_height'],
        hills_width=CONFIG['hills_width'],
        multivariate=CONFIG['multivariate']
    )
    
    evaluator = UniformKLEvaluator2D(
        bin_edges=(bin_edges_phi, bin_edges_psi)
    )
    
    # Override parameter bounds to enforce minimum widths (prevent degenerate narrow Gaussians)
    custom_bounds = bias.get_parameter_bounds()
    n_hills = CONFIG['hills_per_d'] * 2  # hills per dimension × 2 CVs
    for i in range(n_hills):
        idx = i * 6
        # Enforce minimum widths in both dimensions
        custom_bounds[idx + 3] = [CONFIG['min_width'], CONFIG['hills_width'][0]]  # width_x
        custom_bounds[idx + 4] = [CONFIG['min_width'], CONFIG['hills_width'][1]]  # width_y
    
    print(f"\n🔧 Using custom parameter bounds:")
    print(f"   Widths: [{CONFIG['min_width']}, {CONFIG['hills_width'][0]}] rad (minimum width constraint)")
    
    # Create workflow with same settings
    workflow = CMAESWorkflow(
        bias=bias,
        sampler=sampler,
        evaluator=evaluator,
        initial_mean=None,
        sigma=original_sigma,
        population_size=original_pop_size,
        bounds=custom_bounds,  # Use custom bounds with minimum width constraint
        max_generations=original_max_gen,
        n_workers=original_n_workers,
        n_replicas=original_n_replicas
    )
    
    # Resume from checkpoint
    output_dir = SCRIPT_DIR / "output_alanine_2d"
    result = workflow.optimize(str(output_dir), resume_from=str(checkpoint_file))
    
    if result is None:
        print("\n⚠️  Optimization was interrupted before completing any new generations")
        return None
    
    # Print results
    print("\n" + "="*80)
    if result.get('interrupted', False):
        print("OPTIMIZATION INTERRUPTED AGAIN")
    else:
        print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nBest score: {result['best_score']:.6f}")
    print(f"Best generation: {result['best_generation']}")
    
    print(f"\nResults saved to: {output_dir}")
    
    # Save final result
    result_file = output_dir / "optimization_result.pkl"
    with open(result_file, 'wb') as f:
        pickle.dump({'result': result, 'bias': bias}, f)
    print(f"✓ Result saved to: {result_file}")
    
    return result


def generate_plots(result_file=None):
    """Generate plots from saved optimization results."""
    SCRIPT_DIR = Path(__file__).parent.resolve()
    
    if result_file is None:
        checkpoint_file = SCRIPT_DIR / "output_alanine_2d" / "optimization_checkpoint.pkl"
        result_file_path = SCRIPT_DIR / "output_alanine_2d" / "optimization_result.pkl"
        
        if result_file_path.exists():
            result_file = result_file_path
            print("Using final result file (optimization completed)")
        elif checkpoint_file.exists():
            result_file = checkpoint_file
            print("Using checkpoint file (optimization may have been interrupted)")
        else:
            print(f"Error: No result files found in {SCRIPT_DIR / 'output_alanine_2d'}")
            print("Run optimization first with: python alanine_dipeptide_2d_example.py run")
            return
    
    result_file = Path(result_file)
    
    if not result_file.exists():
        print(f"Error: Result file not found: {result_file}")
        print("Run optimization first with: python alanine_dipeptide_2d_example.py run")
        return
    
    # Load saved results
    print(f"\nLoading results from: {result_file}")
    with open(result_file, 'rb') as f:
        data = pickle.load(f)
    
    result = data['result']
    bias = data['bias']
    output_dir = result_file.parent
    
    # Use the centralized plotting function
    create_all_plots(result, bias, output_dir)


def plot_bias_landscape(bias, output_path, title="Bias Potential Landscape", n_points=(150, 150)):
    """Plot 2D bias potential landscape showing Gaussian hills.
    
    Args:
        bias: PlumedHillBias2D object with parameters set
        output_path: Path to save the plot
        title: Plot title
        n_points: Grid resolution (nx, ny)
    """
    # Compute bias landscape
    X, Y, V = bias.compute_bias_landscape(n_points=n_points, periodic=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot bias as contour + filled contours
    levels = 20
    contourf = ax.contourf(X, Y, V, levels=levels, cmap='RdYlBu_r', alpha=0.8)
    contour = ax.contour(X, Y, V, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax, label='Bias Potential (kJ/mol)')
    
    # Plot hill centers as points (with periodic replication)
    if bias._centers_x is not None:
        from matplotlib.patches import Ellipse
        
        # Collect all centers (original + periodic images)
        all_centers_x = []
        all_centers_y = []
        
        # For each hill, generate periodic images if ellipse crosses boundaries
        period_x = 2 * np.pi
        period_y = 2 * np.pi
        
        for cx, cy, wx, wy, corr in zip(bias._centers_x, bias._centers_y, 
                                         bias._widths_x, bias._widths_y,
                                         bias._correlations):
            # Compute eigenvalues and rotation angle from covariance matrix
            # Σ = [[σ_x², ρσ_xσ_y], [ρσ_xσ_y, σ_y²]]
            var_x = wx * wx
            var_y = wy * wy
            cov_xy = corr * wx * wy
            
            # Eigenvalues determine the ellipse axes
            trace = var_x + var_y
            det = var_x * var_y - cov_xy * cov_xy
            
            if det > 1e-10:
                # Semi-axes from eigenvalues
                lambda1 = 0.5 * (trace + np.sqrt(trace**2 - 4*det))
                lambda2 = 0.5 * (trace - np.sqrt(trace**2 - 4*det))
                width_major = 2 * np.sqrt(lambda1)  # 1-sigma contour
                width_minor = 2 * np.sqrt(lambda2)
                
                # Rotation angle
                if abs(cov_xy) > 1e-10:
                    angle_rad = 0.5 * np.arctan2(2 * cov_xy, var_x - var_y)
                    angle_deg = np.degrees(angle_rad)
                else:
                    angle_deg = 0.0
            else:
                # Fallback to circular if determinant is bad
                width_major = 2 * wx
                width_minor = 2 * wy
                angle_deg = 0.0
            
            # Determine which periodic images to show based on ellipse extent
            # An ellipse can extend up to its semi-major axis from center
            max_extent = 0.5 * max(width_major, width_minor)
            
            # Check if ellipse crosses boundaries (conservative: check if any part might cross)
            crosses_left = (cx - max_extent) < -np.pi
            crosses_right = (cx + max_extent) > np.pi
            crosses_bottom = (cy - max_extent) < -np.pi
            crosses_top = (cy + max_extent) > np.pi
            
            # Generate list of positions to plot (original + all periodic images needed)
            positions = [(cx, cy, 0, 0)]  # (x, y, shift_x, shift_y) - original position
            
            # Add periodic images in all 9 positions (3x3 grid centered on original)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # Already have original
                    pos_x = cx + dx * period_x
                    pos_y = cy + dy * period_y
                    positions.append((pos_x, pos_y, dx, dy))
            
            # Plot ellipse at each position, but only if it's visible in the plot region
            for pos_x, pos_y, shift_x, shift_y in positions:
                # Check if this ellipse (or part of it) is visible in [-π, π] x [-π, π]
                ellipse_left = pos_x - max_extent
                ellipse_right = pos_x + max_extent
                ellipse_bottom = pos_y - max_extent
                ellipse_top = pos_y + max_extent
                
                # Check for overlap with visible region
                visible = (ellipse_right >= -np.pi and ellipse_left <= np.pi and
                          ellipse_top >= -np.pi and ellipse_bottom <= np.pi)
                
                if visible:
                    # Add center point only for visible positions
                    if -np.pi <= pos_x <= np.pi and -np.pi <= pos_y <= np.pi:
                        all_centers_x.append(pos_x)
                        all_centers_y.append(pos_y)
                    
                    # Draw ellipse (matplotlib will clip automatically)
                    ellipse = Ellipse((pos_x, pos_y), width_major, width_minor, 
                                    angle=angle_deg,
                                    facecolor='none', edgecolor='red', 
                                    linewidth=1.5, alpha=0.6, linestyle='--',
                                    clip_on=True)
                    ax.add_patch(ellipse)
        
        # Plot all centers (original + periodic images)
        ax.scatter(all_centers_x, all_centers_y, 
                  c='red', s=100, marker='x', linewidths=2,
                  label=f'Hill centers (n={len(bias._centers_x)})')

    
    ax.set_xlabel('φ (phi) [rad]', fontsize=12)
    ax.set_ylabel('ψ (psi) [rad]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Forcefully set axis limits to [-π, π]
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    
    # Set ticks at -pi, 0, pi
    ticks = [-np.pi, 0, np.pi]
    tick_labels = ['-π', '0', 'π']
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Bias landscape plot saved: {output_path}")


def create_all_plots(result, bias, output_dir):
    """Create all visualization plots from 2D optimization results.
    
    Args:
        result: Optimization result dictionary
        bias: Bias object
        output_dir: Directory to save plots (Path object)
    """
    print("\n" + "="*80)
    print("GENERATING 2D PLOTS")
    print("="*80)
    
    from pycmaetad.visualization import (
        plot_convergence, plot_parameter_evolution, plot_sigma_evolution,
        plot_convergence_diagnostics, plot_cv_histogram_evolution,
        plot_bias_evolution
    )
    from pycmaetad.visualization.colvar import (
        plot_colvar_evolution, plot_cv_histogram, plot_cv_time_series
    )
    
    # Core convergence plots
    plot_convergence(result, output_path=output_dir / "convergence.png")
    plot_convergence_diagnostics(result, output_path=output_dir / "convergence_diagnostics.png")
    
    # Generate parameter names for 2D bias (6 params per hill)
    n_hills = CONFIG['hills_per_d'] * 2  # hills per dimension × 2 CVs
    param_names = []
    for hill_idx in range(n_hills):
        param_names.extend([
            f'cx_{hill_idx}',    # center_x
            f'cy_{hill_idx}',    # center_y
            f'h_{hill_idx}',     # height
            f'wx_{hill_idx}',    # width_x
            f'wy_{hill_idx}',    # width_y
            f'ρ_{hill_idx}'      # correlation
        ])
    plot_parameter_evolution(result, bias, param_names=param_names, output_path=output_dir / "parameter_evolution.png")
    plot_sigma_evolution(result, output_path=output_dir / "sigma_evolution.png")
    
    # Bias landscape evolution (best individual per generation)
    bias_landscapes_dir = output_dir / "bias_landscapes"
    plot_bias_evolution(
        bias=bias,
        result=result,
        cv_range=CONFIG['cv_range'],
        output_dir=bias_landscapes_dir,
        generations='all',
        show_best_only=True,
        periodic=True
    )
    
    # 2D-specific: CV histogram evolution (Ramachandran plots)
    cv_histograms_dir = output_dir / "ramachandran_evolution"
    plot_cv_histogram_evolution(
        result=result,
        output_dir=cv_histograms_dir,
        generations='all',
        n_bins=CONFIG['n_bins'],
        show_uniform=True,
        cv_range=CONFIG['cv_range'],
        individual='best'
    )
    
    # COLVAR plots (2D trajectory visualization)
    plot_colvar_evolution(result, output_path=output_dir / "colvar_evolution_2d.png", individual='best')
    plot_cv_histogram(result, output_path=output_dir / "ramachandran_final.png", cv_range=CONFIG['cv_range'])
    plot_cv_time_series(result, output_path=output_dir / "cv_time_series_2d.png", individual='best')
    
    # Plot best bias landscape
    print("\n📊 Generating bias landscape plot...")
    # Get best parameters and set them in a temporary bias object for plotting
    best_gen = result['best_generation']
    best_params = result['history'][best_gen]['best_solution']
    bias.set_normalized_parameters(best_params)
    plot_bias_landscape(
        bias,
        output_path=output_dir / "bias_landscape_best.png",
        title=f"Best Bias Landscape (Generation {best_gen})"
    )
    
    print(f"\n✅ All 2D plots saved to: {output_dir}")
    print(f"   📊 Bias landscape evolution: {bias_landscapes_dir}")
    print(f"   📊 Ramachandran evolution plots: {cv_histograms_dir}")


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Alanine Dipeptide 2D CMA-ES Optimization")
    parser.add_argument('command', nargs='?', default='both', 
                       choices=['run', 'resume', 'plot', 'both'],
                       help='Command to execute (default: both)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (default: configs/config_default.py)')
    parser.add_argument('file', nargs='?', default=None,
                       help='Optional file path for resume/plot commands')
    args = parser.parse_args()
    
    # Load configuration
    SCRIPT_DIR = Path(__file__).parent.resolve()
    if args.config is None:
        config_path = SCRIPT_DIR / "configs" / "config_default.py"
    else:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = SCRIPT_DIR / config_path
    
    print(f"\n📋 Loading configuration from: {config_path}")
    CONFIG = load_config(config_path)
    print(f"   Configuration: {CONFIG['name']}")
    print(f"   Description: {CONFIG['description']}\n")
    
    # Execute command
    command = args.command
    
    if command == "run":
        # Run optimization only
        run_optimization()
    elif command == "resume":
        # Resume optimization from checkpoint
        checkpoint_file = args.file
        resume_optimization(checkpoint_file)
    elif command == "plot":
        # Generate plots only
        result_file = args.file
        generate_plots(result_file)
    elif command == "both":
        # Run optimization and generate plots
        result = run_optimization()
        if result is not None:
            generate_plots()
    else:
        print("Unknown command. Usage:")
        print("  python alanine_dipeptide_2d_example.py [--config CONFIG] run              # Run optimization from start")
        print("  python alanine_dipeptide_2d_example.py [--config CONFIG] resume [file]    # Resume from checkpoint")
        print("  python alanine_dipeptide_2d_example.py [--config CONFIG] plot [file]      # Generate plots from saved results")
        print("  python alanine_dipeptide_2d_example.py [--config CONFIG] both             # Run and plot")
