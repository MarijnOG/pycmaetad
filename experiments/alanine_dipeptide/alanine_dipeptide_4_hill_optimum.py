"""Find optimal 4-hill bias for alanine dipeptide using FES from metadynamics.

This script uses CMA-ES to optimize bias parameters to match a target free energy surface,
WITHOUT running any MD simulations. It directly optimizes the bias landscape to approximate
the negative of the FES, so that FES + bias ≈ constant (uniform sampling).

Input:
    - Target FES from metadynamics (fes_diala.dat)
    - 4 Gaussian hills (2×2 grid) with 6 parameters each (24 total)
    
Parameters per hill (2D):
    - center_phi, center_psi: Hill center position
    - height: Hill height (kJ/mol)
    - width_phi, width_psi: Width in each dimension
    - correlation: Correlation coefficient ρ ∈ [-1, 1]

Output:
    - Optimal bias parameters
    - Visualization of bias vs FES
    - Comparison of FES + bias (should be flat)
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pycmaetad.bias import PlumedHillBias2D
from pycmaetad.evaluator import FESMatchingEvaluator
from pycmaetad.optimizer import CMAESWorkflow
from pycmaetad.visualization import plot_parameter_evolution


# ============================================================================
# CONFIGURATION
# ============================================================================

# FES file (from metadynamics simulation)
FES_FILE = "/home/marijn/AfstudeerprojectInformatica/fes_diala.dat"

# Bias configuration
HILLS_PER_D = 2  # 2×2 = 4 hills
HILLS_SPACE = ((-np.pi, np.pi), (-np.pi, np.pi))
HILLS_HEIGHT = 80.0  # Initial guess - will be optimized (increased from 50)
HILLS_WIDTH = [2.5, 2.5]  # Initial guess - will be optimized (increased from 1.2 to allow broader hills)
MIN_WIDTH = 0.2  # Minimum width to prevent degenerate narrow Gaussians (decreased from 0.3)
MULTIVARIATE = True  # Use correlation

# CMA-ES configuration
SIGMA = 0.25 
MAX_GENERATIONS = 500  # More generations for pure optimization
POPULATION_SIZE = 36 
EARLY_STOP_PATIENCE = 40  # Stop if no improvement for N generations (increased from 30)
EARLY_STOP_THRESHOLD = 0.0003

# Temperature (for Boltzmann weighting)
TEMPERATURE = 300.0  # K


def create_initial_mean_from_fes(fes_evaluator, hills_per_d):
    """Create initial mean by placing hills at FES maxima.
    
    Strategy: Place hills at high-energy regions of FES to flatten them.
    
    Args:
        fes_evaluator: FESMatchingEvaluator with loaded FES
        hills_per_d: Number of hills per dimension
        
    Returns:
        Initial mean in normalized [0, 1] space
    """
    n_hills_total = hills_per_d ** 2
    n_params_total = n_hills_total * 6
    
    mean = np.ones(n_params_total) * 0.5  # Default to middle
    
    # Find high-energy regions in FES
    fes = fes_evaluator.fes_grid_shifted
    phi_grid = fes_evaluator.phi_grid
    psi_grid = fes_evaluator.psi_grid
    
    # Find local maxima by simple approach: divide grid into quadrants
    # and place one hill at the maximum in each quadrant
    phi_mid = 0.0
    psi_mid = 0.0
    
    quadrants = [
        (phi_grid < phi_mid) & (psi_grid < psi_mid),  # Bottom-left
        (phi_grid >= phi_mid) & (psi_grid < psi_mid),  # Bottom-right
        (phi_grid < phi_mid) & (psi_grid >= psi_mid),  # Top-left
        (phi_grid >= phi_mid) & (psi_grid >= psi_mid),  # Top-right
    ]
    
    for i, mask in enumerate(quadrants[:n_hills_total]):
        if np.any(mask):
            # Find maximum FES value in this quadrant
            fes_masked = np.where(mask, fes, -np.inf)
            max_idx = np.unravel_index(np.argmax(fes_masked), fes.shape)
            
            # Get phi, psi at this position
            phi_max = phi_grid[max_idx]
            psi_max = psi_grid[max_idx]
            
            # Normalize to [0, 1]
            phi_norm = (phi_max - (-np.pi)) / (2 * np.pi)
            psi_norm = (psi_max - (-np.pi)) / (2 * np.pi)
            
            # Set in mean vector
            param_start = i * 6
            mean[param_start + 0] = phi_norm  # center_phi
            mean[param_start + 1] = psi_norm  # center_psi
            # height, widths, correlation stay at 0.5 (mid-range)
    
    print("\n🎯 Initial hill placement (at FES maxima):")
    for i in range(n_hills_total):
        param_start = i * 6
        phi_norm = mean[param_start]
        psi_norm = mean[param_start + 1]
        phi_real = -np.pi + phi_norm * 2 * np.pi
        psi_real = -np.pi + psi_norm * 2 * np.pi
        print(f"  Hill {i}: φ={phi_real:.2f}, ψ={psi_real:.2f} rad")
    
    return mean


def run_optimization():
    """Run CMA-ES optimization to find optimal bias parameters."""
    print("\n" + "="*80)
    print("ALANINE DIPEPTIDE 4-HILL BIAS OPTIMIZATION")
    print("Finding optimal bias to match FES (no MD simulations)")
    print("="*80)
    print(f"FES file: {FES_FILE}")
    print(f"Hills: {HILLS_PER_D}×{HILLS_PER_D} = {HILLS_PER_D**2}")
    print(f"Parameters: {HILLS_PER_D**2 * 6}")
    print("="*80 + "\n")
    
    # Create bias object (dummy plumed template - not used for evaluation)
    script_dir = Path(__file__).parent
    plumed_template = script_dir / "plumed_template_2d.dat"
    
    if not plumed_template.exists():
        print(f"⚠️  Warning: plumed template not found at {plumed_template}")
        print("Creating minimal template...")
        plumed_template.parent.mkdir(exist_ok=True, parents=True)
        # Create minimal template (won't be used for evaluation anyway)
        with open(plumed_template, 'w') as f:
            f.write("# Placeholder template for optimization\n")
            f.write("phi: TORSION ATOMS=5,7,9,15\n")
            f.write("psi: TORSION ATOMS=7,9,15,17\n")
    
    bias = PlumedHillBias2D(
        plumed_template=str(plumed_template),
        hills_per_d=HILLS_PER_D,
        hills_space=HILLS_SPACE,
        hills_height=HILLS_HEIGHT,
        hills_width=HILLS_WIDTH,
        multivariate=MULTIVARIATE
    )
    
    # Create evaluator (loads FES)
    print("📊 Loading FES...")
    evaluator = FESMatchingEvaluator(FES_FILE, bias=bias, temperature=TEMPERATURE)
    print(f"✓ FES loaded: {evaluator.phi_grid.shape} grid points")
    print(f"  FES range: [{np.min(evaluator.fes_grid):.2f}, {np.max(evaluator.fes_grid):.2f}] kJ/mol")
    print(f"  Evaluator mode: {'Analytical (no MD)' if not evaluator.requires_simulation else 'MD Simulation'}")
    
    # Create initial mean
    initial_mean = create_initial_mean_from_fes(evaluator, HILLS_PER_D)
    
    # Evaluate initial guess
    print("\n📊 Initial evaluation:")
    initial_params = bias.get_parameter_bounds()[:, 0] + initial_mean * (bias.get_parameter_bounds()[:, 1] - bias.get_parameter_bounds()[:, 0])
    initial_score = evaluator.evaluate(initial_params)
    print(f"  Initial KL divergence: {initial_score:.6f}")
    
    # Override parameter bounds to enforce minimum widths (prevent degenerate narrow Gaussians)
    custom_bounds = bias.get_parameter_bounds()
    n_hills = HILLS_PER_D ** 2
    for i in range(n_hills):
        idx = i * 6
        # Enforce minimum widths in both dimensions
        custom_bounds[idx + 3] = [MIN_WIDTH, np.pi]  # width_x
        custom_bounds[idx + 4] = [MIN_WIDTH, np.pi]  # width_y
    
    print(f"\n🔧 Custom parameter bounds:")
    print(f"  Heights: [0, {HILLS_HEIGHT}] kJ/mol")
    print(f"  Widths:  [{MIN_WIDTH}, {np.pi}] rad (enforcing minimum to prevent pathological narrow hills)")
    print(f"  Correlation: [-1, 1]")
    
    # Create workflow (no sampler needed for analytical evaluation)
    print("\n🚀 Creating CMA-ES workflow...")
    workflow = CMAESWorkflow(
        bias=bias,
        sampler=None,  # No sampler needed for analytical evaluation
        evaluator=evaluator,
        initial_mean=initial_mean,
        sigma=SIGMA,
        population_size=POPULATION_SIZE,
        bounds=custom_bounds,  # Use custom bounds with minimum width constraint
        max_generations=MAX_GENERATIONS,
        n_workers=1,  # Analytical evaluation is fast, no need for parallelization
        early_stop_patience=EARLY_STOP_PATIENCE,
        early_stop_threshold=EARLY_STOP_THRESHOLD
    )
    
    # Run optimization
    output_dir = script_dir / "output_4hill_optimum"
    result = workflow.optimize(str(output_dir))
    
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
    print(f"\nBest score: {result['best_score']:.6f} (generation {result['best_generation']})")
    print(f"Initial score: {initial_score:.6f}")
    print(f"Improvement: {initial_score - result['best_score']:.6f} ({100*(initial_score - result['best_score'])/initial_score:.1f}%)")
    
    # Print best parameters
    print("\n📊 Best bias parameters:")
    best_params = result['best_parameters']
    
    for i in range(HILLS_PER_D ** 2):
        param_start = i * 6
        phi_c = best_params[param_start]
        psi_c = best_params[param_start + 1]
        height = best_params[param_start + 2]
        width_phi = best_params[param_start + 3]
        width_psi = best_params[param_start + 4]
        corr = best_params[param_start + 5]
        print(f"  Hill {i}:")
        print(f"    Center: (φ={phi_c:.2f}, ψ={psi_c:.2f}) rad")
        print(f"    Height: {height:.1f} kJ/mol")
        print(f"    Widths: (σ_φ={width_phi:.2f}, σ_ψ={width_psi:.2f}) rad")
        print(f"    Correlation: {corr:.2f}")
    
    # Save results
    result_file = output_dir / "optimization_result.pkl"
    with open(result_file, 'wb') as f:
        pickle.dump({'result': result, 'bias': bias, 'evaluator': evaluator}, f)
    
    print(f"\n✓ Results saved to: {result_file}")
    
    return result, bias, evaluator


def plot_results(result, bias, evaluator, output_dir):
    """Generate visualization plots.
    
    Args:
        result: Optimization result dictionary
        bias: PlumedHillBias2D with best parameters
        evaluator: FESMatchingEvaluator
        output_dir: Directory to save plots
    """
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Convergence plot
    fig, ax = plt.subplots(figsize=(10, 6))
    history = result['history']
    generations = np.arange(len(history))
    best_scores = [h['best_score'] for h in history]
    mean_scores = [h['mean_score'] for h in history]
    
    ax.plot(generations, best_scores, 'b-', linewidth=2, label='Best')
    ax.plot(generations, mean_scores, 'r--', linewidth=1, label='Mean')
    ax.axhline(result['best_score'], color='g', linestyle=':', label='Final best')
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('KL Divergence', fontsize=12)
    ax.set_title('CMA-ES Convergence', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "convergence.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Convergence plot saved")
    
    # 2. FES visualization - 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Set best parameters on bias
    bias.set_parameters(result['best_parameters'])
    
    # Get FES grid - keep original orientation (matches literature)
    phi_grid_fes = evaluator.phi_grid
    psi_grid_fes = evaluator.psi_grid
    fes_grid = evaluator.fes_grid
    
    # Use bias.compute_bias_landscape() to create a complete periodic grid from -π to +π
    # This avoids artifacts from the incomplete FES grid
    phi_grid_bias, psi_grid_bias, bias_grid = bias.compute_bias_landscape(
        n_points=(100, 100),
        periodic=True
    )
    
    # Transpose bias to match FES orientation (FES matches literature)
    # FES: phi on axis 1, psi on axis 0
    # Bias: phi on axis 0, psi on axis 1 → transpose to match
    phi_grid_bias = phi_grid_bias.T
    psi_grid_bias = psi_grid_bias.T
    bias_grid = bias_grid.T
    
    # For combined plot, interpolate FES onto bias grid since bias has full periodic coverage
    from scipy.interpolate import RegularGridInterpolator
    
    # Create interpolator for FES
    # FES grid: phi varies along axis 1, psi varies along axis 0
    phi_1d_fes = phi_grid_fes[0, :]  # Extract phi values (constant along columns)
    psi_1d_fes = psi_grid_fes[:, 0]  # Extract psi values (constant along rows)
    
    fes_interpolator = RegularGridInterpolator(
        (psi_1d_fes, phi_1d_fes),  # Note: (psi, phi) order matches (axis0, axis1)
        fes_grid,
        method='linear',
        bounds_error=False,
        fill_value=None  # Extrapolate for missing region
    )
    
    # Evaluate FES on bias grid points
    # Bias grid after transpose: phi on axis 1, psi on axis 0 (same as FES)
    points = np.stack([psi_grid_bias, phi_grid_bias], axis=-1)  # Shape: (100, 100, 2)
    fes_on_bias_grid = fes_interpolator(points)
    combined_grid = fes_on_bias_grid + bias_grid
    
    # Plot FES
    ax = axes[0]
    levels = 20
    contourf = ax.contourf(phi_grid_fes, psi_grid_fes, fes_grid, levels=levels, cmap='viridis')
    ax.contour(phi_grid_fes, psi_grid_fes, fes_grid, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contourf, ax=ax, label='FES (kJ/mol)')
    ax.set_xlabel('φ (phi) [rad]', fontsize=12)
    ax.set_ylabel('ψ (psi) [rad]', fontsize=12)
    ax.set_title('Target FES (from Metadynamics)', fontsize=14, fontweight='bold')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect('equal')
    
    # Plot optimal bias (use complete periodic grid from bias.compute_bias_landscape)
    # Use raw grids without transpose
    ax = axes[1]
    contourf = ax.contourf(phi_grid_bias, psi_grid_bias, bias_grid, levels=levels, cmap='RdYlBu_r')
    ax.contour(phi_grid_bias, psi_grid_bias, bias_grid, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contourf, ax=ax, label='Bias (kJ/mol)')
    
    # Add hill centers
    if bias._centers_x is not None:
        ax.scatter(bias._centers_x, bias._centers_y, 
                  c='red', s=150, marker='x', linewidths=3,
                  label=f'Hill centers (n={len(bias._centers_x)})', zorder=5)
    
    ax.set_xlabel('φ (phi) [rad]', fontsize=12)
    ax.set_ylabel('ψ (psi) [rad]', fontsize=12)
    ax.set_title('Optimized Bias Potential', fontsize=14, fontweight='bold')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    # Plot FES + Bias (should be flat) - use bias grid which has full periodic coverage
    ax = axes[2]
    contourf = ax.contourf(phi_grid_bias, psi_grid_bias, combined_grid, levels=levels, cmap='coolwarm')
    ax.contour(phi_grid_bias, psi_grid_bias, combined_grid, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contourf, ax=ax, label='FES + Bias (kJ/mol)')
    ax.set_xlabel('φ (phi) [rad]', fontsize=12)
    ax.set_ylabel('ψ (psi) [rad]', fontsize=12)
    ax.set_title('Combined Landscape (FES + Bias)', fontsize=14, fontweight='bold')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / "fes_bias_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ FES/bias comparison saved")
    
    # 3. 1D projections
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Phi projection (average over psi, which is axis 0 for transposed grids)
    ax = axes[0]
    fes_on_bias_phi = np.mean(fes_on_bias_grid, axis=0)
    bias_phi = np.mean(bias_grid, axis=0)
    combined_phi = np.mean(combined_grid, axis=0)
    
    phi_1d = phi_grid_bias[0, :]
    ax.plot(phi_1d, fes_on_bias_phi, 'b-', linewidth=2, label='FES')
    ax.plot(phi_1d, bias_phi, 'r-', linewidth=2, label='Bias')
    ax.plot(phi_1d, combined_phi, 'g--', linewidth=2, label='FES + Bias')
    ax.axhline(np.mean(combined_phi), color='gray', linestyle=':', label='Mean combined')
    ax.set_xlabel('φ (phi) [rad]', fontsize=12)
    ax.set_ylabel('Energy (kJ/mol)', fontsize=12)
    ax.set_title('Phi (φ) Projection', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-np.pi, np.pi)
    
    # Psi projection (average over phi, which is axis 1 for transposed grids)
    ax = axes[1]
    fes_on_bias_psi = np.mean(fes_on_bias_grid, axis=1)
    bias_psi = np.mean(bias_grid, axis=1)
    combined_psi = np.mean(combined_grid, axis=1)
    
    psi_1d = psi_grid_bias[:, 0]
    ax.plot(psi_1d, fes_on_bias_psi, 'b-', linewidth=2, label='FES')
    ax.plot(psi_1d, bias_psi, 'r-', linewidth=2, label='Bias')
    ax.plot(psi_1d, combined_psi, 'g--', linewidth=2, label='FES + Bias')
    ax.axhline(np.mean(combined_psi), color='gray', linestyle=':', label='Mean combined')
    ax.set_xlabel('ψ (psi) [rad]', fontsize=12)
    ax.set_ylabel('Energy (kJ/mol)', fontsize=12)
    ax.set_title('Psi (ψ) Projection', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-np.pi, np.pi)
    
    plt.tight_layout()
    plt.savefig(output_dir / "1d_projections.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 1D projections saved")
    
    # 4. Parameter evolution
    # Create parameter names for 4 hills × 6 params each
    param_names = []
    for i in range(HILLS_PER_D ** 2):
        param_names.extend([
            f'H{i}_φ_c',      # center phi
            f'H{i}_ψ_c',      # center psi
            f'H{i}_height',   # height (kJ/mol)
            f'H{i}_σ_φ',      # width phi (rad)
            f'H{i}_σ_ψ',      # width psi (rad)
            f'H{i}_ρ',        # correlation
        ])
    
    plot_parameter_evolution(
        result, bias, 
        param_names=param_names,
        output_path=output_dir / "parameter_evolution.png",
        show=False
    )
    
    print(f"\n✅ All plots saved to: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "run":
            result, bias, evaluator = run_optimization()
            output_dir = Path(__file__).parent / "output_4hill_optimum"
            plot_results(result, bias, evaluator, output_dir)
        elif command == "plot":
            # Load saved results
            result_file = Path(__file__).parent / "output_4hill_optimum" / "optimization_result.pkl"
            if not result_file.exists():
                print(f"Error: No saved results found at {result_file}")
                print("Run optimization first with: python alanine_dipeptide_4_hill_optimum.py run")
                sys.exit(1)
            
            with open(result_file, 'rb') as f:
                data = pickle.load(f)
            
            result = data['result']
            bias = data['bias']
            evaluator = data['evaluator']
            
            output_dir = Path(__file__).parent / "output_4hill_optimum"
            plot_results(result, bias, evaluator, output_dir)
        else:
            print("Usage:")
            print("  python alanine_dipeptide_4_hill_optimum.py run   # Run optimization and plot")
            print("  python alanine_dipeptide_4_hill_optimum.py plot  # Plot saved results")
    else:
        # Default: run and plot
        result, bias, evaluator = run_optimization()
        output_dir = Path(__file__).parent / "output_4hill_optimum"
        plot_results(result, bias, evaluator, output_dir)
