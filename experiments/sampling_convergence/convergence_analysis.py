"""Sampling convergence analysis with multiple replicates.

This script evaluates how well sampled trajectories converge to the true 
Boltzmann distribution with increasing simulation length. 

The analytical Boltzmann distribution of (MB + bias) serves as ground truth,
and KL divergence measures how closely the empirical distribution from 
trajectories matches this ground truth. Multiple replicates provide statistics.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pycmaetad.sampler import MullerBrownSampler
from pycmaetad.bias import MultiGaussian2DForceBias
from pycmaetad.evaluator import UniformKLEvaluator2D


def muller_brown_potential(X, Y):
    """Muller-Brown potential (kJ/mol)."""
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


class AnalyticalMullerBrownEvaluator:
    """Analytical evaluator for Muller-Brown potential.
    
    Computes the true Boltzmann distribution on a grid without MD simulation.
    This serves as ground truth for comparing sampled distributions.
    """
    
    def __init__(self, bias, x_range, y_range, temperature=300.0, n_bins=50, energy_cutoff=100.0):
        """
        Args:
            bias: MultiGaussian2DForceBias instance
            x_range: (x_min, x_max) evaluation range
            y_range: (y_min, y_max) evaluation range
            temperature: Temperature in Kelvin
            n_bins: Number of bins per dimension
            energy_cutoff: Energy cutoff above minimum for accessible region (kJ/mol)
        """
        self.bias = bias
        self.x_range = x_range
        self.y_range = y_range
        self.temperature = temperature
        self.kT = 8.314462e-3 * temperature  # kJ/mol
        self.n_bins = n_bins
        self.energy_cutoff = energy_cutoff
        
        # Create bin edges (same as sampled evaluator)
        self.x_edges = np.linspace(x_range[0], x_range[1], n_bins + 1)
        self.y_edges = np.linspace(y_range[0], y_range[1], n_bins + 1)
        
        # Create evaluation grid (finer than bins)
        n_grid = n_bins * 4  # 4x oversampling
        self.x_grid = np.linspace(x_range[0], x_range[1], n_grid)
        self.y_grid = np.linspace(y_range[0], y_range[1], n_grid)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        
        # Compute Muller-Brown potential once
        self.V_MB = muller_brown_potential(self.X, self.Y)
        
        # Identify accessible region
        self.V_MB_min = np.min(self.V_MB)
        self.accessible_mask = (self.V_MB - self.V_MB_min) < energy_cutoff
        
    def get_analytical_distribution(self, params: np.ndarray) -> np.ndarray:
        """Compute the analytical Boltzmann distribution (binned).
        
        Args:
            params: Denormalized bias parameters
            
        Returns:
            Binned probability distribution (n_bins * n_bins array)
        """
        # Set parameters on bias
        self.bias.set_parameters(params)
        
        # Compute bias on grid
        V_bias = self.bias.evaluate_numpy(self.X, self.Y)
        
        # Total potential
        V_total = self.V_MB + V_bias
        
        # Restrict to accessible region
        V_accessible = V_total[self.accessible_mask]
        X_accessible = self.X[self.accessible_mask]
        Y_accessible = self.Y[self.accessible_mask]
        
        # Compute Boltzmann distribution
        V_min = np.min(V_accessible)
        exp_term = np.exp(-(V_accessible - V_min) / self.kT)
        
        # Normalize
        Z = np.sum(exp_term)
        if Z == 0 or not np.isfinite(Z):
            raise ValueError("Failed to compute partition function")
        
        p_boltzmann_accessible = exp_term / Z
        
        # Bin the probability distribution
        hist, _, _ = np.histogram2d(
            X_accessible, 
            Y_accessible,
            bins=(self.x_edges, self.y_edges),
            weights=p_boltzmann_accessible
        )
        
        # Normalize histogram
        hist_flat = hist.flatten()
        p_binned = hist_flat / np.sum(hist_flat)
        
        return p_binned


def create_test_bias():
    """Create a fixed bias for testing.
    
    Uses a strong multi-Gaussian bias to flatten the MB surface.
    Heights are ~2-3x the well depths to enable barrier crossing.
    """
    bias = MultiGaussian2DForceBias(
        n_gaussians=3,
        height_range=(0, 300),
        center_x_range=(-1.5, 1.5),
        center_y_range=(-0.5, 2.0),
        log_variance_x_range=(-6, 0),
        log_variance_y_range=(-6, 0),
    )
    
    # Set stronger parameters (3 Gaussians near MB wells)
    # MB well depths are ~150-200 kJ/mol, so use heights of 200-250 kJ/mol
    # Wider sigmas (log_var ~ -1.5 → sigma ~ 0.47 nm) for better coverage
    # Parameter order: height, cx, cy, log_var_x, rho, log_var_y
    params = np.array([
        220.0, 0.9, 0.0, -1.5, 0.0, -1.5,    # Well at (1.0, 0.0), no correlation
        200.0, 0.0, 0.5, -1.7, 0.0, -1.7,    # Well at (0.0, 0.5), no correlation
        240.0, -0.5, 1.5, -1.5, 0.0, -1.5,   # Well at (-0.5, 1.5), no correlation
    ])
    
    bias.set_parameters(params)
    return bias, params


def run_convergence_analysis():
    """Run convergence analysis with multiple replicates per sampling time."""
    
    print("\n" + "="*80)
    print("SAMPLING CONVERGENCE ANALYSIS")
    print("Multiple Replicates per Sampling Time")
    print("="*80)
    
    # Configuration
    x_range = (-1.5, 1.5)
    y_range = (-0.5, 2.0)
    temperature = 300.0
    n_bins = 50
    time_step = 0.001  # ps
    
    # Sampling times: 5k, 10k, 15k etc.
    sampling_steps = [5000*s for s in range(1, 26)] # 5k to 125k steps (5 ps to 125 ps)
    sampling_times_ps = [s * time_step for s in sampling_steps]
    n_replicates = 5  # Number of replicates per sampling time
    
    print(f"\nConfiguration:")
    print(f"  CV Range: X={x_range}, Y={y_range}")
    print(f"  Temperature: {temperature} K")
    print(f"  Bins: {n_bins}x{n_bins}")
    print(f"  Time step: {time_step} ps")
    print(f"  Sampling steps: {sampling_steps}")
    print(f"  Sampling times: {[f'{t:.0f}' for t in sampling_times_ps]} ps")
    print(f"  Replicates per time: {n_replicates}")
    print(f"  Total simulations: {len(sampling_steps) * n_replicates}")
    
    # Setup bias
    print(f"\nSetting up bias...")
    bias, params = create_test_bias()
    print(f"  Bias: {bias.n_gaussians} Gaussians")
    
    # Print bias parameters for reference
    for i in range(bias.n_gaussians):
        idx = i * 6
        h, cx, cy, log_vx, log_vy, rho = params[idx:idx+6]
        sx, sy = np.exp(log_vx/2), np.exp(log_vy/2)
        print(f"    Gaussian {i+1}: h={h:.0f} kJ/mol, center=({cx:.2f}, {cy:.2f}), σ=({sx:.3f}, {sy:.3f})")
    
    # Compute analytical Boltzmann distribution (ground truth)
    print(f"\nComputing analytical Boltzmann distribution (ground truth)...")
    analytical_evaluator = AnalyticalMullerBrownEvaluator(
        bias=bias, x_range=x_range, y_range=y_range,
        temperature=temperature, n_bins=n_bins, energy_cutoff=100.0
    )
    analytical_distribution = analytical_evaluator.get_analytical_distribution(params)
    print(f"  Analytical distribution computed: {n_bins}x{n_bins} bins")
    print(f"  Non-zero bins: {np.sum(analytical_distribution > 1e-6)} / {n_bins * n_bins}")
    
    # Also compute analytical KLD to uniform (ground truth bias quality)
    from scipy.stats import entropy
    p_uniform = np.ones_like(analytical_distribution) / len(analytical_distribution)
    analytical_kld_uniform = entropy(analytical_distribution, p_uniform)
    print(f"  Analytical KLD to uniform: {analytical_kld_uniform:.4f} (ground truth bias quality)")
    
    # Create sampled evaluator
    sampled_evaluator = UniformKLEvaluator2D.from_ranges(
        ranges=(x_range, y_range), n_bins=n_bins
    )
    
    # Run simulations
    print(f"\n{'='*80}")
    print("RUNNING SIMULATIONS")
    print("="*80)
    
    # Storage for results
    all_sampled_klds = []  # List of lists: [sampling_time][replicate]
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    for i, (steps, time_ps) in enumerate(zip(sampling_steps, sampling_times_ps)):
        print(f"\n[{i+1}/{len(sampling_steps)}] Sampling time: {time_ps:.0f} ps ({steps} steps)")
        print("-" * 80)
        
        replicate_klds = []
        
        for rep in range(n_replicates):
            print(f"  Replicate {rep+1}/{n_replicates}...", end=" ", flush=True)
            
            # Create sampler
            sampler = MullerBrownSampler(
                temperature=temperature,
                time_step=time_step,
                friction=1.0,
                simulation_steps=steps,
                report_interval=max(1, steps // 1000),
                initial_position=None,  # Random start for each replicate
                cv_range=(x_range, y_range)
            )
            
            # Run simulation
            output_path = output_dir / f"sim_{steps}steps_rep{rep}"
            sampler.run(output_path=str(output_path), bias=bias)
            
            # Evaluate: compute empirical distribution and KLD to analytical
            pdb_file = output_path / "output.pdb"
            
            # Get positions from PDB (simple parser)
            x_coords = []
            y_coords = []
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        # PDB format: x at columns 30-38, y at 38-46
                        # Units are Angstroms in PDB, convert to nm
                        x = float(line[30:38].strip()) / 10.0  # Angstrom to nm
                        y = float(line[38:46].strip()) / 10.0  # Angstrom to nm
                        x_coords.append(x)
                        y_coords.append(y)
            
            x_coords = np.array(x_coords)
            y_coords = np.array(y_coords)
            
            if len(x_coords) == 0:
                print(f"no frames in PDB")
                replicate_klds.append((np.nan, np.nan))
                continue
            
            # Debug: print coordinate ranges
            print(f"coords: x=[{np.min(x_coords):.3f}, {np.max(x_coords):.3f}], y=[{np.min(y_coords):.3f}, {np.max(y_coords):.3f}]")
            print(f"bins: x=[{analytical_evaluator.x_edges[0]:.3f}, {analytical_evaluator.x_edges[-1]:.3f}], y=[{analytical_evaluator.y_edges[0]:.3f}, {analytical_evaluator.y_edges[-1]:.3f}]")
            
            # Compute empirical distribution (histogram)
            hist, _, _ = np.histogram2d(
                x_coords, y_coords,
                bins=(analytical_evaluator.x_edges, analytical_evaluator.y_edges)
            )
            empirical_distribution = hist.flatten()
            
            if np.sum(empirical_distribution) == 0:
                print(f"no points in histogram bins")
                replicate_klds.append((np.nan, np.nan))
                continue
                
            empirical_distribution = empirical_distribution / np.sum(empirical_distribution)
            
            # Compute both KLD metrics from the same empirical distribution
            from scipy.stats import entropy
            
            # 1. KLD to uniform (standard approach - all bins)
            # Uniform distribution over all bins
            p_uniform = np.ones_like(empirical_distribution) / len(empirical_distribution)
            sampled_kld_uniform = entropy(empirical_distribution, p_uniform)
            
            # 2. KLD to analytical Boltzmann (only overlapping bins)
            # Need to mask where q=0 to avoid log(0)
            eps = 1e-10
            mask = (empirical_distribution > eps) & (analytical_distribution > eps)
            
            if np.sum(mask) < 10:
                # Too few overlapping bins
                print(f"Insufficient overlap (only {np.sum(mask)} bins), KLD_uniform={sampled_kld_uniform:.4f}")
                sampled_kld_analytical = np.nan
            else:
                # Renormalize over overlapping bins only
                p = empirical_distribution[mask]
                q = analytical_distribution[mask]
                p = p / np.sum(p)
                q = q / np.sum(q)
                sampled_kld_analytical = entropy(p, q)
                print(f"KLD_analytical={sampled_kld_analytical:.4f}, KLD_uniform={sampled_kld_uniform:.4f}")
            
            replicate_klds.append((sampled_kld_analytical, sampled_kld_uniform))
            
            # Cleanup immediately to save space
            import shutil
            shutil.rmtree(output_path, ignore_errors=True)
        
        all_sampled_klds.append(replicate_klds)
        
        # Print statistics for this sampling time (excluding NaN values)
        klds_analytical = [k[0] for k in replicate_klds if not np.isnan(k[0])]
        klds_uniform = [k[1] for k in replicate_klds if not np.isnan(k[1])]
        
        if len(klds_analytical) > 0:
            mean_analytical = np.mean(klds_analytical)
            std_analytical = np.std(klds_analytical, ddof=1) if len(klds_analytical) > 1 else 0.0
            mean_uniform = np.mean(klds_uniform)
            std_uniform = np.std(klds_uniform, ddof=1) if len(klds_uniform) > 1 else 0.0
            
            print(f"  Statistics: KLD_analytical={mean_analytical:.4f}±{std_analytical:.4f}, KLD_uniform={mean_uniform:.4f}±{std_uniform:.4f} ({len(klds_analytical)}/{n_replicates} valid)")
            print(f"  (Lower KLD_analytical = better sampling; Lower KLD_uniform = better bias quality)")
        else:
            print(f"  Statistics: No valid replicates (all NaN)")
    
    # Compute statistics
    print(f"\n{'='*80}")
    print("COMPUTING STATISTICS")
    print("="*80)
    
    means_analytical = []
    stds_analytical = []
    means_uniform = []
    stds_uniform = []
    
    for klds in all_sampled_klds:
        klds_analytical = [k[0] for k in klds if not np.isnan(k[0])]
        klds_uniform = [k[1] for k in klds if not np.isnan(k[1])]
        
        if len(klds_analytical) > 0:
            means_analytical.append(np.mean(klds_analytical))
            stds_analytical.append(np.std(klds_analytical, ddof=1) if len(klds_analytical) > 1 else 0.0)
        else:
            means_analytical.append(np.nan)
            stds_analytical.append(np.nan)
            
        if len(klds_uniform) > 0:
            means_uniform.append(np.mean(klds_uniform))
            stds_uniform.append(np.std(klds_uniform, ddof=1) if len(klds_uniform) > 1 else 0.0)
        else:
            means_uniform.append(np.nan)
            stds_uniform.append(np.nan)
    
    print(f"\n{'Steps':<10} {'Time (ps)':<12} {'KLD_analytical':<20} {'KLD_uniform':<20}")
    print("-" * 80)
    for i, (steps, time_ps, mean_a, std_a, mean_u, std_u) in enumerate(
        zip(sampling_steps, sampling_times_ps, means_analytical, stds_analytical, means_uniform, stds_uniform)
    ):
        if np.isnan(mean_a):
            analytical_str = "NaN"
        else:
            analytical_str = f"{mean_a:.4f}±{std_a:.4f}"
            
        if np.isnan(mean_u):
            uniform_str = "NaN"
        else:
            uniform_str = f"{mean_u:.4f}±{std_u:.4f}"
            
        print(f"{steps:<10} {time_ps:<12.0f} {analytical_str:<20} {uniform_str:<20}")
    
    # Create plot
    print(f"\n{'='*80}")
    print("GENERATING PLOT")
    print("="*80)
    
    # Filter out NaN values for plotting
    valid_indices_analytical = [i for i, m in enumerate(means_analytical) if not np.isnan(m)]
    valid_indices_uniform = [i for i, m in enumerate(means_uniform) if not np.isnan(m)]
    
    if len(valid_indices_analytical) == 0 and len(valid_indices_uniform) == 0:
        print("  Warning: No valid data points to plot!")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot KLD to analytical (sampling quality)
    if len(valid_indices_analytical) > 0:
        valid_times_a = [sampling_times_ps[i] for i in valid_indices_analytical]
        valid_means_a = [means_analytical[i] for i in valid_indices_analytical]
        valid_stds_a = [stds_analytical[i] for i in valid_indices_analytical]
        
        ax.errorbar(valid_times_a, valid_means_a, yerr=valid_stds_a, 
                    fmt='o-', linewidth=2.5, markersize=10, 
                    capsize=5, capthick=2, 
                    label='KLD to analytical Boltzmann', 
                    color='#2E86AB', ecolor='#2E86AB', 
                    elinewidth=2, alpha=0.9)
    
    # Plot KLD to uniform (bias quality)
    if len(valid_indices_uniform) > 0:
        valid_times_u = [sampling_times_ps[i] for i in valid_indices_uniform]
        valid_means_u = [means_uniform[i] for i in valid_indices_uniform]
        valid_stds_u = [stds_uniform[i] for i in valid_indices_uniform]
        
        ax.errorbar(valid_times_u, valid_means_u, yerr=valid_stds_u, 
                    fmt='s--', linewidth=2.5, markersize=10, 
                    capsize=5, capthick=2, 
                    label='KLD to uniform', 
                    color='#A23B72', ecolor='#A23B72', 
                    elinewidth=2, alpha=0.9)
    
    ax.set_xlabel('Sampling Time (ps)', fontsize=14, fontweight='bold')
    ax.set_ylabel('KL Divergence', fontsize=14, fontweight='bold')
    ax.set_title('Sampling Convergence Analysis\nMuller-Brown Surface with Fixed Bias',
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(fontsize=13, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, which='both', linestyle=':')
    
    # Add text box with key statistics
    textstr_lines = [
        f'Temperature: {temperature} K',
        f'Bins: {n_bins}×{n_bins}',
        f'Replicates: {n_replicates}',
    ]
    
    textstr = '\n'.join(textstr_lines)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    # Save plot to output directory
    output_plot_path = output_dir / "sampling_convergence_analysis.png"
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_plot_path}")
    
    # Save data to output directory
    data_path = output_dir / "convergence_data.npz"
    np.savez(data_path,
             sampling_steps=sampling_steps,
             sampling_times_ps=sampling_times_ps,
             analytical_distribution=analytical_distribution,
             analytical_kld_uniform=analytical_kld_uniform,
             means_analytical=means_analytical,
             stds_analytical=stds_analytical,
             means_uniform=means_uniform,
             stds_uniform=stds_uniform,
             all_sampled_klds=all_sampled_klds)
    print(f"✓ Data saved to: {data_path}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nKey findings:")
    
    if len(valid_indices_analytical) > 0:
        first_idx = valid_indices_analytical[0]
        last_idx = valid_indices_analytical[-1]
        print(f"\n  Sampling Quality (KLD to analytical Boltzmann):")
        print(f"    - Initial: {means_analytical[first_idx]:.4f} ± {stds_analytical[first_idx]:.4f}")
        print(f"    - Final: {means_analytical[last_idx]:.4f} ± {stds_analytical[last_idx]:.4f}")
        improvement_a = means_analytical[first_idx] - means_analytical[last_idx]
        improvement_pct_a = (improvement_a / means_analytical[first_idx] * 100) if means_analytical[first_idx] > 0 else 0
        print(f"    - Improvement: {improvement_a:.4f} ({improvement_pct_a:.1f}%)")
        print(f"    - Trend: {'Improving' if means_analytical[last_idx] < means_analytical[first_idx] else 'Variable'}")
    else:
        print(f"\n  Sampling Quality: No valid KLD_analytical values")
    
    if len(valid_indices_uniform) > 0:
        first_idx = valid_indices_uniform[0]
        last_idx = valid_indices_uniform[-1]
        print(f"\n  Bias Quality (KLD to uniform):")
        print(f"    - Analytical (ground truth): {analytical_kld_uniform:.4f}")
        print(f"    - Sampled initial: {means_uniform[first_idx]:.4f} ± {stds_uniform[first_idx]:.4f}")
        print(f"    - Sampled final: {means_uniform[last_idx]:.4f} ± {stds_uniform[last_idx]:.4f}")
        print(f"    - Reflects how flat the biased landscape is")
    else:
        print(f"\n  Bias Quality: No valid KLD_uniform values")
    
    print(f"\nInterpretation:")
    print(f"  - KLD to analytical: Measures sampling convergence (empirical vs true Boltzmann)")
    print(f"    → Lower = better sampling quality")
    print(f"    → Should decrease with longer simulation time")
    print(f"  - KLD to uniform: Measures bias effectiveness (used in optimizer)")
    print(f"    → Lower = flatter landscape = better bias")
    print(f"    → Should be constant (same bias, different sampling length)")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_convergence_analysis()
