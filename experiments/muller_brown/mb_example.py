"""
Muller-Brown CMA-ES Optimization Experiment
==========================================

This script runs a CMA-ES optimization of bias parameters for uniform sampling
on the 2D Muller-Brown potential using multiple Gaussians as the bias.
It supports running, resuming, and plotting results for different configurations.

Features:
    - Flexible configuration via Python config files
    - End-to-end optimization workflow (ask/run/evaluate/tell)
    - Uniform sampling in 2D potential energy landscape
    - Diagnostic plots: convergence, parameter evolution, bias landscape, histograms

Usage:
    python mb_example.py --config configs/config_tight.py run
    python mb_example.py --config configs/config_wide.py both
    python mb_example.py run  # Uses default config

Arguments:
    --config   Path to configuration file (default: configs/config_tight.py)
    run        Run optimization only
    resume     Resume optimization from checkpoint file
    plot       Generate plots from saved results
    both       Run optimization and generate plots (default)

Outputs:
    - Optimization results and checkpoints
    - Diagnostic plots: convergence, parameter evolution, bias landscape, histograms
    - Pickled result and bias objects for further analysis
"""

import sys
import pickle
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Ellipse

# Import pycmaetad modules for sampling, bias, evaluation, and optimization
from pycmaetad.sampler import MullerBrownSampler
from pycmaetad.bias import MultiGaussian2DForceBias
from pycmaetad.evaluator import UniformKLEvaluator2D
from pycmaetad.evaluator.base import Evaluator
from pycmaetad.optimizer import CMAESWorkflow


# =================== CONFIGURATION LOADING ===================

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
        raise ValueError(f"Config file must define a CONFIG dictionary: {config_path}")
    
    return config_module.CONFIG


# Global config (loaded in main)
CONFIG = None


# =================== ANALYTICAL EVALUATOR ===================

class AnalyticalMullerBrownEvaluator(Evaluator):
    """Analytical evaluator for Muller-Brown potential.
    
    Computes variance of the biased potential (V_MB + V_bias) over the accessible region.
    Lower variance = flatter landscape = better for uniform sampling.
    
    This directly measures what the bias achieves (landscape flattening) without
    the equilibrium vs finite-time sampling mismatch of KL divergence comparisons.
    """
    
    @property
    def requires_simulation(self) -> bool:
        """This evaluator does not require MD simulation."""
        return False
    
    def __init__(self, bias, x_range, y_range, temperature=300.0, n_bins=50, energy_cutoff=200.0):
        """
        Args:
            bias: MultiGaussian2DForceBias instance
            x_range: (x_min, x_max) evaluation range
            y_range: (y_min, y_max) evaluation range
            temperature: Temperature in Kelvin (for Boltzmann distribution)
            n_bins: Number of bins per dimension (match sampled evaluator)
            energy_cutoff: Energy cutoff above minimum for accessible region (kJ/mol)
                          Default 200 kJ/mol allows inclusion of high-barrier regions
                          that metadynamics aims to explore (typical range: 150-300 kJ/mol)
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
        
        # Create evaluation grid (finer than bins for accurate integration)
        n_grid = n_bins * 4  # 4x oversampling
        self.x_grid = np.linspace(x_range[0], x_range[1], n_grid)
        self.y_grid = np.linspace(y_range[0], y_range[1], n_grid)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        
        # Compute Muller-Brown potential once (doesn't change)
        self.V_MB = self._muller_brown_potential(self.X, self.Y)
        
        # Identify accessible region (energy within cutoff of minimum)
        self.V_MB_min = np.min(self.V_MB)
        self.accessible_mask = (self.V_MB - self.V_MB_min) < energy_cutoff
        
        print(f"  Analytical evaluator setup:")
        print(f"    Grid: {n_grid}×{n_grid}, Bins: {n_bins}×{n_bins}")
        print(f"    Energy cutoff: {energy_cutoff} kJ/mol above minimum")
        print(f"    Accessible region: {np.sum(self.accessible_mask)/self.accessible_mask.size*100:.1f}% of grid")
        
    def _muller_brown_potential(self, X, Y):
        """Vectorized Muller-Brown potential (kJ/mol)."""
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
    
    def _compute_bias_on_grid(self):
        """Compute bias potential on the grid using current parameters."""
        # Use the bias object's built-in evaluation method
        return self.bias.evaluate_numpy(self.X, self.Y)
    
    def evaluate(self, params: np.ndarray, debug=False) -> float:
        """Evaluate variance of biased potential analytically.
        
        Args:
            params: Denormalized bias parameters
            debug: If True, print diagnostic information
            
        Returns:
            Variance of (V_MB + V_bias) over accessible region (kJ/mol)²
            Lower variance = flatter landscape = better
        """
        # Set parameters on bias
        self.bias.set_parameters(params)
        
        # Compute bias on grid
        V_bias = self._compute_bias_on_grid()
        
        # Total potential = Muller-Brown + bias
        V_total = self.V_MB + V_bias
        
        # Restrict to accessible region (based on unbiased MB potential)
        V_min_MB = np.min(self.V_MB)
        accessible_mask = (self.V_MB - V_min_MB) < self.energy_cutoff
        
        # Get biased potential in accessible region
        V_total_accessible = V_total[accessible_mask]
        
        # Compute variance (lower = flatter = better)
        variance = np.var(V_total_accessible)
        
        if debug:
            V_min = np.min(V_total_accessible)
            V_max = np.max(V_total_accessible)
            V_range = V_max - V_min
            V_std = np.sqrt(variance)
            print(f"  V_MB range (full): [{np.min(self.V_MB):.1f}, {np.max(self.V_MB):.1f}] kJ/mol")
            print(f"  V_bias range: [{np.min(V_bias):.1f}, {np.max(V_bias):.1f}] kJ/mol")
            print(f"  Accessible region: {np.sum(accessible_mask)} / {accessible_mask.size} grid points ({np.sum(accessible_mask)/accessible_mask.size*100:.1f}%)")
            print(f"  V_total (accessible) range: [{V_min:.1f}, {V_max:.1f}] kJ/mol")
            print(f"  V_total range (span): {V_range:.1f} kJ/mol")
            print(f"  V_total std: {V_std:.1f} kJ/mol")
            print(f"  V_total variance: {variance:.1f} (kJ/mol)²")
        
        if not np.isfinite(variance):
            return 1e6
        
        return variance


# =================== HELPER FUNCTIONS ===================


def create_initial_mean_for_wells(bias, well_positions, initial_height=500.0, initial_sigma=0.2):
    """Create initial mean vector for CMA-ES seeded near well positions.

    Args:
        bias: MultiGaussian2DForceBias instance (to get parameter ranges)
        well_positions: List of (x, y) tuples for well centers
        initial_height: Initial height value (kJ/mol)
        initial_sigma: Initial sigma value (nm)

    Returns:
        Initial mean vector in normalized [0,1] space
    """
    n_gaussians = bias.n_gaussians
    n_params = n_gaussians * 6  # 6 parameters per Gaussian

    mean = np.ones(n_params) * 0.5  # Default to middle

    # Set Gaussian centers near wells
    for i in range(min(n_gaussians, len(well_positions))):
        x_well, y_well = well_positions[i]

        # Convert from physical coordinates to [0,1] space
        # center_param = (center_value - range_min) / (range_max - range_min)
        x_range = bias.center_x_range
        y_range = bias.center_y_range

        mean[i * 6 + 0] = initial_height / \
            (bias.height_range[1] - bias.height_range[0])  # height
        mean[i * 6 + 1] = (x_well - x_range[0]) / \
                           (x_range[1] - x_range[0])  # center_x
        mean[i * 6 + 2] = (y_well - y_range[0]) / \
                           (y_range[1] - y_range[0])  # center_y

        # Log variance
        log_var = np.log(initial_sigma**2)
        log_var_x_range = bias.log_variance_x_range
        log_var_y_range = bias.log_variance_y_range

        mean[i * 6 + 3] = (log_var - log_var_x_range[0]) / \
                           (log_var_x_range[1] -
                            log_var_x_range[0])  # log_var_x
        mean[i * 6 + 4] = (log_var - log_var_y_range[0]) / \
                           (log_var_y_range[1] -
                            log_var_y_range[0])  # log_var_y
        mean[i * 6 + 5] = 0.5  # rho (no correlation initially)

    return mean


# =================== OPTIMIZATION FUNCTIONS ===================


def run_optimization():
    """
    Run the CMA-ES optimization and save results.
    """
    SCRIPT_DIR = Path(__file__).parent.resolve()

    # ---- Logging: Run info ----
    print("\n" + "="*60)
    print("MULLER-BROWN CMA-ES OPTIMIZATION")
    print("Optimizing multi-Gaussian bias for uniform sampling")
    print("="*60)
    print(f"Configuration: {CONFIG['name']}")
    print(f"Description: {CONFIG['description']}")
    print(f"Working directory: {SCRIPT_DIR}")
    print(f"Simulation time: {CONFIG['simulation_steps']} steps ({CONFIG['simulation_steps']*CONFIG['time_step']:.1f} ps)")
    print(f"Population size: {CONFIG['population_size']}")
    print(f"Evaluation region: X={CONFIG['evaluation_x_range']}, Y={CONFIG['evaluation_y_range']}")
    print("="*60 + "\n")

    # ---- Use FULL BOX for evaluation ----
    evaluation_ranges = [(CONFIG['evaluation_x_range'][0], CONFIG['evaluation_x_range'][1]),
                         (CONFIG['evaluation_y_range'][0], CONFIG['evaluation_y_range'][1])]

    # ---- Create bias ----
    bias = MultiGaussian2DForceBias(
        n_gaussians=CONFIG['num_gaussians'],
        height_range=CONFIG['height_range'],
        center_x_range=CONFIG['center_x_range'],
        center_y_range=CONFIG['center_y_range'],
        log_variance_x_range=CONFIG['log_variance_x_range'],
        log_variance_y_range=CONFIG['log_variance_y_range'],
    )

    # ---- Compute initial mean seeded near well positions ----
    initial_mean = create_initial_mean_for_wells(
        bias=bias,
        well_positions=CONFIG['well_positions'],
        initial_height=CONFIG['initial_height'],
        initial_sigma=CONFIG['initial_sigma']
    )

    # ---- Create sampler ----
    # Note: If initial_position=None, sampler will randomize starting position per evaluation
    sampler = MullerBrownSampler(
        temperature=CONFIG['temperature'],
        time_step=CONFIG['time_step'],
        friction=CONFIG['friction'],
        simulation_steps=CONFIG['simulation_steps'],
        report_interval=CONFIG['report_interval'],
        initial_position=CONFIG['initial_position'],
        cv_range=CONFIG['cv_range']
    )

    # ---- Create evaluator ----
    evaluator=UniformKLEvaluator2D.from_ranges(
        ranges=evaluation_ranges,
        n_bins=CONFIG['n_bins']
    )

    # ---- Debug: Show initial parameters ----
    print("\n" + "="*60)
    print("INITIAL PARAMETERS")
    print("="*60)
    print(f"Initial mean (normalized [0,1]): {initial_mean}")
    bounds=bias.get_parameter_bounds()
    initial_params_denorm=bounds[:, 0] + initial_mean * (bounds[:, 1] - bounds[:, 0])
    print(f"Initial parameters (denormalized):")
    for i in range(CONFIG['num_gaussians']):
        idx=i * 6
        h, cx, cy=initial_params_denorm[idx:idx+3]
        log_vx, rho, log_vy=initial_params_denorm[idx+3:idx+6]
        sx, sy=np.exp(log_vx/2), np.exp(log_vy/2)
        print(
            f"  Gaussian {i}: h={h:.1f} kJ/mol, center=({cx:.2f}, {cy:.2f}), σ=({sx:.3f}, {sy:.3f}), ρ={rho:.2f}")
    print("="*60 + "\n")

    # ---- Create workflow ----
    workflow=CMAESWorkflow(
        bias=bias,
        sampler=sampler,
        evaluator=evaluator,
        initial_mean=initial_mean,
        sigma=CONFIG['sigma'],
        population_size=CONFIG['population_size'],
        max_generations=CONFIG['max_generations'],
        n_workers=CONFIG['n_workers'],
        n_replicas=CONFIG['n_replicas'],
        early_stop_patience=CONFIG['early_stop_patience']
    )

    # ---- Run optimization ----
    output_dir=SCRIPT_DIR / "output_muller_brown"
    result=workflow.optimize(str(output_dir))

    # ---- Check if optimization was interrupted ----
    if result is None:
        print("\n⚠️  Optimization was interrupted before completing any generations")
        return None, None, None

    # ---- Print results ----
    print("\n" + "="*60)
    if result.get('interrupted', False):
        print("OPTIMIZATION INTERRUPTED")
    else:
        print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nBest score: {result['best_score']:.6f}")
    print(f"Best generation: {result['best_generation']}")
    print(f"\nResults saved to: {output_dir}")

    # ---- Save result dictionary for later plotting ----
    result_file=output_dir / "optimization_result.pkl"
    with open(result_file, 'wb') as f:
        pickle.dump({
            'result': result,
            'bias': bias,
            'evaluation_ranges': evaluation_ranges
        }, f)
    print(f"✓ Result saved to: {result_file}")

    # ---- Note checkpoint file location ----
    checkpoint_file=output_dir / "optimization_checkpoint.pkl"
    if checkpoint_file.exists():
        print(f"✓ Checkpoint file: {checkpoint_file}")

    return result, bias, evaluation_ranges, output_dir


def resume_optimization(checkpoint_file=None,
                       override_simulation_steps=None,
                       override_max_generations=None,
                       override_n_workers=None,
                       override_population_size=None):
    """
    Resume optimization from a checkpoint file with optional parameter overrides.
    Note: override_population_size is not recommended as it breaks CMA-ES internal state.
    """
    SCRIPT_DIR=Path(__file__).parent.resolve()

    if checkpoint_file is None:
        checkpoint_file=SCRIPT_DIR / "output_muller_brown" / "optimization_checkpoint.pkl"
    checkpoint_file=Path(checkpoint_file)
    if not checkpoint_file.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_file}")
        print("Run optimization first with: python mb_example.py run")
        return None, None, None, None

    # ---- Logging: Resume info ----
    print("\n" + "="*60)
    print("RESUMING MULLER-BROWN CMA-ES OPTIMIZATION")
    print("="*60)
    print(f"Checkpoint: {checkpoint_file}")

    # ---- Use FULL BOX for evaluation (matching config) ----
    evaluation_ranges=[(CONFIG['evaluation_x_range'][0], CONFIG['evaluation_x_range'][1]),
                         (CONFIG['evaluation_y_range'][0], CONFIG['evaluation_y_range'][1])]

    # ---- Recreate bias as in original optimization ----
    bias=MultiGaussian2DForceBias(
        n_gaussians=CONFIG['num_gaussians'],
        height_range=CONFIG['height_range'],
        center_x_range=CONFIG['center_x_range'],
        center_y_range=CONFIG['center_y_range'],
        log_variance_x_range=CONFIG['log_variance_x_range'],
        log_variance_y_range=CONFIG['log_variance_y_range'],
    )

    # ---- Load checkpoint to get original settings ----
    with open(checkpoint_file, 'rb') as f:
        checkpoint_data=pickle.load(f)

    original_pop_size=checkpoint_data.get('population_size', CONFIG['population_size'])
    original_n_workers=checkpoint_data.get('n_workers', CONFIG['n_workers'])
    original_sigma=checkpoint_data.get('sigma', CONFIG['sigma'])
    original_max_gen=checkpoint_data.get('max_generations', CONFIG['max_generations'])
    original_n_replicas=checkpoint_data.get('n_replicas', CONFIG['n_replicas'])
    original_early_stop=checkpoint_data.get('early_stop_patience', CONFIG['early_stop_patience'])
    original_sim_steps=checkpoint_data.get(
        'simulation_steps', CONFIG['simulation_steps'])

    # ---- Apply overrides if provided ----
    use_sim_steps=override_simulation_steps if override_simulation_steps is not None else original_sim_steps
    use_max_gen=override_max_generations if override_max_generations is not None else original_max_gen
    use_n_workers=override_n_workers if override_n_workers is not None else original_n_workers
    use_pop_size=override_population_size if override_population_size is not None else original_pop_size

    print(f"\nCheckpoint settings:")
    print(f"  Population size: {original_pop_size}")
    print(f"  Workers: {original_n_workers}")
    print(f"  Sigma: {original_sigma}")
    print(f"  Max generations: {original_max_gen}")
    print(f"  N replicas: {original_n_replicas}")
    print(f"  Early stop patience: {original_early_stop}")
    print(f"  Simulation steps: {original_sim_steps}")

    if any([override_simulation_steps, override_max_generations, override_n_workers, override_population_size]):
        print(f"\n⚠️  OVERRIDES APPLIED:")
        if override_simulation_steps:
            print(
                f"  Simulation steps: {original_sim_steps} → {use_sim_steps}")
        if override_max_generations:
            print(f"  Max generations: {original_max_gen} → {use_max_gen}")
        if override_n_workers:
            print(f"  Workers: {original_n_workers} → {use_n_workers}")
        if override_population_size:
            print(f"  Population size: {original_pop_size} → {use_pop_size}")
            print(f"  ⚠️  WARNING: Changing population size may disrupt CMA-ES state!")
    else:
        print(f"\nUsing original settings (no overrides)")

    # ---- Create sampler (with any overridden simulation steps) ----
    sampler=MullerBrownSampler(
        temperature=CONFIG['temperature'],
        time_step=CONFIG['time_step'],
        friction=CONFIG['friction'],
        simulation_steps=use_sim_steps,
        report_interval=CONFIG['report_interval'],
        initial_position=CONFIG['initial_position'],
        cv_range=CONFIG['cv_range']
    )

    # ---- Create evaluator ----
    evaluator=UniformKLEvaluator2D.from_ranges(
        ranges=evaluation_ranges,
        n_bins=CONFIG['n_bins']
    )

    # ---- Create workflow with original (or overridden) settings ----
    workflow=CMAESWorkflow(
        bias=bias,
        sampler=sampler,
        evaluator=evaluator,
        initial_mean=None,
        sigma=original_sigma,  # Never override sigma - it's part of CMA-ES state
        population_size=use_pop_size,
        max_generations=use_max_gen,
        n_workers=use_n_workers,
        n_replicas=original_n_replicas,  # Keep original n_replicas
        early_stop_patience=original_early_stop  # Keep original early_stop_patience
    )

    # ---- Resume from checkpoint ----
    output_dir=SCRIPT_DIR / "output_muller_brown"
    result=workflow.optimize(str(output_dir), resume_from=str(checkpoint_file))

    # ---- Check if optimization was interrupted again ----
    if result is None:
        print("\n⚠️  Optimization was interrupted before completing any new generations")
        return None, None, None, None

    # ---- Print results ----
    print("\n" + "="*60)
    if result.get('interrupted', False):
        print("OPTIMIZATION INTERRUPTED AGAIN")
    else:
        print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nBest score: {result['best_score']:.6f}")
    print(f"Best generation: {result['best_generation']}")
    print(f"\nResults saved to: {output_dir}")

    # ---- Save final result ----
    result_file=output_dir / "optimization_result.pkl"
    with open(result_file, 'wb') as f:
        pickle.dump({
            'result': result,
            'bias': bias,
            'evaluation_ranges': evaluation_ranges
        }, f)
    print(f"✓ Result saved to: {result_file}")

    return result, bias, evaluation_ranges, output_dir


def generate_plots(result_file=None):
    """
    Generate plots from saved optimization results.
    """
    SCRIPT_DIR=Path(__file__).parent.resolve()

    # ---- Locate result or checkpoint file ----
    if result_file is None:
        checkpoint_file=SCRIPT_DIR / "output_muller_brown" / "optimization_checkpoint.pkl"
        result_file_path=SCRIPT_DIR / "output_muller_brown" / "optimization_result.pkl"
        if result_file_path.exists():
            result_file=result_file_path
            print("Using final result file (optimization completed)")
        elif checkpoint_file.exists():
            result_file=checkpoint_file
            print("Using checkpoint file (optimization may have been interrupted)")
        else:
            print(f"Error: No result files found in {SCRIPT_DIR / 'output_muller_brown'}")
            print("Run optimization first with: python mb_example.py run")
            return
    result_file=Path(result_file)
    if not result_file.exists():
        print(f"Error: Result file not found: {result_file}")
        print("Run optimization first with: python mb_example.py run")
        return

    # ---- Load saved results ----
    print(f"\nLoading results from: {result_file}")
    with open(result_file, 'rb') as f:
        data=pickle.load(f)
    result=data['result']
    bias=data['bias']
    evaluation_ranges=data.get('evaluation_ranges', CONFIG['cv_range'])
    output_dir=result_file.parent

    # ---- Use the centralized plotting function ----
    create_all_plots(result, bias, evaluation_ranges, output_dir)


def analyze_evaluation_quality(result, bias, x_range, y_range, output_dir):
    """Compare analytical variance vs sampled KLD across generations.
    
    Creates visualization showing:
    1. Analytical metric: Variance of biased potential (landscape flatness)
    2. Sampled metric: KL divergence from MD-based evaluation
    
    These measure different but related things:
    - Analytical variance: How flat is the biased landscape?
    - Sampled KLD: How uniform is the finite-time sampling?
    
    Args:
        result: Optimization result dictionary with history
        bias: MultiGaussian2DForceBias object
        x_range: (x_min, x_max) for evaluation
        y_range: (y_min, y_max) for evaluation
        output_dir: Directory to save plots
    """
    print("\n" + "="*60)
    print("ANALYZING EVALUATION QUALITY")
    print("="*60)
    
    # Create analytical evaluator (computes variance of biased potential)
    analytical_evaluator = AnalyticalMullerBrownEvaluator(
        bias=bias,
        x_range=x_range,
        y_range=y_range,
        temperature=CONFIG['temperature'],
        n_bins=CONFIG['n_bins'],  
        energy_cutoff=200.0  # Standard cutoff for metadynamics accessible region (150-300 kJ/mol typical)
    )
    
    # Extract data from each generation
    generations = []
    analytical_variances = []  # Variance of biased potential
    sampled_klds = []  # KL divergence from finite-time sampling
    
    print("Computing analytical variance for each generation...")
    for i, gen in enumerate(result['history']):
        # Get best parameters from this generation
        best_normalized = gen['best_solution']
        bounds = bias.get_parameter_bounds()
        best_params = bounds[:, 0] + best_normalized * (bounds[:, 1] - bounds[:, 0])
        
        # Analytical evaluation (variance of biased potential) with debug for first, middle, and last
        debug = (i == 0 or i == len(result['history'])//2 or i == len(result['history']) - 1)
        if debug:
            print(f"\n  Gen {gen['generation']:3d} (detailed):")
        analytical_variance = analytical_evaluator.evaluate(best_params, debug=debug)
        
        # Sampled evaluation (KL divergence from MD)
        sampled_kld = gen['best_score']
        
        generations.append(gen['generation'])
        analytical_variances.append(analytical_variance)
        sampled_klds.append(sampled_kld)
        
        if (i + 1) % 10 == 0 or i == 0 or i == len(result['history']) - 1:
            if not debug:
                print(f"  Gen {gen['generation']:3d}: Variance={analytical_variance:.1f} (kJ/mol)², Sampled KLD={sampled_kld:.4f}")
    
    # Convert to numpy arrays
    generations = np.array(generations)
    analytical_variances = np.array(analytical_variances)
    sampled_klds = np.array(sampled_klds)
    
    # Create figure with 2 subplots (different metrics, no error comparison)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top: Analytical variance (landscape flatness)
    ax1.plot(generations, analytical_variances, 'b-', linewidth=2.5, marker='o', markersize=6, label='Potential variance')
    ax1.fill_between(generations, 0, analytical_variances, alpha=0.2, color='b')
    ax1.set_ylabel('Variance (kJ/mol)²', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Analytical: Variance of Biased Potential (Lower = Flatter Landscape)', fontsize=14, fontweight='bold')
    
    # Bottom: Sampled KLD
    ax2.plot(generations, sampled_klds, 'r-', linewidth=2.5, marker='s', markersize=6, label='KL divergence (MD)')
    ax2.fill_between(generations, 0, sampled_klds, alpha=0.2, color='r')
    ax2.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Sampled: KL Divergence from Finite-Time Sampling (Lower = More Uniform)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "evaluation_quality_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved evaluation quality analysis to: {output_path.name}")
    
    # Print statistics
    print("\n" + "="*60)
    print("EVALUATION QUALITY STATISTICS")
    print("="*60)
    print(f"Analytical (Variance of Biased Potential):")
    print(f"  Initial variance:  {analytical_variances[0]:.1f} (kJ/mol)²")
    print(f"  Final variance:    {analytical_variances[-1]:.1f} (kJ/mol)²")
    print(f"  Best variance:     {np.min(analytical_variances):.1f} (kJ/mol)² (gen {generations[np.argmin(analytical_variances)]})")
    print(f"  Improvement:       {(analytical_variances[0] - analytical_variances[-1])/analytical_variances[0]*100:.1f}%")
    print(f"\nSampled (KL Divergence from MD):")
    print(f"  Initial KLD:       {sampled_klds[0]:.4f}")
    print(f"  Final KLD:         {sampled_klds[-1]:.4f}")
    print(f"  Best KLD:          {np.min(sampled_klds):.4f} (gen {generations[np.argmin(sampled_klds)]})")
    print(f"  Improvement:       {(sampled_klds[0] - sampled_klds[-1])/sampled_klds[0]*100:.1f}%")
    print("="*60)
    
    return {
        'generations': generations,
        'analytical_variances': analytical_variances,
        'sampled_klds': sampled_klds
    }


# =================== PLOTTING FUNCTIONS ===================


def create_all_plots(result, bias, cv_range, output_dir):
    """
    Create all visualization plots from optimization results.
    Args:
        result: Optimization result dictionary
        bias: MultiGaussian2DForceBias object
        cv_range: CV range for plotting ((x_min, x_max), (y_min, y_max))
        output_dir: Directory to save plots (Path object)
    """
    from pycmaetad.visualization import (
        plot_convergence, plot_parameter_evolution, plot_sigma_evolution,
        plot_bias_landscape_2d, plot_bias_evolution, plot_convergence_diagnostics,
        plot_cv_histogram_evolution
    )
    from pycmaetad.visualization.colvar import plot_colvar_evolution, plot_cv_histogram, plot_cv_time_series

    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    # Core convergence plots from visualization module
    plot_convergence(result, output_path=output_dir / "convergence.png")
    plot_convergence_diagnostics(
        result, output_path=output_dir / "convergence_diagnostics.png")
    plot_parameter_evolution(result, bias, output_path=output_dir / "parameter_evolution.png", param_names=['height_1', 'center_x_1', 'center_y_1', 'log_var_x_1', 'rho_1', 'log_var_y_1', 'height_2', 'center_x_2', 'center_y_2', 'log_var_x_2', 'rho_2', 'log_var_y_2',
                                                                                                 'height_3', 'center_x_3', 'center_y_3', 'log_var_x_3', 'rho_3', 'log_var_y_3'])
    plot_sigma_evolution(result, output_path=output_dir /
                         "sigma_evolution.png")

    # # Bias landscape evolution
    # bias_landscapes_dir = output_dir / "bias_landscapes"
    # plot_bias_evolution(
    #     bias=bias,
    #     result=result,
    #     cv_range=cv_range,
    #     output_dir=bias_landscapes_dir,
    #     generations='all',
    #     show_best_only=False
    # )

    # CV histogram evolution
    cv_histograms_dir=output_dir / "cv_histograms"
    plot_cv_histogram_evolution(
        result=result,
        output_dir=cv_histograms_dir,
        generations='all',
        n_bins=30,
        show_uniform=True,
        cv_range=cv_range,
        individual='best',
        initial_position=CONFIG['initial_position']
    )

    # Colvar plots
    plot_colvar_evolution(result, output_path=output_dir /
                          "colvar_evolution.png", individual='all')
    plot_cv_histogram(result, output_path=output_dir /
                      "cv_histogram.png", cv_range=((-np.pi, np.pi), (-np.pi, np.pi)))
    plot_cv_time_series(result, output_path=output_dir /
                        "cv_time_series.png", individual='best')

    # Trajectory density plot
    plot_trajectory_density_2d(
        result, output_dir, cv_range, initial_position=CONFIG['initial_position'])

    # Best bias landscape (single plot using the visualization module)
    best_params=result.get('best_parameters')
    if best_params is not None:
        bias.set_parameters(best_params)
        # Convert cv_range format: ((x_min, x_max), (y_min, y_max)) -> [(x_min, x_max), (y_min, y_max)]
        cv_ranges=[cv_range[0], cv_range[1]]
        plot_bias_landscape_2d(
            bias=bias,
            all_params=[best_params],  # Pass as list
            cv_ranges=cv_ranges,
            output_path=output_dir / "best_bias_landscape.png",
            periodic=False,  # Muller-Brown CVs are not periodic
            mintozero=False
        )

        # Plot MB potential with bias overlay
        plot_mb_with_bias_overlay(
            bias=bias,
            best_params=best_params,
            output_path=output_dir / "mb_bias_overlay.png",
            cv_range=cv_range
        )
    
    # Bias landscape evolution (per generation)
    plot_bias_landscape_evolution(
        result=result,
        bias=bias,
        cv_range=cv_range,
        output_dir=output_dir,
        generations='auto'  # Every ~5th generation
    )
    
    # Evaluation quality analysis (analytical variance vs sampled KLD)
    analyze_evaluation_quality(
        result=result,
        bias=bias,
        x_range=cv_range[0],
        y_range=cv_range[1],
        output_dir=output_dir
    )

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
    x=np.linspace(-1.5, 1.5, 500)
    y=np.linspace(-0.5, 2.5, 500)
    X, Y=np.meshgrid(x, y)

    Z=muller_brown_potential(X, Y)
    min_energy=Z.min()

    # Find regions within energy cutoff
    accessible_mask=(Z - min_energy) < energy_cutoff

    # Get bounding box of accessible region
    accessible_y, accessible_x=np.where(accessible_mask)

    x_min=x[accessible_x.min()]
    x_max=x[accessible_x.max()]
    y_min=y[accessible_y.min()]
    y_max=y[accessible_y.max()]

    # Add small margin
    margin_x=0.1 * (x_max - x_min)
    margin_y=0.1 * (y_max - y_min)

    return (
        (x_min - margin_x, x_max + margin_x),
        (y_min - margin_y, y_max + margin_y)
    )


def plot_convergence_detailed(result, output_dir):
    """Plot CMA-ES convergence with more detail."""
    history=result['history']
    gens=[h['generation'] for h in history]
    best=np.array([h['best_score'] for h in history])
    mean=np.array([h['mean_score'] for h in history])
    std=np.array([h['std_score'] for h in history])

    # Determine cap value for display (handle outliers/penalties)
    valid_scores=best[np.isfinite(best) & (best < 1e4)]
    if len(valid_scores) > 3:
        cap_value=np.percentile(valid_scores, 95) * 1.5
    else:
        cap_value=np.max(best[np.isfinite(best)]) if np.any(
            np.isfinite(best)) else 10.0

    # Apply cap
    best_capped=np.clip(best, None, cap_value)
    mean_capped=np.clip(mean, None, cap_value)
    std_capped=np.clip(std, None, cap_value)

    lower_bound=np.clip(mean_capped - std_capped, 0, cap_value)
    upper_bound=np.clip(mean_capped + std_capped, 0, cap_value)

    fig, axes=plt.subplots(2, 1, figsize=(12, 10))

    # Top: Convergence
    axes[0].plot(gens, best_capped, 'b-o', label='Best',
                 linewidth=2, markersize=6)
    axes[0].plot(gens, mean_capped, 'r--s',
                 label='Mean', linewidth=2, markersize=5)
    axes[0].fill_between(gens, lower_bound, upper_bound,
                          alpha=0.2, color='red', label='±1 std')

    # Mark capped values
    n_capped=np.sum((best > cap_value) | (mean > cap_value))
    if n_capped > 0:
        axes[0].axhline(cap_value, color='gray', linestyle=':',
                        linewidth=1.5, alpha=0.7)
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
    all_scores=[]
    positions=[]
    for h in history[::max(1, len(history)//10)]:  # Sample every 10th gen
        all_scores.append(h['all_scores'])
        positions.append(h['generation'])

    parts=axes[1].violinplot(all_scores, positions=positions, widths=2,
                                showmeans=True, showmedians=True)
    axes[1].set_xlabel('Generation', fontsize=12)
    axes[1].set_ylabel('KL Divergence', fontsize=12)
    axes[1].set_title('Score Distribution (violin plot)',
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / "convergence_detailed.png", dpi=150)
    plt.close()
    print(f"📊 Saved: convergence_detailed.png")


def plot_best_bias_multi_improved(bias, best_params, output_dir, evaluation_ranges):
    """Visualize with MUCH better colormaps and scaling.

    Args:
        bias: MultiGaussian2DForceBias object
        best_params: Best parameters from optimization
        output_dir: Directory to save plots
        evaluation_ranges: ((x_min, x_max), (y_min, y_max)) evaluation region
    """
    bias.set_parameters(best_params)

    # Create grid
    x=np.linspace(-1.5, 1.5, 300)
    y=np.linspace(-0.5, 2.5, 300)
    X, Y=np.meshgrid(x, y)

    # Compute MB potential
    Z_mb=muller_brown_potential(X, Y)

    # Compute total bias
    Z_bias=np.zeros_like(X)

    gaussians_info=[]
    for i, gauss in enumerate(bias.get_all_gaussians()):
        params=gauss.get_parameters()
        height, cx, cy, log_var_x, rho, log_var_y=params

        var_x=np.exp(log_var_x)
        var_y=np.exp(log_var_y)
        cov_xy=rho * np.sqrt(var_x * var_y)

        # Compute inverse
        det=var_x * var_y - cov_xy**2
        i11=var_y / det
        i12=-cov_xy / det
        i22=var_x / det

        # Add this Gaussian
        dx=X - cx
        dy=Y - cy
        exponent=-0.5 * (dx*dx*i11 + 2*dx*dy*i12 + dy*dy*i22)
        gauss_contribution=height * np.exp(exponent)
        Z_bias += gauss_contribution

        # Compute rotation angle from covariance matrix
        cov_matrix=np.array([[var_x, cov_xy], [cov_xy, var_y]])
        eigenvalues, eigenvectors=np.linalg.eigh(cov_matrix)

        # Angle of first eigenvector (in degrees)
        angle_rad=np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        angle_deg=np.degrees(angle_rad)

        # Ellipse width/height are 2*sqrt(eigenvalue) for 2σ
        width_2sigma=2 * np.sqrt(eigenvalues[0]) * 2  # 2σ = 95% contour
        height_2sigma=2 * np.sqrt(eigenvalues[1]) * 2

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

    Z_total=Z_mb + Z_bias

    # Energy ranges for visualization
    E_min=Z_mb.min()
    E_vis_min=-150
    E_vis_max=-50

    # Log-scale transformation (same offset for both)
    E_offset=E_min - 10  # Offset to make all values positive
    Z_mb_log=np.log10(Z_mb - E_offset + 1)  # +1 to avoid log(0)
    Z_total_log=np.log10(Z_total - E_offset + 1)  # Same transformation

    # Create evaluation region mask
    eval_x=evaluation_ranges[0]
    eval_y=evaluation_ranges[1]
    accessible_mask=(
        (X >= eval_x[0]) & (X <= eval_x[1]) &
        (Y >= eval_y[0]) & (Y <= eval_y[1])
    )

    # Create figure
    fig=plt.figure(figsize=(20, 14))
    gs=fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Row 1: Main plots
    ax1=fig.add_subplot(gs[0, 0])
    ax2=fig.add_subplot(gs[0, 1])
    ax3=fig.add_subplot(gs[0, 2])

    # Row 2: Individual Gaussians
    ax4=fig.add_subplot(gs[1, 0])
    ax5=fig.add_subplot(gs[1, 1])
    ax6=fig.add_subplot(gs[1, 2])

    # Row 3: Analysis
    ax7=fig.add_subplot(gs[2, :2])
    ax8=fig.add_subplot(gs[2, 2])

    # Bias colormap (symmetric around 0)
    bias_max=np.percentile(np.abs(Z_bias), 95)
    bias_norm=TwoSlopeNorm(vmin=-bias_max, vcenter=0, vmax=bias_max)

    # ========== 1. MB Potential (LOG SCALE) ==========
    cs1=ax1.contourf(X, Y, Z_mb_log, levels=40, cmap='viridis')
    ax1.contour(X, Y, Z_mb, levels=20, colors='white',
                alpha=0.3, linewidths=0.5)

    # Overlay accessible region
    ax1.contour(X, Y, accessible_mask.astype(float), levels=[0.5],
               colors='red', linewidths=3, linestyles='--', alpha=0.8)

    for i, info in enumerate(gaussians_info):
        ax1.plot(info['center'][0], info['center'][1], 'r*',
                markersize=15, markeredgecolor='white', markeredgewidth=1.5,
                label=f'G{i+1}')

    # Add labeled contour lines at key energies
    key_levels=np.array([-150, -130, -110, -90, -70, -50])
    contour_lines=ax1.contour(X, Y, Z_mb, levels=key_levels,
                                colors='white', linewidths=1.2, alpha=0.8)
    ax1.clabel(contour_lines, inline=True, fontsize=8,
              fmt='%0.0f', inline_spacing=10)

    # Mark well positions with actual energy
    well_x=[-0.55, 0.62, -0.05]
    well_y=[1.45, 0.03, 0.47]
    well_names=['Main', 'Right', 'Center']

    for name, wx, wy in zip(well_names, well_x, well_y):
        ix=np.argmin(np.abs(x - wx))
        iy=np.argmin(np.abs(y - wy))
        energy=Z_mb[iy, ix]

        ax1.plot(wx, wy, 'w*', markersize=12,
                 markeredgecolor='black', markeredgewidth=1)
        ax1.text(wx, wy - 0.15, f'{name}\n{energy:.0f} kJ/mol',
                ha='center', va='top', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.85))

    ax1.set_title('Muller-Brown Potential\n(red dashed = evaluation region)',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (nm)')
    ax1.set_ylabel('Y (nm)')
    ax1.legend(loc='upper right', fontsize=8)
    cb1=plt.colorbar(cs1, ax=ax1)
    cb1.set_label('log₁₀(E - E_min + 1)', fontsize=9)

    # ========== 2. Total Bias ==========
    cs2=ax2.contourf(X, Y, Z_bias, levels=40, cmap='RdBu_r', norm=bias_norm,
                      extend='both')
    ax2.contour(X, Y, Z_bias, levels=20, colors='black',
                alpha=0.2, linewidths=0.5)

    # Overlay accessible region
    ax2.contour(X, Y, accessible_mask.astype(float), levels=[0.5],
               colors='green', linewidths=2, linestyles='--', alpha=0.8)

    from matplotlib.patches import Ellipse

    for i, info in enumerate(gaussians_info):
        ax2.plot(info['center'][0], info['center'][1], 'k*',
                markersize=15, markeredgecolor='white', markeredgewidth=1.5)

        # Draw ROTATED ellipse showing Gaussian extent (2σ)
        ellipse=Ellipse(
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
    cb2=plt.colorbar(cs2, ax=ax2)
    cb2.set_label(f'Bias (kJ/mol)\n[±{bias_max:.0f}]', fontsize=9)

    # ========== 3. Biased Potential (LOG SCALE - MATCHING PLOT 1) ==========
    cs3=ax3.contourf(X, Y, Z_total_log, levels=40,
                     cmap='viridis')  # ← Changed to log scale
    ax3.contour(X, Y, Z_total, levels=30, colors='white',
                alpha=0.3, linewidths=0.5)

    # Overlay accessible region
    ax3.contour(X, Y, accessible_mask.astype(float), levels=[0.5],
               colors='red', linewidths=2, linestyles='--', alpha=0.8)

    ax3.set_title('Biased Potential\n(ideally: flat in evaluation region)',
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (nm)')
    ax3.set_ylabel('Y (nm)')
    cb3=plt.colorbar(cs3, ax=ax3)
    cb3.set_label('log₁₀(E - E_min + 1)', fontsize=9)

    # ========== 4-6. Individual Gaussians ==========
    for i, (ax, info) in enumerate(zip([ax4, ax5, ax6], gaussians_info)):
        contrib_levels=np.linspace(0, info['height'], 30)

        cs=ax.contourf(X, Y, info['contribution'], levels=contrib_levels,
                        cmap='Reds', extend='max')
        ax.contour(X, Y, info['contribution'], levels=15, colors='black',
                   alpha=0.2, linewidths=0.5)
        ax.contour(X, Y, accessible_mask.astype(float), levels=[0.5],
                  colors='green', linewidths=1.5, linestyles='--', alpha=0.6)
        ax.plot(info['center'][0], info['center'][1], 'k*',
               markersize=12, markeredgecolor='white')

        # Add rotated ellipse
        ellipse=Ellipse(
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
    well_x=[-0.55, 0.62, -0.05]
    well_y=[1.45, 0.03, 0.47]
    colors=['b', 'r', 'g']

    for i, (wx, wy, color) in enumerate(zip(well_x, well_y, colors)):
        ix=np.argmin(np.abs(x - wx))
        iy=np.argmin(np.abs(y - wy))

        ax7.plot(x, Z_mb[:, ix], f'{color}--', alpha=0.5, linewidth=1.5,
                label=f'Well {i+1} MB')
        ax7.plot(x, Z_total[:, ix], f'{color}-', linewidth=2.5,
                label=f'Well {i+1} Biased')

    ax7.axvspan(eval_x[0], eval_x[1], alpha=0.1, color='green',
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
    Z_accessible=Z_total[accessible_mask]

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

    plt.savefig(output_dir / "best_bias_multi_detailed.png",
                dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n📊 Saved: best_bias_multi_detailed.png")

    # Print detailed stats
    print("\n" + "="*60)
    print("GAUSSIAN PLACEMENT ANALYSIS")
    print("="*60)
    for i, info in enumerate(gaussians_info):
        print(f"\nGaussian {i+1}:")
        print(f"  Height:      {info['height']:>7.2f} kJ/mol")
        print(
            f"  Center:      ({info['center'][0]:>6.3f}, {info['center'][1]:>6.3f}) nm")
        print(
            f"  Sigma:       ({info['sigma'][0]:>6.3f}, {info['sigma'][1]:>6.3f}) nm")
        print(f"  Correlation: {info['rho']:>6.3f}")
        print(f"  Rotation:    {info['angle']:>6.1f}°")
        print(
            f"  Coverage:    ~{4*info['sigma'][0]*4*info['sigma'][1]:.2f} nm² (4σ ellipse)")

    print(f"\nBiased potential statistics (ACCESSIBLE REGION ONLY):")
    print(
        f"  Range:  [{Z_accessible.min():.1f}, {Z_accessible.max():.1f}] kJ/mol")
    print(f"  Spread: {Z_accessible.max() - Z_accessible.min():.1f} kJ/mol")
    print(f"  Std:    {Z_accessible.std():.1f} kJ/mol")
    print(f"  → Goal: minimize spread & std for flat landscape")

    print(f"\nEvaluation region:")
    print(f"  X: [{eval_x[0]:.2f}, {eval_x[1]:.2f}] nm")
    print(f"  Y: [{eval_y[0]:.2f}, {eval_y[1]:.2f}] nm")
    print(f"  Area: {(eval_x[1]-eval_x[0])*(eval_y[1]-eval_y[0]):.2f} nm²")

    print(f"\nVisualization range: [{E_vis_min}, {E_vis_max}] kJ/mol")


def plot_sampling_coverage(result, output_dir):
    """Show how sampling coverage evolves."""
    history=result['history']
    last_gen=history[-1]

    fig, ax=plt.subplots(figsize=(10, 6))

    scores=last_gen['all_scores']
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
    history=result['history']

    print("\n" + "="*60)
    print("OPTIMIZATION DIAGNOSTICS")
    print("="*60)

    # Convergence rate
    initial_best=history[0]['best_score']
    final_best=history[-1]['best_score']
    improvement=(initial_best - final_best) / initial_best * 100

    print(f"\n📈 Convergence:")
    print(f"  Initial best:  {initial_best:.4f}")
    print(f"  Final best:    {final_best:.4f}")
    print(f"  Improvement:   {improvement:.1f}%")

    if improvement < 10:
        print(f"  ⚠️  WARNING: Less than 10% improvement!")
        print(f"      → Try more generations or better initialization")

    # Stagnation check
    last_5=[h['best_score'] for h in history[-5:]]
    if len(last_5) >= 5:
        stagnation=np.std(last_5) / np.mean(last_5) * 100
        print(f"\n📊 Stagnation (last 5 gens):")
        print(f"  Relative std: {stagnation:.2f}%")
        if stagnation < 1:
            print(f"  ⚠️  Likely converged (or stuck!)")

    # Noise estimate
    mean_stds=[h['std_score'] / h['mean_score'] for h in history]
    avg_noise=np.mean(mean_stds) * 100
    print(f"\n🔊 Noise level:")
    print(f"  Avg CV:  {avg_noise:.1f}%")
    if avg_noise > 30:
        print(f"  ⚠️  High noise! Consider:")
        print(f"      → Longer simulations (more steps)")
        print(f"      → More bins in evaluator")
        print(f"      → Averaging multiple runs per individual")

    # Efficiency
    total_time=sum(h['generation_time'] for h in history)
    total_evals=len(history) * len(history[0]['all_scores'])
    avg_time_per_eval=total_time / total_evals

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
    history=result['history']
    n_gaussians=bias.n_gaussians

    # Identify which generations to plot
    gens_to_plot=list(range(0, len(history), save_frequency))
    if (len(history) - 1) not in gens_to_plot:
        gens_to_plot.append(len(history) - 1)  # Always include final

    print(
        f"\n📊 Creating CMA-ES exploration plots for {len(gens_to_plot)} generations...")

    for gen_idx in gens_to_plot:
        gen_data=history[gen_idx]
        gen=gen_data['generation']
        # Convert list to numpy array
        population=np.array(gen_data['population'])
        cma_mean=gen_data['cma_mean']
        scores=gen_data['all_scores']
        best_idx=np.argmin(scores)

        # Denormalize population for visualization
        population_denorm=np.array(
            [bias.denormalize_parameters(p) for p in population])
        mean_denorm=bias.denormalize_parameters(cma_mean)

        # Create visualization - simpler layout
        fig=plt.figure(figsize=(16, 6))
        gs=fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

        # Plot 1: Gaussian centers in 2D space with CMA-ES search ellipse
        ax1=fig.add_subplot(gs[0, 0])

        # Draw Muller-Brown wells for reference
        well_positions=[(-0.55, 1.45), (0.62, 0.03), (-0.05, 0.47)]
        for wx, wy in well_positions:
            ax1.scatter(wx, wy, c='red', marker='*', s=200, edgecolor='darkred',
                       linewidth=1.5, zorder=10, alpha=0.7)

        # Plot each Gaussian's mean and best positions (not full population)
        colors=plt.cm.Set1(np.linspace(0, 1, n_gaussians))
        for g in range(n_gaussians):
            cx_idx=g * 6 + 1  # Parameter index for cx
            cy_idx=g * 6 + 2  # Parameter index for cy

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
                cov_matrix=gen_data['cma_cov']

                # Check if covariance matrix is valid
                if isinstance(cov_matrix, np.ndarray) and cov_matrix.ndim == 2:
                    try:
                        # Extract 2x2 covariance for (cx, cy) of this Gaussian
                        cov_cx_cy=cov_matrix[np.ix_(
                            [cx_idx, cy_idx], [cx_idx, cy_idx])]

                        # Denormalize covariance (scale by parameter ranges)
                        bounds=bias.get_parameter_bounds()
                        scale_x=bounds[cx_idx, 1] - bounds[cx_idx, 0]
                        scale_y=bounds[cy_idx, 1] - bounds[cy_idx, 0]
                        cov_cx_cy_denorm=cov_cx_cy * \
                            np.outer([scale_x, scale_y], [scale_x, scale_y])

                        # Compute eigenvalues and eigenvectors for ellipse
                        eigenvalues, eigenvectors=np.linalg.eigh(
                            cov_cx_cy_denorm)
                        angle=np.degrees(np.arctan2(
                            eigenvectors[1, 0], eigenvectors[0, 0]))
                        width=2 * np.sqrt(eigenvalues[0]) * 2  # 2-sigma
                        height=2 * np.sqrt(eigenvalues[1]) * 2

                        ellipse=Ellipse(
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
        ax2=fig.add_subplot(gs[0, 1])

        # Show distribution of heights and correlation coefficients
        param_data=[]
        param_labels=[]
        param_colors=[]

        for g in range(n_gaussians):
            h_idx=g * 6
            h_vals=population_denorm[:, h_idx]
            param_data.append(h_vals)
            param_labels.append(f'G{g+1}\nHeight')
            param_colors.append(colors[g])

        bp=ax2.boxplot(param_data, tick_labels=param_labels, patch_artist=True,
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
            h_idx=g * 6
            ax2.scatter(g + 1, population_denorm[best_idx, h_idx],
                       c='lime', marker='D', s=120, edgecolor='darkgreen',
                       linewidth=2, zorder=10)

        ax2.set_ylabel('Height (kJ/mol)', fontsize=12)
        ax2.set_title(f'Generation {gen}: Gaussian Heights\n(red=median, blue=mean, green=best)',
                     fontsize=12, weight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Score distribution and improvement
        ax3=fig.add_subplot(gs[0, 2])

        # Show score range and statistics
        ax3.hist(scores, bins=12, alpha=0.7, color='skyblue',
                 edgecolor='black', linewidth=1.5)
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
        output_path=Path(output_dir) / f'cmaes_gen{gen:03d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✅ Saved: {output_path.name}")

    print(f"✅ CMA-ES exploration plots complete!\n")


def plot_trajectory_density_2d(result, output_dir, evaluation_ranges, n_bins=50, initial_position=None):
    """Plot 2D density heatmap of sampled trajectory positions from COLVAR.

    Shows the actual sampling distribution overlaid on the Muller-Brown potential,
    allowing visual comparison with the target uniform distribution.

    Args:
        result: Optimization result dictionary with 'best_generation' and 'history'
        output_dir: Directory to save plots
        evaluation_ranges: ((x_min, x_max), (y_min, y_max)) evaluation region
        n_bins: Number of bins for the 2D histogram (default: 50)
        initial_position: (x, y) tuple of starting position to mark on plot
    """
    # Find the best COLVAR file from the best generation
    best_gen=result['best_generation']
    best_gen_data=result['history'][best_gen]

    # Find the individual with the best score in that generation
    best_idx=np.argmin(best_gen_data['all_scores'])

    # Construct path to COLVAR file
    gen_dir=Path(output_dir) / f"gen{best_gen:03d}"
    ind_dir=gen_dir / f"ind{best_idx:03d}"

    # Try to find COLVAR file (check both direct and replica subdirectories)
    colvar_file=ind_dir / "COLVAR"
    if not colvar_file.exists():
        replica_colvars=list(ind_dir.glob("replica_*/COLVAR"))
        if replica_colvars:
            # Use first replica if multiple exist
            colvar_file=replica_colvars[0]
        else:
            print(f"⚠️  COLVAR file not found in {ind_dir}")
            print(f"    (Expected from gen {best_gen}, individual {best_idx})")
            return

    # Read COLVAR file to get CV positions
    try:
        data=np.loadtxt(colvar_file, skiprows=1)
        if data.ndim == 1:
            data=data.reshape(1, -1)

        # Extract CV values (columns 1 and 2 are x, y)
        if data.shape[1] < 3:
            print(f"⚠️  COLVAR file has insufficient columns: {data.shape[1]}")
            return

        positions=data[:, 1:3]  # CV_x and CV_y

    except Exception as e:
        print(f"⚠️  Error reading COLVAR file {colvar_file}: {e}")
        return

    if len(positions) == 0:
        print(f"⚠️  No positions found in COLVAR file: {colvar_file}")
        return

    print(
        f"📊 Loaded {len(positions)} trajectory frames from {colvar_file.name}")

    # Create 2D histogram
    x_range=evaluation_ranges[0]
    y_range=evaluation_ranges[1]

    hist, x_edges, y_edges=np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=n_bins,
        range=[x_range, y_range],
        density=False  # Get counts
    )

    # Normalize to probabilities
    hist_prob=hist / np.sum(hist)

    # Create grid for Muller-Brown potential
    x=np.linspace(x_range[0], x_range[1], 300)
    y=np.linspace(y_range[0], y_range[1], 300)
    X, Y=np.meshgrid(x, y)
    Z=muller_brown_potential(X, Y)

    # Create figure with two subplots
    fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Muller-Brown potential with trajectory overlay
    cs1=ax1.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
    ax1.contour(X, Y, Z, levels=15, colors='white', linewidths=0.5, alpha=0.5)

    # Overlay trajectory density as scatter plot with alpha
    ax1.scatter(positions[::10, 0], positions[::10, 1],  # Subsample for visibility
                c='red', s=2, alpha=0.15, rasterized=True)

    # Mark starting position if provided
    if initial_position is not None:
        ax1.scatter(*initial_position, c='lime', s=200, marker='*',
                   edgecolors='black', linewidths=2, zorder=10,
                   label='Start')
        ax1.legend(loc='upper right', fontsize=10)

    ax1.set_xlabel('X (nm)', fontsize=12)
    ax1.set_ylabel('Y (nm)', fontsize=12)
    ax1.set_title(f'Muller-Brown Potential + Trajectory\n({len(positions)} frames, every 10th shown (gen {best_gen}, ind {best_idx}))',
                  fontsize=14, fontweight='bold')
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    plt.colorbar(cs1, ax=ax1, label='Energy (kJ/mol)')

    # Plot 2: 2D density heatmap
    X_bins=(x_edges[:-1] + x_edges[1:]) / 2
    Y_bins=(y_edges[:-1] + y_edges[1:]) / 2
    X_bins, Y_bins=np.meshgrid(X_bins, Y_bins)

    im=ax2.contourf(X_bins, Y_bins, hist_prob.T, levels=30, cmap='hot')
    ax2.contour(X_bins, Y_bins, hist_prob.T, levels=10, colors='white',
               linewidths=0.5, alpha=0.5)

    # Add colorbar
    cb=plt.colorbar(im, ax=ax2)
    cb.set_label('Probability Density', fontsize=11)

    # Mark starting cell if initial position is provided
    if initial_position is not None:
        # Find which bin the starting position belongs to
        start_x, start_y=initial_position
        bin_x_idx=int((start_x - x_range[0]) /
                      (x_range[1] - x_range[0]) * n_bins)
        bin_y_idx=int((start_y - y_range[0]) /
                      (y_range[1] - y_range[0]) * n_bins)

        # Ensure indices are within bounds
        bin_x_idx=np.clip(bin_x_idx, 0, n_bins - 1)
        bin_y_idx=np.clip(bin_y_idx, 0, n_bins - 1)

        # Calculate bin edges for the starting cell
        bin_width_x=(x_range[1] - x_range[0]) / n_bins
        bin_width_y=(y_range[1] - y_range[0]) / n_bins

        start_bin_x=x_range[0] + bin_x_idx * bin_width_x
        start_bin_y=y_range[0] + bin_y_idx * bin_width_y

        # Draw rectangle around starting cell
        from matplotlib.patches import Rectangle
        rect=Rectangle((start_bin_x, start_bin_y), bin_width_x, bin_width_y,
                        linewidth=3, edgecolor='lime', facecolor='none',
                        zorder=10, label='Starting cell')
        ax2.add_patch(rect)

        # Mark center with a star
        ax2.scatter(*initial_position, c='lime', s=200, marker='*',
                   edgecolors='black', linewidths=2, zorder=11)
        ax2.legend(loc='upper right', fontsize=10)

    ax2.set_xlabel('X (nm)', fontsize=12)
    ax2.set_ylabel('Y (nm)', fontsize=12)
    ax2.set_title(f'Sampling Density (2D Histogram)\nBins: {n_bins}×{n_bins} \n(gen {best_gen}, ind {best_idx})',
                  fontsize=14, fontweight='bold')
    ax2.set_xlim(x_range)
    ax2.set_ylim(y_range)

    # Add statistics
    uniformity=1.0 / (n_bins * n_bins)  # Target uniform probability
    max_prob=np.max(hist_prob)
    min_prob=np.min(hist_prob[hist_prob > 0]) if np.any(hist_prob > 0) else 0

    stats_text=(f"Target uniform: {uniformity:.6f}\n"
                  f"Max observed: {max_prob:.6f}\n"
                  f"Min observed: {min_prob:.6f}\n"
                  f"Ratio: {max_prob/uniformity:.2f}×")

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save plot
    output_path=Path(output_dir) / 'trajectory_density_2d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved trajectory density plot: {output_path.name}\n")


def muller_brown_potential(X, Y):
    """Vectorized Muller-Brown potential."""
    A=np.array([-200, -100, -170, 15])
    a=np.array([-1, -1, -6.5, 0.7])
    b=np.array([0, 0, 11, 0.6])
    c=np.array([-10, -10, -6.5, 0.7])
    x0=np.array([1, 0, -0.5, -1])
    y0=np.array([0, 0.5, 1.5, 1])

    V=np.zeros_like(X)
    for i in range(4):
        V += A[i] * np.exp(
            a[i] * (X - x0[i])**2 +
            b[i] * (X - x0[i]) * (Y - y0[i]) +
            c[i] * (Y - y0[i])**2
        )
    return V


def plot_mb_with_bias_overlay(bias, best_params, output_path, cv_range, n_points=200):
    """Plot Muller-Brown potential with bias overlay showing the flattening effect.

    Creates a multi-panel figure showing:
    1. Original MB potential
    2. Optimized bias potential
    3. Combined landscape (MB + bias)
    4. Overlay view with contours

    Args:
        bias: MultiGaussian2DForceBias object
        best_params: Best parameters from optimization
        output_path: Path to save the plot
        cv_range: ((x_min, x_max), (y_min, y_max))
        n_points: Grid resolution
    """
    # Set bias parameters
    bias.set_parameters(best_params)

    # Create grid
    x=np.linspace(cv_range[0][0], cv_range[0][1], n_points)
    y=np.linspace(cv_range[1][0], cv_range[1][1], n_points)
    X, Y=np.meshgrid(x, y)

    # Compute potentials
    V_mb=muller_brown_potential(X, Y)

    # Use bias's built-in method to compute bias potential (ensures consistency)
    V_bias=bias.evaluate_numpy(X, Y)

    V_combined=V_mb + V_bias

    # Create figure with 2x2 subplots
    fig=plt.figure(figsize=(16, 14))
    gs=fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Common settings
    levels_mb=np.linspace(V_mb.min(), V_mb.min() + 100, 20)
    levels_bias=15
    levels_combined=np.linspace(V_combined.min(), V_combined.min() + 100, 20)

    # 1. Original MB potential
    ax1=fig.add_subplot(gs[0, 0])
    im1=ax1.contourf(X, Y, V_mb, levels=levels_mb, cmap='viridis', alpha=0.8)
    ax1.contour(X, Y, V_mb, levels=levels_mb,
                colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(im1, ax=ax1, label='Energy (kJ/mol)')
    ax1.set_xlabel('x (nm)', fontsize=11)
    ax1.set_ylabel('y (nm)', fontsize=11)
    ax1.set_title('(a) Muller-Brown Potential', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')

    # 2. Bias potential
    ax2=fig.add_subplot(gs[0, 1])
    im2=ax2.contourf(X, Y, V_bias, levels=levels_bias,
                     cmap='RdYlBu_r', alpha=0.8)
    ax2.contour(X, Y, V_bias, levels=levels_bias,
                colors='black', alpha=0.3, linewidths=0.5)

    # Plot Gaussian centers
    for i in range(bias.n_gaussians):
        g=bias._gaussians[i]
        ax2.scatter(g['cx'], g['cy'], c='red', s=100, marker='x',
                   linewidths=2, zorder=10)

        # Draw ellipse for each Gaussian
        # Compute eigenvalues/eigenvectors from inverse covariance
        inv_cov=np.array([[g['inv_11'], g['inv_12']],
                           [g['inv_12'], g['inv_22']]])
        # Get covariance from inverse
        cov=np.linalg.inv(inv_cov)
        eigenvalues, eigenvectors=np.linalg.eigh(cov)

        # Ellipse parameters (1-sigma contour)
        width=2 * np.sqrt(eigenvalues[0])
        height=2 * np.sqrt(eigenvalues[1])
        angle=np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        from matplotlib.patches import Ellipse
        ellipse=Ellipse((g['cx'], g['cy']), width, height, angle=angle,
                         facecolor='none', edgecolor='red', linewidth=1.5,
                         alpha=0.6, linestyle='--')
        ax2.add_patch(ellipse)

    plt.colorbar(im2, ax=ax2, label='Bias Potential (kJ/mol)')
    ax2.set_xlabel('x (nm)', fontsize=11)
    ax2.set_ylabel('y (nm)', fontsize=11)
    ax2.set_title(f'(b) Optimized Bias ({bias.n_gaussians} Gaussians)',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')

    # 3. Combined landscape
    ax3=fig.add_subplot(gs[1, 0])
    im3=ax3.contourf(X, Y, V_combined, levels=levels_combined,
                     cmap='viridis', alpha=0.8)
    ax3.contour(X, Y, V_combined, levels=levels_combined,
                colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(im3, ax=ax3, label='Energy (kJ/mol)')
    ax3.set_xlabel('x (nm)', fontsize=11)
    ax3.set_ylabel('y (nm)', fontsize=11)
    ax3.set_title('(c) Combined Landscape (MB + Bias)',
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle=':')

    # 4. Overlay with contours of both
    ax4=fig.add_subplot(gs[1, 1])
    # Show MB as filled contours
    im4=ax4.contourf(X, Y, V_mb, levels=levels_mb, cmap='viridis', alpha=0.5)
    # Overlay bias as line contours in red
    bias_contours=ax4.contour(X, Y, V_bias, levels=10, colors='red',
                                linewidths=2, alpha=0.8)
    ax4.clabel(bias_contours, inline=True, fontsize=8, fmt='%d')

    # Combined contours in white
    combined_contours=ax4.contour(X, Y, V_combined, levels=10, colors='white',
                                    linewidths=1.5, alpha=0.7, linestyles='--')

    plt.colorbar(im4, ax=ax4, label='MB Energy (kJ/mol)')
    ax4.set_xlabel('x (nm)', fontsize=11)
    ax4.set_ylabel('y (nm)', fontsize=11)
    ax4.set_title('(d) Overlay View\n(MB=filled, Bias=red, Combined=white dashed)',
                 fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle=':')

    # Add legend for overlay
    from matplotlib.lines import Line2D
    legend_elements=[
        Line2D([0], [0], color='red', linewidth=2, label='Bias contours'),
        Line2D([0], [0], color='white', linewidth=1.5,
               linestyle='--', label='Combined contours')
    ]
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.suptitle('Muller-Brown Landscape Flattening via Optimized Bias',
                fontsize=14, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ MB + Bias overlay plot saved: {output_path}")


def plot_mb_bias_landscape_simple(bias, params, output_path, cv_range, generation, n_points=200):
    """Simplified bias landscape plot for per-generation visualization.
    
    Creates a single-panel plot showing MB potential with bias overlay.
    
    Args:
        bias: MultiGaussian2DForceBias object
        params: Bias parameters (denormalized)
        output_path: Path to save the plot
        cv_range: ((x_min, x_max), (y_min, y_max))
        generation: Generation number (for title)
        n_points: Grid resolution
    """
    # Set bias parameters
    bias.set_parameters(params)
    
    # Create grid
    x = np.linspace(cv_range[0][0], cv_range[0][1], n_points)
    y = np.linspace(cv_range[1][0], cv_range[1][1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Compute potentials
    V_mb = muller_brown_potential(X, Y)
    V_bias = bias.evaluate_numpy(X, Y)
    V_combined = V_mb + V_bias
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Plot MB as filled contours
    levels_mb = np.linspace(V_mb.min(), V_mb.min() + 100, 20)
    im = ax.contourf(X, Y, V_mb, levels=levels_mb, cmap='viridis', alpha=0.6)
    
    # Overlay bias as red contours
    bias_contours = ax.contour(X, Y, V_bias, levels=10, colors='red', 
                                linewidths=2, alpha=0.8)
    ax.clabel(bias_contours, inline=True, fontsize=8, fmt='%d')
    
    # Combined as white dashed contours
    combined_contours = ax.contour(X, Y, V_combined, levels=10, colors='white',
                                    linewidths=1.5, alpha=0.7, linestyles='--')
    
    # Plot Gaussian centers
    for i in range(bias.n_gaussians):
        g = bias._gaussians[i]
        ax.scatter(g['cx'], g['cy'], c='yellow', s=150, marker='x',
                  linewidths=3, zorder=10)
    
    plt.colorbar(im, ax=ax, label='MB Energy (kJ/mol)')
    ax.set_xlabel('x (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y (nm)', fontsize=12, fontweight='bold')
    ax.set_title(f'Generation {generation}: Muller-Brown + Bias Landscape',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Bias potential'),
        Line2D([0], [0], color='white', linewidth=1.5, linestyle='--', label='Combined (MB+Bias)'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='yellow', 
               markeredgecolor='black', markersize=10, linewidth=0, label='Gaussian centers')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_bias_landscape_evolution(result, bias, cv_range, output_dir, generations='auto'):
    """Generate bias landscape plots for multiple generations.
    
    Args:
        result: Optimization result dictionary with history
        bias: MultiGaussian2DForceBias object
        cv_range: ((x_min, x_max), (y_min, y_max))
        output_dir: Directory to save plots (will create bias_landscapes subfolder)
        generations: 'all', 'auto' (every 5th), or list of generation numbers
    """
    from pathlib import Path
    
    output_dir = Path(output_dir) / "bias_landscapes"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING BIAS LANDSCAPE EVOLUTION")
    print("="*60)
    
    history = result['history']
    total_gens = len(history)
    
    # Determine which generations to plot
    if generations == 'all':
        gen_indices = list(range(total_gens))
    elif generations == 'auto':
        # Every 5th generation + first and last
        step = max(1, total_gens // 20)  # Aim for ~20 plots
        gen_indices = list(range(0, total_gens, step))
        if (total_gens - 1) not in gen_indices:
            gen_indices.append(total_gens - 1)
    elif isinstance(generations, (list, tuple)):
        gen_indices = [g for g in generations if g < total_gens]
    else:
        gen_indices = [0, total_gens - 1]  # Just first and last
    
    print(f"  Generating {len(gen_indices)} plots from {total_gens} generations")
    print(f"  Generations: {gen_indices[:5]}{'...' if len(gen_indices) > 5 else ''}")
    
    # Generate plots
    bounds = bias.get_parameter_bounds()
    
    for idx, gen_idx in enumerate(gen_indices):
        gen = history[gen_idx]
        gen_num = gen['generation']
        
        # Get best parameters from this generation
        best_normalized = gen['best_solution']
        best_params = bounds[:, 0] + best_normalized * (bounds[:, 1] - bounds[:, 0])
        
        # Generate plot
        output_path = output_dir / f"landscape_gen{gen_num:03d}.png"
        plot_mb_bias_landscape_simple(
            bias=bias,
            params=best_params,
            output_path=output_path,
            cv_range=cv_range,
            generation=gen_num,
            n_points=200
        )
        
        if (idx + 1) % 10 == 0 or idx == len(gen_indices) - 1:
            print(f"  ✓ Generated {idx + 1}/{len(gen_indices)} plots...")
    
    print(f"\n✅ Bias landscape evolution plots saved to:")
    print(f"   {output_dir}")
    print("="*60)


# =================== ENTRY POINT ===================

if __name__ == "__main__":
    # ---- Parse command-line arguments ----
    parser = argparse.ArgumentParser(description='Muller-Brown CMA-ES Optimization')
    parser.add_argument('--config', type=str,
                       default='configs/config_tight.py',
                       help='Path to config file (default: configs/config_tight.py)')
    parser.add_argument('command', nargs='?', default='both',
                       choices=['run', 'resume', 'plot', 'both'],
                       help='Command to execute (default: both)')
    parser.add_argument('extra', nargs='*', help='Additional arguments for specific commands')

    args, unknown = parser.parse_known_args()
    args.extra = args.extra + unknown

    # ---- Load configuration ----
    SCRIPT_DIR = Path(__file__).parent.resolve()
    config_path = SCRIPT_DIR / args.config

    print(f"\n{'='*60}")
    print(f"Loading configuration: {config_path}")
    print(f"{'='*60}")
    
    try:
        CONFIG = load_config(config_path)
        print(f"✓ Loaded config: {CONFIG['name']}")
        print(f"  Description: {CONFIG['description']}")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Handle commands
    command = args.command.lower() if args.command else 'both'
    
    if command == "run":
        # Run optimization only
        run_optimization()
    elif command == "resume":
        # Resume optimization from checkpoint with optional overrides
        # Usage: python mb_example.py --config CONFIG resume [checkpoint_file] [--sim-steps N] [--max-gen N] [--workers N]
        checkpoint_file = None
        override_sim_steps = None
        override_max_gen = None
        override_workers = None

        print(f"\n🔍 DEBUG: Parsing resume arguments from args.extra: {args.extra}")
        
        i = 0
        while i < len(args.extra):
            arg = args.extra[i]
            if arg.endswith('.pkl'):
                checkpoint_file = arg
                print(f"  Found checkpoint file: {checkpoint_file}")
            elif arg == '--sim-steps':
                # Look for value before or after the flag
                if i + 1 < len(args.extra) and not args.extra[i + 1].startswith('--'):
                    override_sim_steps = int(args.extra[i + 1])
                    print(f"  Found --sim-steps: {override_sim_steps}")
                    i += 1
                elif i > 0 and not args.extra[i - 1].startswith('--'):
                    override_sim_steps = int(args.extra[i - 1])
                    print(f"  Found --sim-steps: {override_sim_steps}")
            elif arg in ['--max-gen', '--max-generations']:
                if i + 1 < len(args.extra) and not args.extra[i + 1].startswith('--'):
                    override_max_gen = int(args.extra[i + 1])
                    print(f"  Found --max-gen: {override_max_gen}")
                    i += 1
                elif i > 0 and not args.extra[i - 1].startswith('--'):
                    override_max_gen = int(args.extra[i - 1])
                    print(f"  Found --max-gen: {override_max_gen}")
            elif arg == '--workers':
                if i + 1 < len(args.extra) and not args.extra[i + 1].startswith('--'):
                    override_workers = int(args.extra[i + 1])
                    print(f"  Found --workers: {override_workers}")
                    i += 1
                elif i > 0 and not args.extra[i - 1].startswith('--'):
                    override_workers = int(args.extra[i - 1])
                    print(f"  Found --workers: {override_workers}")
            i += 1

        print(f"\n🔍 DEBUG: Calling resume_optimization with:")
        print(f"  override_simulation_steps={override_sim_steps}")
        print(f"  override_max_generations={override_max_gen}")
        print(f"  override_n_workers={override_workers}")
        
        resume_optimization(checkpoint_file,
                          override_simulation_steps=override_sim_steps,
                          override_max_generations=override_max_gen,
                          override_n_workers=override_workers)
    elif command == "plot":
        result_file = args.extra[0] if args.extra else None
        generate_plots(result_file)
    elif command == "both":
        result, bias, evaluation_ranges, output_dir = run_optimization()
        if result is not None:
            create_all_plots(result, bias, evaluation_ranges, output_dir)
    else:
        print("Unknown command. Usage:")
        print("  python mb_example.py [--config CONFIG] run                     # Run optimization from start")
        print("  python mb_example.py [--config CONFIG] resume [file] [options] # Resume from checkpoint")
        print("    Options:")
        print("      --sim-steps N         # Override simulation steps")
        print("      --max-gen N           # Override max generations")
        print("      --workers N           # Override number of workers")
        print("  python mb_example.py plot [file]                               # Generate plots from saved results")
        print("  python mb_example.py [--config CONFIG] both                    # Run optimization and generate plots (default)")
        print("\nAvailable configs:")
        print("  configs/config_wide.py  - Wide search space")
        print("  configs/config_tight.py - Tight search space (default)")
