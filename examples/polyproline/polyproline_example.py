import numpy as np
import pickle
import sys
from pathlib import Path
from pycmaetad.bias import PlumedHillBias
from pycmaetad.sampler import OpenMMPlumedSampler
from pycmaetad.optimizer import CMAESWorkflow
from pycmaetad.evaluator import HybridUniformEvaluator, UniformKLEvaluator


# ============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# ============================================================================

# Sampler parameters
TEMPERATURE = 300.0
TIME_STEP = 0.004 # 4 fs 
FRICTION = 1.0
SIMULATION_STEPS = 250000  # 1 microsecond at 4 fs timestep
REPORT_INTERVAL = 1000

# Bias parameters
HILLS_PER_D = 2
HILLS_SPACE = (-np.pi, np.pi)
HILLS_HEIGHT = 70.0
HILLS_WIDTH = 1.5

# Evaluator parameters
BIN_EDGES = np.linspace(-np.pi, np.pi, 31)
IS_2D = False

# Optimizer parameters
# Initialize centers evenly spaced, heights/widths at midpoint (0.5 in normalized space)
# This ensures hills start in different CV regions for better exploration
hills_per_d = HILLS_PER_D
INITIAL_MEAN = np.ones(hills_per_d * 3) * 0.5  # Heights and widths at midpoint
INITIAL_MEAN[:hills_per_d] = np.arange(0, 1, 1 / hills_per_d)  # Centers evenly spaced: [0.0, 0.5]

SIGMA = 0.1
POPULATION_SIZE = 20
MAX_GENERATIONS = 2
N_WORKERS = 20
N_REPLICAS = 2  # one replica per initial structure (2 pdb files)
EARLY_STOP_PATIENCE = 0

# CV range for plotting
CV_RANGE = (-np.pi, np.pi)


def run_optimization():
    """Run the CMA-ES optimization and save results."""
    # Get the directory containing this script
    SCRIPT_DIR = Path(__file__).parent.resolve()

    print("\n" + "="*60)
    print("POLYPROLINE CMA-ES OPTIMIZATION")
    print("Optimizing bias parameters for uniform sampling")
    print("="*60)
    print(f"Working directory: {SCRIPT_DIR}")

    # Create sampler (using global parameters)
    sampler = OpenMMPlumedSampler(
        pdb_file=[str(SCRIPT_DIR / "pp1.pdb"), str(SCRIPT_DIR / "pp2.pdb")],
        forcefield_files=["amber14-all.xml", "amber14/tip3pfb.xml"],
        temperature=TEMPERATURE,
        time_step=TIME_STEP,
        friction=FRICTION,
        simulation_steps=SIMULATION_STEPS,
        report_interval=REPORT_INTERVAL
    )

    # Create bias (using global parameters)
    bias = PlumedHillBias(
        plumed_template=str(SCRIPT_DIR / "plumed_template.dat"),
        hills_per_d=HILLS_PER_D,
        hills_space=HILLS_SPACE,
        hills_height=HILLS_HEIGHT,
        hills_width=HILLS_WIDTH
    )

    # Create evaluator (using global parameters)
    evaluator = UniformKLEvaluator(
        bin_edges=BIN_EDGES,
        is_2d=IS_2D
    )

    # Debug: Show initial parameters
    print("\n" + "="*60)
    print("INITIAL PARAMETERS")
    print("="*60)
    print(f"Initial mean (normalized [0,1]): {INITIAL_MEAN}")
    bounds = bias.get_parameter_bounds()
    initial_params_denorm = bounds[:, 0] + INITIAL_MEAN * (bounds[:, 1] - bounds[:, 0])
    print(f"Initial parameters (denormalized):")
    print(f"  Centers: {initial_params_denorm[:hills_per_d]}")
    print(f"  Heights: {initial_params_denorm[hills_per_d:2*hills_per_d]}")
    print(f"  Widths:  {initial_params_denorm[2*hills_per_d:]}")
    print("="*60 + "\n")

    # Create workflow (using global parameters)
    # Note: CMAESWorkflow manages the entire optimization loop internally
    workflow = CMAESWorkflow(
        bias=bias,
        sampler=sampler,
        evaluator=evaluator,
        initial_mean=INITIAL_MEAN,
        sigma=SIGMA,
        population_size=POPULATION_SIZE,
        max_generations=MAX_GENERATIONS,
        n_workers=N_WORKERS,
        n_replicas=N_REPLICAS,
        early_stop_patience=EARLY_STOP_PATIENCE
    )

    # Run optimization
    # The workflow handles the entire ask/run/evaluate/tell loop internally
    output_dir = SCRIPT_DIR / "output_polyproline"
    result = workflow.optimize(str(output_dir))

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
    print(f"\nBest parameters:")
    for i, param in enumerate(result['best_parameters']):
        print(f"  {i}: {param:.6f}")

    print(f"\nResults saved to: {output_dir}")

    # Save result dictionary for later plotting
    # Note: A checkpoint file is also saved after each generation as optimization_checkpoint.pkl
    result_file = output_dir / "optimization_result.pkl"
    with open(result_file, 'wb') as f:
        pickle.dump({'result': result, 'bias': bias}, f)
    print(f"✓ Result saved to: {result_file}")
    
    # Also note the checkpoint file location
    checkpoint_file = output_dir / "optimization_checkpoint.pkl"
    if checkpoint_file.exists():
        print(f"✓ Checkpoint file: {checkpoint_file}")
        print(f"  (This file is updated after each generation and can be used to resume or analyze partial results)")

    return result, bias, output_dir


def resume_optimization(checkpoint_file=None):
    """Resume optimization from a checkpoint file."""
    # Get the directory containing this script
    SCRIPT_DIR = Path(__file__).parent.resolve()
    
    if checkpoint_file is None:
        checkpoint_file = SCRIPT_DIR / "output_polyproline" / "optimization_checkpoint.pkl"
    
    checkpoint_file = Path(checkpoint_file)
    
    if not checkpoint_file.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_file}")
        print("Run optimization first with: python polyproline_example.py run")
        return None, None, None, None
    
    print("\n" + "="*60)
    print("RESUMING ALANINE DIPEPTIDE CMA-ES OPTIMIZATION")
    print("="*60)
    print(f"Checkpoint: {checkpoint_file}")
    
    # Recreate the same setup as original optimization (using global parameters)
    sampler = OpenMMPlumedSampler(
        pdb_file=[str(SCRIPT_DIR / "pp1.pdb"), str(SCRIPT_DIR / "pp2.pdb")],
        forcefield_files=["amber14-all.xml", "amber14/tip3pfb.xml"],
        temperature=TEMPERATURE,
        time_step=TIME_STEP,
        friction=FRICTION,
        simulation_steps=SIMULATION_STEPS,
        report_interval=REPORT_INTERVAL
    )
    
    bias = PlumedHillBias(
        plumed_template=str(SCRIPT_DIR / "plumed_template.dat"),
        hills_per_d=HILLS_PER_D,
        hills_space=HILLS_SPACE,
        hills_height=HILLS_HEIGHT,
        hills_width=HILLS_WIDTH
    )
    
    evaluator = UniformKLEvaluator(
        bin_edges=BIN_EDGES,
        is_2d=IS_2D
    )
    
    # Load checkpoint to get original settings
    import pickle
    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Extract original optimizer settings from checkpoint
    original_pop_size = checkpoint_data.get('population_size', 6)
    original_n_workers = checkpoint_data.get('n_workers', 1)
    original_n_replicas = checkpoint_data.get('n_replicas', 1)
    original_sigma = checkpoint_data.get('sigma', 0.1)
    original_max_gen = checkpoint_data.get('max_generations', 75)
    
    print(f"\nRestoring original settings from checkpoint:")
    print(f"  Population size: {original_pop_size}")
    print(f"  Workers: {original_n_workers}")
    print(f"  Replicas: {original_n_replicas}")
    print(f"  Sigma: {original_sigma}")
    print(f"  Max generations: {original_max_gen}")
    
    # Create workflow with SAME settings as original
    workflow = CMAESWorkflow(
        bias=bias,
        sampler=sampler,
        evaluator=evaluator,
        initial_mean=None,
        sigma=original_sigma,
        population_size=original_pop_size,
        max_generations=original_max_gen,
        n_workers=original_n_workers,
        n_replicas=original_n_replicas
    )
    
    # Resume from checkpoint
    output_dir = SCRIPT_DIR / "output_polyproline"
    result = workflow.optimize(str(output_dir), resume_from=str(checkpoint_file))
    
    # Check if optimization was interrupted again
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
    print(f"\nBest parameters:")
    for i, param in enumerate(result['best_parameters']):
        print(f"  {i}: {param:.6f}")
    
    print(f"\nResults saved to: {output_dir}")
    
    # Save final result
    result_file = output_dir / "optimization_result.pkl"
    with open(result_file, 'wb') as f:
        pickle.dump({'result': result, 'bias': bias}, f)
    print(f"✓ Result saved to: {result_file}")
    
    return result, bias, output_dir


def generate_plots(result_file=None):
    """Generate plots from saved optimization results."""
    SCRIPT_DIR = Path(__file__).parent.resolve()
    
    if result_file is None:
        # Try checkpoint file first, then final result file
        checkpoint_file = SCRIPT_DIR / "output_polyproline" / "optimization_checkpoint.pkl"
        result_file_path = SCRIPT_DIR / "output_polyproline" / "optimization_result.pkl"
        
        if result_file_path.exists():
            result_file = result_file_path
            print("Using final result file (optimization completed)")
        elif checkpoint_file.exists():
            result_file = checkpoint_file
            print("Using checkpoint file (optimization may have been interrupted)")
        else:
            print(f"Error: No result files found in {SCRIPT_DIR / 'output_polyproline'}")
            print("Run optimization first with: python polyproline_example.py run")
            return
    
    result_file = Path(result_file)
    
    if not result_file.exists():
        print(f"Error: Result file not found: {result_file}")
        print("Run optimization first with: python polyproline_example.py run")
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


def create_all_plots(result, bias, output_dir):
    """Create all visualization plots from optimization results.
    
    Args:
        result: Optimization result dictionary
        bias: Bias object
        output_dir: Directory to save plots (Path object)
    """
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    from pycmaetad.visualization import (
        plot_convergence, plot_parameter_evolution, plot_sigma_evolution, 
        plot_bias_landscape_1d, plot_bias_evolution, plot_convergence_diagnostics,
        plot_cv_histogram_evolution
    )
    from pycmaetad.visualization.colvar import plot_colvar_evolution, plot_cv_histogram, plot_cv_time_series
    
    # Core convergence plots
    plot_convergence(result, output_path=output_dir / "convergence.png")
    plot_convergence_diagnostics(result, output_path=output_dir / "convergence_diagnostics.png")
    plot_parameter_evolution(result, bias, output_path=output_dir / "parameter_evolution.png")
    plot_sigma_evolution(result, output_path=output_dir / "sigma_evolution.png")
    
    # Bias landscape evolution for ALL generations
    bias_landscapes_dir = output_dir / "bias_landscapes"
    plot_bias_evolution(
        bias=bias,
        result=result,
        cv_range=CV_RANGE,
        output_dir=bias_landscapes_dir,
        generations='all',
        show_best_only=False
    )
    
    # CV histogram evolution for ALL generations
    cv_histograms_dir = output_dir / "cv_histograms"
    plot_cv_histogram_evolution(
        result=result,
        output_dir=cv_histograms_dir,
        generations='all',
        n_bins=30,
        show_uniform=True,
        cv_range=CV_RANGE,
        individual='best'  # Only best individual (consistent with time series)
    )
    
    # Quick reference: best generation landscape
    best_gen = result.get('best_generation', 0)
    bounds = bias.get_parameter_bounds()
    
    if 'history' in result and best_gen < len(result['history']):
        history_entry = result['history'][best_gen]
        if 'population' in history_entry:
            # Denormalize all solutions from the population
            all_params = [
                bounds[:, 0] + norm_params * (bounds[:, 1] - bounds[:, 0])
                for norm_params in history_entry['population']
            ]
        elif 'best_solution' in history_entry:
            # Fallback: use best_solution from history entry
            best_params_norm = history_entry['best_solution']
            all_params = [bounds[:, 0] + best_params_norm * (bounds[:, 1] - bounds[:, 0])]
        else:
            # Final fallback: use top-level best_parameters
            best_params_norm = result.get('best_parameters')
            if best_params_norm is not None:
                all_params = [bounds[:, 0] + best_params_norm * (bounds[:, 1] - bounds[:, 0])]
            else:
                all_params = None
    else:
        # Use top-level best parameters
        best_params_norm = result.get('best_parameters')
        if best_params_norm is not None:
            all_params = [bounds[:, 0] + best_params_norm * (bounds[:, 1] - bounds[:, 0])]
        else:
            all_params = None
    
    # Only plot if we have parameters
    if all_params is not None:
        plot_bias_landscape_1d(
            bias, all_params, cv_range=CV_RANGE,
            generation=best_gen,
            output_path=output_dir / "bias_landscape_best_gen.png",
            mintozero=True
        )
    
    # Colvar plots
    plot_colvar_evolution(result, output_path=output_dir / "colvar_evolution.png", individual='all')
    plot_cv_histogram(result, output_path=output_dir / "cv_histogram.png")
    plot_cv_time_series(result, output_path=output_dir / "cv_time_series.png", individual='best')

    print(f"\n✅ All plots saved to: {output_dir}")


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
            result, bias, output_dir = run_optimization()
            if result is not None:
                create_all_plots(result, bias, output_dir)
        else:
            print("Unknown command. Usage:")
            print("  python polyproline_example.py run              # Run optimization from start")
            print("  python polyproline_example.py resume [file]    # Resume from checkpoint")
            print("  python polyproline_example.py plot [file]      # Generate plots from saved results")
            print("  python polyproline_example.py both             # Run optimization and generate plots")
    else:
        # Default: run both optimization and plotting
        result, bias, output_dir = run_optimization()
        if result is not None:
            create_all_plots(result, bias, output_dir)

