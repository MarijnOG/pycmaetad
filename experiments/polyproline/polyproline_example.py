
"""
Polyproline CMA-ES Optimization Experiment
=========================================

This script runs a CMA-ES optimization of bias parameters for uniform sampling
of polyproline conformational space using OpenMM and PLUMED via pycmaetad.
It supports running, resuming, and plotting results for different configurations.

Usage:
    python polyproline_example.py [--config CONFIG] run              # Run optimization from start
    python polyproline_example.py [--config CONFIG] resume [file]    # Resume from checkpoint
    python polyproline_example.py plot [file]                        # Generate plots from saved results
    python polyproline_example.py [--config CONFIG] both             # Run optimization and generate plots (default)

Arguments:
    --config   Path to configuration file (default: configs/config_default.py)
    run        Run optimization only
    resume     Resume optimization from checkpoint file
    plot       Generate plots from saved results
    both       Run optimization and generate plots (default)

Outputs:
    - Optimization results and checkpoints
    - Diagnostic plots: convergence, parameter evolution, bias landscape, CV histograms
    - Pickled result and bias objects for further analysis
"""

import numpy as np
import pickle
import sys
import argparse
import importlib.util
from pathlib import Path
# Import pycmaetad modules for bias, sampling, optimization, and evaluation
from pycmaetad.bias import PlumedHillBias
from pycmaetad.sampler import OpenMMPlumedSampler
from pycmaetad.optimizer import CMAESWorkflow
from pycmaetad.evaluator import HybridUniformEvaluator, UniformKLEvaluator



# =================== CONFIGURATION LOADING ===================


def load_config(config_path):
    """
    Load configuration from a Python file.
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



# =================== OPTIMIZATION FUNCTIONS ===================



def run_optimization():
    """
    Run the CMA-ES optimization and save results.
    """
    SCRIPT_DIR = Path(__file__).parent.resolve()

    # ---- Logging: Run info ----
    print("\n" + "="*60)
    print("POLYPROLINE CMA-ES OPTIMIZATION")
    print("Optimizing bias parameters for uniform sampling")
    print("="*60)
    print(f"Configuration: {CONFIG['name']}")
    print(f"Description: {CONFIG['description']}")
    print(f"Working directory: {SCRIPT_DIR}")

    # ---- Create sampler ----
    sampler = OpenMMPlumedSampler(
        pdb_file=[str(SCRIPT_DIR / pdb_file) for pdb_file in CONFIG['pdb_files']],
        forcefield_files=CONFIG['forcefield_files'],
        temperature=CONFIG['temperature'],
        time_step=CONFIG['time_step'],
        friction=CONFIG['friction'],
        simulation_steps=CONFIG['simulation_steps'],
        report_interval=CONFIG['report_interval']
    )

    # ---- Create bias ----
    bias = PlumedHillBias(
        plumed_template=str(SCRIPT_DIR / CONFIG['plumed_template']),
        hills_per_d=CONFIG['hills_per_d'],
        hills_space=CONFIG['hills_space'],
        hills_height=CONFIG['hills_height'],
        hills_width=CONFIG['hills_width']
    )

    # ---- Create evaluator ----
    evaluator = UniformKLEvaluator(
        bin_edges=CONFIG['bin_edges'],
        is_2d=CONFIG['is_2d'],
    )

    # ---- Debug: Show initial parameters ----
    print("\n" + "="*60)
    print("INITIAL PARAMETERS")
    print("="*60)
    print(f"Initial mean (normalized [0,1]): {CONFIG['initial_mean']}")
    bounds = bias.get_parameter_bounds()
    initial_params_denorm = bounds[:, 0] + CONFIG['initial_mean'] * (bounds[:, 1] - bounds[:, 0])
    hills_per_d = CONFIG['hills_per_d']
    print(f"Initial parameters (denormalized):")
    print(f"  Centers: {initial_params_denorm[:hills_per_d]}")
    print(f"  Heights: {initial_params_denorm[hills_per_d:2*hills_per_d]}")
    print(f"  Widths:  {initial_params_denorm[2*hills_per_d:]}")
    print("="*60 + "\n")

    # ---- Create workflow ----
    workflow = CMAESWorkflow(
        bias=bias,
        sampler=sampler,
        evaluator=evaluator,
        initial_mean=CONFIG['initial_mean'],
        sigma=CONFIG['sigma'],
        population_size=CONFIG['population_size'],
        max_generations=CONFIG['max_generations'],
        n_workers=CONFIG['n_workers'],
        n_replicas=CONFIG['n_replicas'],
        early_stop_patience=CONFIG['early_stop_patience']
    )

    # ---- Run optimization ----
    output_dir = SCRIPT_DIR / "output_polyproline"
    result = workflow.optimize(str(output_dir))

    # ---- Check if optimization was interrupted ----
    if result is None:
        print("\n⚠️  Optimization was interrupted before completing any generations")
        return None, bias, output_dir

    # ---- Print results ----
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

    # ---- Save result dictionary for later plotting ----
    result_file = output_dir / "optimization_result.pkl"
    with open(result_file, 'wb') as f:
        pickle.dump({'result': result, 'bias': bias}, f)
    print(f"✓ Result saved to: {result_file}")

    # ---- Also note the checkpoint file location ----
    checkpoint_file = output_dir / "optimization_checkpoint.pkl"
    if checkpoint_file.exists():
        print(f"✓ Checkpoint file: {checkpoint_file}")
        print(f"  (This file is updated after each generation and can be used to resume or analyze partial results)")

    return result, bias, output_dir



def resume_optimization(checkpoint_file=None):
    """
    Resume optimization from a checkpoint file.
    """
    SCRIPT_DIR = Path(__file__).parent.resolve()

    if checkpoint_file is None:
        checkpoint_file = SCRIPT_DIR / "output_polyproline" / "optimization_checkpoint.pkl"
    checkpoint_file = Path(checkpoint_file)
    if not checkpoint_file.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_file}")
        print("Run optimization first with: python polyproline_example.py run")
        return None, None, None, None

    # ---- Logging: Resume info ----
    print("\n" + "="*60)
    print("RESUMING POLYPROLINE CMA-ES OPTIMIZATION")
    print("="*60)
    print(f"Checkpoint: {checkpoint_file}")

    # ---- Recreate setup as original optimization ----
    sampler = OpenMMPlumedSampler(
        pdb_file=[str(SCRIPT_DIR / pdb_file) for pdb_file in CONFIG['pdb_files']],
        forcefield_files=CONFIG['forcefield_files'],
        temperature=CONFIG['temperature'],
        time_step=CONFIG['time_step'],
        friction=CONFIG['friction'],
        simulation_steps=CONFIG['simulation_steps'],
        report_interval=CONFIG['report_interval']
    )
    bias = PlumedHillBias(
        plumed_template=str(SCRIPT_DIR / CONFIG['plumed_template']),
        hills_per_d=CONFIG['hills_per_d'],
        hills_space=CONFIG['hills_space'],
        hills_height=CONFIG['hills_height'],
        hills_width=CONFIG['hills_width']
    )
    evaluator = UniformKLEvaluator(
        bin_edges=CONFIG['bin_edges'],
        is_2d=CONFIG['is_2d']
    )

    # ---- Load checkpoint to get original settings ----
    import pickle
    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)
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

    # ---- Create workflow with SAME settings as original ----
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

    # ---- Resume from checkpoint ----
    output_dir = SCRIPT_DIR / "output_polyproline"
    result = workflow.optimize(str(output_dir), resume_from=str(checkpoint_file))

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
    print(f"\nBest parameters:")
    for i, param in enumerate(result['best_parameters']):
        print(f"  {i}: {param:.6f}")

    print(f"\nResults saved to: {output_dir}")

    # ---- Save final result ----
    result_file = output_dir / "optimization_result.pkl"
    with open(result_file, 'wb') as f:
        pickle.dump({'result': result, 'bias': bias}, f)
    print(f"✓ Result saved to: {result_file}")

    return result, bias, output_dir



def generate_plots(result_file=None):
    """
    Generate plots from saved optimization results.
    """
    SCRIPT_DIR = Path(__file__).parent.resolve()

    # ---- Locate result or checkpoint file ----
    if result_file is None:
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

    # ---- Load saved results ----
    print(f"\nLoading results from: {result_file}")
    with open(result_file, 'rb') as f:
        data = pickle.load(f)
    result = data['result']
    bias = data['bias']
    output_dir = result_file.parent

    # ---- Use the centralized plotting function ----
    create_all_plots(result, bias, output_dir)



def create_all_plots(result, bias, output_dir):
    """
    Create all visualization plots from optimization results.
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

    # ---- Core convergence plots ----
    plot_convergence(result, output_path=output_dir / "convergence.png")
    plot_convergence_diagnostics(result, output_path=output_dir / "convergence_diagnostics.png")

    # ---- Parameter evolution plots ----
    hills_per_d = CONFIG['hills_per_d']
    param_names = (
        [f'center_{i}' for i in range(hills_per_d)] +
        [f'height_{i}' for i in range(hills_per_d)] +
        [f'width_{i}' for i in range(hills_per_d)]
    )
    plot_parameter_evolution(result, bias, param_names=param_names, output_path=output_dir / "parameter_evolution.png")
    plot_sigma_evolution(result, output_path=output_dir / "sigma_evolution.png")

    # ---- Bias landscape evolution for ALL generations ----
    bias_landscapes_dir = output_dir / "bias_landscapes"
    plot_bias_evolution(
        bias=bias,
        result=result,
        cv_range=CONFIG['cv_range'],
        output_dir=bias_landscapes_dir,
        generations='all',
        show_best_only=False,
        periodic=True,
        mintozero=True
    )

    # ---- CV histogram evolution for ALL generations ----
    cv_histograms_dir = output_dir / "cv_histograms"
    plot_cv_histogram_evolution(
        result=result,
        output_dir=cv_histograms_dir,
        generations='all',
        n_bins=30,
        show_uniform=True,
        cv_range=CONFIG['cv_range'],
        individual='best'  # Only best individual (consistent with time series)
    )

    # ---- Quick reference: best generation landscape ----
    best_gen = result.get('best_generation', 0)
    bounds = bias.get_parameter_bounds()
    if 'history' in result and best_gen < len(result['history']):
        history_entry = result['history'][best_gen]
        if 'population' in history_entry:
            all_params = [
                bounds[:, 0] + norm_params * (bounds[:, 1] - bounds[:, 0])
                for norm_params in history_entry['population']
            ]
        elif 'best_solution' in history_entry:
            best_params_norm = history_entry['best_solution']
            all_params = [bounds[:, 0] + best_params_norm * (bounds[:, 1] - bounds[:, 0])]
        else:
            best_params_norm = result.get('best_parameters')
            if best_params_norm is not None:
                all_params = [bounds[:, 0] + best_params_norm * (bounds[:, 1] - bounds[:, 0])]
            else:
                all_params = None
    else:
        best_params_norm = result.get('best_parameters')
        if best_params_norm is not None:
            all_params = [bounds[:, 0] + best_params_norm * (bounds[:, 1] - bounds[:, 0])]
        else:
            all_params = None

    # ---- Only plot if we have parameters ----
    if all_params is not None:
        plot_bias_landscape_1d(
            bias, all_params, cv_range=CONFIG['cv_range'],
            generation=best_gen,
            output_path=output_dir / "bias_landscape_best_gen.png",
            periodic=True,
            mintozero=True
        )

    # ---- Colvar plots ----
    plot_colvar_evolution(result, output_path=output_dir / "colvar_evolution.png", individual='all')
    plot_cv_histogram(result, output_path=output_dir / "cv_histogram.png")
    plot_cv_time_series(result, output_path=output_dir / "cv_time_series.png", individual='best')
    plot_cv_time_series(result, output_path=output_dir / "cv_time_series_100.png", generation=100, individual='best')

    print(f"\n✅ All plots saved to: {output_dir}")



# =================== ENTRY POINT ===================

if __name__ == "__main__":
    """
    Parse command-line arguments and run the requested experiment mode.
    """
    parser = argparse.ArgumentParser(description='Polyproline CMA-ES optimization')
    parser.add_argument('command', nargs='?', default='both', 
                       choices=['run', 'resume', 'plot', 'both'],
                       help='Command to execute (default: both)')
    parser.add_argument('--config', type=str, 
                       default=str(Path(__file__).parent / 'configs' / 'config_default.py'),
                       help='Path to configuration file')
    parser.add_argument('extra', nargs='*', help='Additional arguments (e.g., checkpoint file)')

    args = parser.parse_args()

    # ---- Load configuration ----
    CONFIG = load_config(args.config)
    print(f"Loaded config: {args.config}")

    command = args.command.lower()

    if command == "run":
        run_optimization()
    elif command == "resume":
        checkpoint_file = args.extra[0] if args.extra else None
        resume_optimization(checkpoint_file)
    elif command == "plot":
        result_file = args.extra[0] if args.extra else None
        generate_plots(result_file)
    elif command == "both":
        result, bias, output_dir = run_optimization()
        if result is not None:
            create_all_plots(result, bias, output_dir)
    else:
        print("Unknown command. Usage:")
        print("  python polyproline_example.py [--config CONFIG] run              # Run optimization from start")
        print("  python polyproline_example.py [--config CONFIG] resume [file]    # Resume from checkpoint")
        print("  python polyproline_example.py plot [file]                        # Generate plots from saved results")
        print("  python polyproline_example.py [--config CONFIG] both             # Run optimization and generate plots (default)")
        print("\nAvailable configs:")
        print("  configs/config_default.py - Default parameters")

