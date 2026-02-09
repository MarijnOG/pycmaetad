"""Bias potential visualization."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_bias_landscape_1d(bias, all_params, cv_range, generation=None, 
                           output_path=None, show=False, n_points=500,
                           save_data=True, periodic=True, mintozero=False):
    """Plot 1D free energy surface for all individuals in a generation.
    
    Uses the bias.sum_hills() method to compute the FES (inverted bias).
    
    Args:
        bias: Bias object with sum_hills() method
        all_params: List of parameter arrays for each individual in the generation
        cv_range: Tuple of (min, max) for CV values
        generation: Generation number (optional, for title)
        output_path: Path to save plot (PNG)
        show: Whether to display plot
        n_points: Number of points to evaluate
        save_data: If True, also save raw data as .npz file
        periodic: If True, use periodic boundaries (for dihedral angles)
        mintozero: If True, shift FES minimum to zero
        
    Returns:
        matplotlib Figure object
    """
    # Ensure all_params is a list
    if not isinstance(all_params, list):
        all_params = [all_params]
    
    # Check if bias supports sum_hills
    if not hasattr(bias, 'sum_hills'):
        print("Warning: Bias type doesn't support sum_hills() method")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use a colormap for different individuals
    cmap = plt.cm.get_cmap('tab10')
    
    # Store data for saving
    all_landscapes = []
    cv_values_ref = None
    
    # Plot each individual
    for ind_idx, params in enumerate(all_params):
        bias.set_parameters(params)
        
        # Compute FES using sum_hills with periodic boundaries
        cv_values, fes_values = bias.sum_hills(cv_range, n_points, periodic=periodic, mintozero=mintozero)
        
        if cv_values_ref is None:
            cv_values_ref = cv_values
        
        all_landscapes.append(fes_values)
        
        color = cmap(ind_idx % 10)
        ax.plot(cv_values, fes_values, '-', linewidth=2, alpha=0.7, 
               color=color, label=f'Ind {ind_idx}')
        
        # Mark hill centers for this individual
        if hasattr(bias, '_centers'):
            for center in bias._centers:
                ax.axvline(center, color=color, linestyle=':', alpha=0.2, linewidth=1)
    
    ax.set_xlabel('Collective Variable (rad)', fontsize=12)
    ax.set_ylabel('Free Energy (kJ/mol)', fontsize=12)
    
    # Create title with generation number
    if generation is not None:
        title = f'Free Energy Surface - Generation {generation} ({len(all_params)} individuals)'
    else:
        title = f'Free Energy Surface ({len(all_params)} individuals)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Adjust legend based on number of individuals
    if len(all_params) <= 10:
        ax.legend(fontsize=9, ncol=2)
    else:
        ax.legend(fontsize=8, ncol=3, loc='upper right')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved plot: {output_path.name}")
        
        # Save raw data if requested
        if save_data and cv_values_ref is not None:
            data_path = output_path.with_suffix('.npz')
            np.savez(
                data_path,
                cv_values=cv_values_ref,
                landscapes=np.array(all_landscapes),
                generation=generation if generation is not None else -1,
                n_individuals=len(all_params)
            )
            print(f"  ✓ Saved data: {data_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_bias_evolution(bias, result, cv_range, output_dir, n_points=500, 
                        generations='all', show_best_only=False, mintozero=False):
    """Plot free energy surface evolution across generations.
    
    Creates individual plots for each generation and saves raw data.
    
    Args:
        bias: Bias object with sum_hills() method
        result: Optimization result dictionary with 'history'
        cv_range: Tuple of (min, max) for CV values
        output_dir: Directory to save plots and data
        n_points: Number of points to evaluate per landscape
        generations: 'all' or list of generation indices to plot
        show_best_only: If True, only plot best individual per generation
        mintozero: If True, shift FES minimum to zero
        
    Returns:
        List of figure objects
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history = result['history']
    bounds = bias.get_parameter_bounds()
    
    # Determine which generations to plot
    if generations == 'all':
        gen_indices = range(len(history))
    else:
        gen_indices = generations
    
    figures = []
    
    print("\n" + "="*60)
    print("GENERATING BIAS LANDSCAPE EVOLUTION")
    print("="*60)
    
    for gen_idx in gen_indices:
        if gen_idx >= len(history):
            print(f"  ⚠️  Generation {gen_idx} not in history, skipping")
            continue
            
        gen_data = history[gen_idx]
        gen_num = gen_data['generation']
        
        # Get parameters for this generation
        if show_best_only:
            # Just best individual
            best_params = gen_data.get('best_solution')
            if best_params is None:
                print(f"  ⚠️  Generation {gen_num} missing best_solution, skipping")
                continue
            # Denormalize
            all_params = [bounds[:, 0] + best_params * (bounds[:, 1] - bounds[:, 0])]
        else:
            # All individuals from population
            if 'population' in gen_data:
                all_params = []
                for normalized_params in gen_data['population']:
                    actual_params = bounds[:, 0] + normalized_params * (bounds[:, 1] - bounds[:, 0])
                    all_params.append(actual_params)
            elif 'best_solution' in gen_data:
                # Fallback: just best
                best_params = gen_data['best_solution']
                all_params = [bounds[:, 0] + best_params * (bounds[:, 1] - bounds[:, 0])]
            else:
                print(f"  ⚠️  Generation {gen_num} missing population data, skipping")
                continue
        
        # Create plot for this generation
        suffix = "_best" if show_best_only else "_all"
        plot_path = output_dir / f"bias_landscape_gen{gen_num:03d}{suffix}.png"
        
        fig = plot_bias_landscape_1d(
            bias=bias,
            all_params=all_params,
            cv_range=cv_range,
            generation=gen_num,
            output_path=plot_path,
            n_points=n_points,
            save_data=True,
            mintozero=mintozero
        )
        
        figures.append(fig)
    
    print("="*60)
    print(f"✅ Generated {len(figures)} bias landscape plots")
    print(f"   Saved to: {output_dir}")
    print("="*60 + "\n")
    
    return figures


def plot_cv_histogram(colvar_file, cv_range=None, output_path=None, show=False, n_bins=50):
    """Plot histogram of collective variable values from COLVAR file.
    
    Args:
        colvar_file: Path to COLVAR file
        cv_range: Tuple of (min, max) for binning (or None for auto)
        output_path: Path to save plot
        show: Whether to display
        n_bins: Number of histogram bins
        
    Returns:
        matplotlib Figure object
    """
    # Load COLVAR
    data = np.loadtxt(colvar_file, comments='#')
    if data.ndim == 1:
        cv_values = data[1] if len(data) > 1 else np.array([data[1]])
    else:
        cv_values = data[:, 1]  # Second column is CV
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if cv_range:
        bins = np.linspace(cv_range[0], cv_range[1], n_bins + 1)
    else:
        bins = n_bins
    
    counts, edges, patches = ax.hist(cv_values, bins=bins, density=True, 
                                      alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add uniform reference line
    if cv_range:
        uniform_density = 1.0 / (cv_range[1] - cv_range[0])
        ax.axhline(uniform_density, color='red', linestyle='--', linewidth=2,
                  label='Uniform distribution')
    
    ax.set_xlabel('Collective Variable', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Collective Variable Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {Path(output_path).name}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig
