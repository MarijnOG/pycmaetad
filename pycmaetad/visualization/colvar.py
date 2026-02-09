import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_colvar_evolution(result, output_path=None, show=False, individual='best'):
    """Plot the evolution of the collective variable (CV) space occupation over generations.
    
    Args:
        result: Optimizer result dictionary with generation output directories
        output_path: Path to save the plot
        show: Whether to display the plot
        individual: Which individual(s) to plot:
            - 'best': Only the best individual from each generation (default)
            - 'all': All individuals combined (shows population diversity)
            - int: Specific individual index across all generations
        
    Returns:
        matplotlib Figure object
    """
    # Extract COLVAR files from generation directories
    history = result.get('history', [])
    
    if not history:
        print("Warning: No history found in result")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect CV values per generation
    for gen_idx, h in enumerate(history):
        gen_dir = h.get('output_dir')
        if not gen_dir:
            continue
            
        gen_path = Path(gen_dir)
        if not gen_path.exists():
            continue
        
        # Determine which individual(s) to use
        if individual == 'best':
            # Find best individual by score
            scores = h.get('all_scores', [])
            if scores:
                best_idx = np.argmin(scores)
            else:
                best_idx = 0
            ind_indices = [best_idx]
        elif individual == 'all':
            # Use all individuals
            ind_dirs = sorted(gen_path.glob('ind*'))
            ind_indices = list(range(len(ind_dirs)))
        elif isinstance(individual, int):
            ind_indices = [individual]
        else:
            print(f"Warning: Unknown individual selector '{individual}'")
            continue
        
        # Collect CV values from selected individual(s)
        all_cv_values = []
        
        for ind_idx in ind_indices:
            # Find COLVAR files for this individual
            ind_pattern = f'ind{ind_idx:03d}'
            colvar_paths = list(gen_path.glob(f'{ind_pattern}/COLVAR'))
            if not colvar_paths:
                colvar_paths = list(gen_path.glob(f'{ind_pattern}/replica_*/COLVAR'))
            
            # Read COLVAR files
            for colvar_path in colvar_paths:
                try:
                    data = np.loadtxt(colvar_path, skiprows=1)
                    if data.ndim == 1:
                        data = data.reshape(1, -1)
                    cv_values = data[:, 1]  # CV values in second column
                    all_cv_values.extend(cv_values)
                except Exception as e:
                    print(f"Warning: Could not read {colvar_path}: {e}")
                    continue
        
        if not all_cv_values:
            continue
        
        all_cv_values = np.array(all_cv_values)
        
        # Create x-coordinates (all same generation number)
        gen_coords = np.full_like(all_cv_values, gen_idx)
        
        # Plot points for this generation
        ax.scatter(gen_coords, all_cv_values, alpha=0.3, s=1, 
                  color=f'C{gen_idx % 10}', rasterized=True)

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Collective Variable (CV)', fontsize=12)
    
    # Update title based on mode
    if individual == 'best':
        title = 'CV Space Exploration - Best Individual Per Generation'
    elif individual == 'all':
        title = 'CV Space Exploration - All Individuals (Population Diversity)'
    else:
        title = f'CV Space Exploration - Individual #{individual}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlim(-0.5, len(history) - 0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {Path(output_path).name}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_cv_histogram(result, generation=None, output_path=None, show=False):
    """Plot histogram of collective variable (CV) values from a COLVAR file.
    
    Args:
        result: Optimizer result dictionary with generation output directories
        generation: Generation number to plot (if None, uses last generation)
        output_path: Path to save the plot
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    history = result.get('history', [])
    if not history:
        print("Warning: No history found in result")
        return None
    
    # Select generation
    if generation is None:
        gen_data = history[-1]
    else:
        gen_data = next((h for h in history if h['generation'] == generation), None)
        if gen_data is None:
            print(f"Warning: Generation {generation} not found")
            return None
    
    # Find COLVAR file
    gen_dir = gen_data.get('output_dir')
    if not gen_dir:
        print("Warning: No output directory found")
        return None
    
    gen_path = Path(gen_dir)
    # Try to find COLVAR files - handle both replica and non-replica structures
    colvar_paths = list(gen_path.glob('*/COLVAR'))
    if not colvar_paths:
        colvar_paths = list(gen_path.glob('*/replica_*/COLVAR'))
    if not colvar_paths:
        print(f"Warning: No COLVAR files found in {gen_dir}")
        return None
    
    # Read COLVAR file
    data = np.loadtxt(colvar_paths[0], skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    cv_values = data[:, 1]  # Assuming CV values are in the second column

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cv_values, bins=50, alpha=0.7, color='blue', density=True)
    ax.set_xlabel('Collective Variable (CV)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    gen_num = gen_data['generation']
    ax.set_title(f'Histogram of CV Values - Generation {gen_num}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {Path(output_path).name}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_cv_time_series(result, generation=None, individual='best', output_path=None, show=False):
    """Plot time series of collective variable (CV) values.
    
    Args:
        result: Optimizer result dictionary with generation output directories
        generation: Generation number to plot (if None, uses best generation)
        individual: Which individual(s) to plot:
            - 'best': Only the best individual from the generation (default)
            - 'all': All individuals (can be cluttered)
            - int: Specific individual index
        output_path: Path to save the plot
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    history = result.get('history', [])
    if not history:
        print("Warning: No history found in result")
        return None
    
    # Select generation (default to best, not last)
    if generation is None:
        best_gen = result.get('best_generation', len(history) - 1)
        gen_data = history[best_gen]
    else:
        gen_data = next((h for h in history if h['generation'] == generation), None)
        if gen_data is None:
            print(f"Warning: Generation {generation} not found")
            return None
    
    # Find COLVAR files
    gen_dir = gen_data.get('output_dir')
    if not gen_dir:
        print("Warning: No output directory found")
        return None
    
    gen_path = Path(gen_dir)
    
    # Find all individual directories
    ind_dirs = sorted(gen_path.glob('ind*'))
    if not ind_dirs:
        print(f"Warning: No individual directories found in {gen_dir}")
        return None

    # Determine which individuals to plot
    if individual == 'best':
        # Find best individual by score
        scores = gen_data.get('all_scores', [])
        if scores:
            best_idx = np.argmin(scores)
            individuals_to_plot = [best_idx]
            plot_title_suffix = f" - Best Individual (#{best_idx})"
        else:
            individuals_to_plot = [0]
            plot_title_suffix = " - Individual 0"
    elif individual == 'all':
        individuals_to_plot = list(range(len(ind_dirs)))
        plot_title_suffix = " - All Individuals"
    elif isinstance(individual, int):
        if individual >= len(ind_dirs):
            print(f"Warning: Individual {individual} not found, only {len(ind_dirs)} available")
            return None
        individuals_to_plot = [individual]
        plot_title_suffix = f" - Individual #{individual}"
    else:
        print(f"Warning: Unknown individual selector '{individual}'")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get colormap for individuals
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(individuals_to_plot))))
    
    # Plot selected individuals
    for plot_idx, ind_idx in enumerate(individuals_to_plot):
        ind_dir = ind_dirs[ind_idx]
        
        # Find all COLVAR files for this individual (handles replicas)
        colvar_paths = list(ind_dir.glob('COLVAR'))
        if not colvar_paths:
            colvar_paths = list(ind_dir.glob('replica_*/COLVAR'))
        
        # Plot all replicas for this individual
        for replica_idx, colvar_path in enumerate(colvar_paths):
            try:
                data = np.loadtxt(colvar_path, skiprows=1)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                time = data[:, 0]
                cv_values = data[:, 1]
                
                # Label strategy depends on what we're plotting
                if individual == 'all':
                    label = f'Ind {ind_idx}' if replica_idx == 0 else None
                    alpha = 0.4
                    marker_size = 1
                elif len(colvar_paths) > 1:
                    # Multiple replicas for single individual
                    label = f'Replica {replica_idx}'
                    alpha = 0.6
                    marker_size = 3
                else:
                    label = None
                    alpha = 0.7
                    marker_size = 3
                
                ax.scatter(time, cv_values, color=colors[plot_idx], alpha=alpha, 
                          s=marker_size, label=label, rasterized=True)
            except Exception as e:
                print(f"Warning: Could not read {colvar_path}: {e}")
                continue
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Collective Variable (CV)', fontsize=12)
    gen_num = gen_data['generation']
    ax.set_title(f'Time Series of CV Values - Generation {gen_num}{plot_title_suffix}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Show legend if useful
    if individual == 'all' and len(individuals_to_plot) <= 10:
        ax.legend(loc='best', markerscale=3, fontsize=8, ncol=2)
    elif len(colvar_paths) > 1 and individual != 'all':
        ax.legend(loc='best', markerscale=3, fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {Path(output_path).name}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig





def plot_cv_histogram_evolution(result, output_dir, generations='all', n_bins=50, 
                                 show_uniform=True, cv_range=None, individual='best'):
    """Plot CV histograms for multiple generations showing evolution of sampling.
    
    Creates individual histogram plots for each generation and saves raw data.
    
    Args:
        result: Optimization result dictionary with 'history'
        output_dir: Directory to save plots and data
        generations: 'all' or list of generation indices to plot
        n_bins: Number of bins for histogram
        show_uniform: If True, overlay uniform distribution for reference
        cv_range: Tuple of (min, max) for CV range (for uniform reference)
        individual: Which individual(s) to plot:
            - 'best': Only the best individual from the generation (default)
            - 'all': All individuals combined
            - int: Specific individual index
        
    Returns:
        List of figure objects
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history = result['history']
    
    # Determine which generations to plot
    if generations == 'all':
        gen_indices = range(len(history))
    else:
        gen_indices = generations
    
    figures = []
    
    print("\n" + "="*60)
    print("GENERATING CV HISTOGRAM EVOLUTION")
    if individual == 'best':
        print("  Mode: Best individual only")
    elif individual == 'all':
        print("  Mode: All individuals combined")
    else:
        print(f"  Mode: Individual #{individual}")
    print("="*60)
    
    for gen_idx in gen_indices:
        if gen_idx >= len(history):
            print(f"  ⚠️  Generation {gen_idx} not in history, skipping")
            continue
            
        gen_data = history[gen_idx]
        gen_num = gen_data['generation']
        
        # Find COLVAR file for this generation
        gen_dir = gen_data.get('output_dir')
        if not gen_dir:
            print(f"  ⚠️  Generation {gen_num} missing output_dir, skipping")
            continue
        
        gen_path = Path(gen_dir)
        
        # Determine which individual(s) to use
        if individual == 'best':
            # Find best individual by score
            scores = gen_data.get('all_scores', [])
            if scores:
                best_idx = np.argmin(scores)
            else:
                best_idx = 0
            ind_indices = [best_idx]
            title_suffix = f" - Best Individual (#{best_idx})"
        elif individual == 'all':
            # Use all individuals
            ind_dirs = sorted(gen_path.glob('ind*'))
            ind_indices = list(range(len(ind_dirs)))
            title_suffix = " - All Individuals"
        elif isinstance(individual, int):
            ind_indices = [individual]
            title_suffix = f" - Individual #{individual}"
        else:
            print(f"  ⚠️  Unknown individual selector '{individual}', skipping")
            continue
        
        # Collect CV values from selected individuals
        all_cv_values = []
        
        for ind_idx in ind_indices:
            # Try to find COLVAR files for this individual
            ind_pattern = f'ind{ind_idx:03d}'
            colvar_paths = list(gen_path.glob(f'{ind_pattern}/COLVAR'))
            if not colvar_paths:
                colvar_paths = list(gen_path.glob(f'{ind_pattern}/replica_*/COLVAR'))
            
            # Read COLVAR files
            for colvar_path in colvar_paths:
                try:
                    data = np.loadtxt(colvar_path, skiprows=1)
                    if data.ndim == 1:
                        data = data.reshape(1, -1)
                    cv_values = data[:, 1]  # CV values in second column
                    all_cv_values.extend(cv_values)
                except Exception as e:
                    print(f"  ⚠️  Could not read {colvar_path}: {e}")
                    continue
        
        if not all_cv_values:
            print(f"  ⚠️  Generation {gen_num} has no valid CV data, skipping")
            continue
        
        all_cv_values = np.array(all_cv_values)
        
        # Create histogram plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        counts, bins, patches = ax.hist(all_cv_values, bins=n_bins, alpha=0.7, 
                                        color='steelblue', density=True, 
                                        edgecolor='black', linewidth=0.5)
        
        # Overlay uniform distribution if requested
        if show_uniform and cv_range is not None:
            uniform_height = 1.0 / (cv_range[1] - cv_range[0])
            ax.axhline(uniform_height, color='red', linestyle='--', linewidth=2, 
                      label=f'Uniform ({uniform_height:.4f})', alpha=0.7)
            ax.legend(fontsize=10)
        
        ax.set_xlabel('Collective Variable (rad)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'CV Sampling Distribution - Generation {gen_num}{title_suffix}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add stats text box
        stats_text = f'Samples: {len(all_cv_values)}\n'
        stats_text += f'Mean: {np.mean(all_cv_values):.3f}\n'
        stats_text += f'Std: {np.std(all_cv_values):.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot with appropriate suffix
        if individual == 'best':
            suffix = '_best'
        elif individual == 'all':
            suffix = '_all'
        else:
            suffix = f'_ind{individual:03d}'
        plot_path = output_dir / f"cv_histogram_gen{gen_num:03d}{suffix}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save raw data
        data_path = output_dir / f"cv_histogram_gen{gen_num:03d}{suffix}.npz"
        np.savez(data_path, cv_values=all_cv_values, counts=counts, bins=bins)
        
        figures.append(fig)
        print(f"  ✓ Gen {gen_num:3d}: {len(all_cv_values):6d} samples")
    
    print("="*60)
    print(f"✅ Generated {len(figures)} CV histogram plots")
    print(f"   Saved to: {output_dir}")
    print("="*60)
    
    return figures
