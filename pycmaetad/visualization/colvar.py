import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _detect_cv_dimensionality(colvar_file):
    """Detect CV dimensionality from COLVAR file header.
    
    Parses the PLUMED COLVAR header to determine if CVs are 1D or 2D.
    
    Args:
        colvar_file: Path to COLVAR file
        
    Returns:
        tuple: (is_2d, n_cvs) where is_2d is bool and n_cvs is the number of CV columns
    """
    try:
        with open(colvar_file, 'r') as f:
            header = f.readline().strip()
            if header.startswith('#! FIELDS'):
                # Parse fields: #! FIELDS time <cv1> [<cv2>] [metad.bias] [...]
                fields = header.split()[2:]  # Skip '#!' and 'FIELDS'
                
                # Count CV fields (exclude 'time' and fields containing 'bias')
                cv_fields = [f for f in fields if f.lower() != 'time' and 'bias' not in f.lower()]
                n_cvs = len(cv_fields)
                
                return (n_cvs == 2, n_cvs)
    except:
        pass
    
    # Fallback: try to detect from data shape
    try:
        data = np.loadtxt(colvar_file, skiprows=1, max_rows=1)
        # Heuristic: if 4+ columns, likely 2D (time, cv1, cv2, bias)
        # if 3 columns, likely 1D (time, cv1, bias)
        if len(data) >= 4:
            return (True, 2)
        else:
            return (False, 1)
    except:
        return (False, 1)  # Default to 1D


def plot_colvar_evolution(result, output_path=None, show=False, individual='best'):
    """Plot the evolution of the collective variable (CV) space occupation over generations.
    
    Automatically detects 1D vs 2D CVs and creates appropriate visualization:
    - 1D: CV value vs generation
    - 2D: 2D scatter plot (CV_x vs CV_y) colored by generation
    
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
    
    # Auto-detect CV dimensionality from first COLVAR file
    is_2d = False
    for h in history:
        gen_dir = h.get('output_dir')
        if gen_dir:
            gen_path = Path(gen_dir)
            colvar_files = list(gen_path.glob('*/COLVAR')) or list(gen_path.glob('*/replica_*/COLVAR'))
            if colvar_files:
                is_2d, n_cvs = _detect_cv_dimensionality(colvar_files[0])
                break
    
    fig, ax = plt.subplots(figsize=(10, 8) if is_2d else (12, 6))

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
                    
                    if is_2d and data.shape[1] >= 3:
                        # Extract both CV columns
                        cv_values = data[:, 1:3]  # columns 1 and 2
                        all_cv_values.append(cv_values)
                    else:
                        # Extract single CV column
                        cv_values = data[:, 1]
                        all_cv_values.extend(cv_values)
                except Exception as e:
                    print(f"Warning: Could not read {colvar_path}: {e}")
                    continue
        
        if not all_cv_values:
            continue
        
        if is_2d:
            # Stack 2D data
            all_cv_values = np.vstack(all_cv_values)
            
            # 2D scatter plot: CV_x vs CV_y
            ax.scatter(all_cv_values[:, 0], all_cv_values[:, 1], 
                      alpha=0.4, s=3, color=f'C{gen_idx % 10}', 
                      label=f'Gen {gen_idx}', rasterized=True)
        else:
            # 1D plot: CV vs generation
            all_cv_values = np.array(all_cv_values)
            gen_coords = np.full_like(all_cv_values, gen_idx)
            ax.scatter(gen_coords, all_cv_values, alpha=0.3, s=1, 
                      color=f'C{gen_idx % 10}', rasterized=True)

    # Configure axes based on dimensionality
    if is_2d:
        ax.set_xlabel('CV X (nm)', fontsize=16)
        ax.set_ylabel('CV Y (nm)', fontsize=16)
        ax.legend(loc='best', markerscale=2, fontsize=14, ncol=2)
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Collective Variable (CV)', fontsize=12)
        ax.set_xlim(-0.5, len(history) - 0.5)
    
    # Update title based on mode
    if individual == 'best':
        title = 'CV Space Exploration - Best Individual Per Generation'
    elif individual == 'all':
        title = 'CV Space Exploration - All Individuals (Population Diversity)'
    else:
        title = f'CV Space Exploration - Individual #{individual}'
    
    if is_2d:
        title = '2D ' + title
    
    ax.set_title(title, fontsize=14, fontweight='bold')
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

def plot_cv_histogram(result, generation=None, output_path=None, show=False, cv_range=None):
    """Plot histogram of collective variable (CV) values from a COLVAR file.
    
    For 2D CVs, creates a 2D histogram heatmap.
    For 1D CVs, creates a standard 1D histogram.
    
    Args:
        result: Optimizer result dictionary with generation output directories
        generation: Generation number to plot (if None, uses last generation)
        output_path: Path to save the plot
        show: Whether to display the plot
        cv_range: Optional CV range for histogram bounds
            - 1D: (min, max) tuple (e.g., (-np.pi, np.pi))
            - 2D: ((x_min, x_max), (y_min, y_max)) tuple
        
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
    
    # Read COLVAR file and detect dimensionality
    is_2d, n_cvs = _detect_cv_dimensionality(colvar_paths[0])
    
    data = np.loadtxt(colvar_paths[0], skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    gen_num = gen_data['generation']
    
    if is_2d:
        # 2D histogram
        cv_x = data[:, 1]
        cv_y = data[:, 2]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Determine range for hist2d
        if cv_range is not None:
            # Check if cv_range is 2D (nested tuples) or 1D (simple tuple)
            if isinstance(cv_range[0], (tuple, list)):
                # 2D range: ((x_min, x_max), (y_min, y_max))
                range_arg = [[cv_range[0][0], cv_range[0][1]], [cv_range[1][0], cv_range[1][1]]]
                xlim = cv_range[0]
                ylim = cv_range[1]
            else:
                # 1D range: (min, max) - apply to both axes
                range_arg = [[cv_range[0], cv_range[1]], [cv_range[0], cv_range[1]]]
                xlim = cv_range
                ylim = cv_range
        else:
            range_arg = None
            xlim = None
            ylim = None
        
        counts, x_edges, y_edges, im = ax.hist2d(cv_x, cv_y, bins=50, 
                                                  range=range_arg,
                                                  density=True, cmap='viridis', cmin=1e-10)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Density', fontsize=11)
        
        # Detect units based on CV range (angular if close to ±π)
        if xlim is not None and np.abs(xlim[0] + np.pi) < 0.1 and np.abs(xlim[1] - np.pi) < 0.1:
            x_unit = ' (rad)'
        else:
            x_unit = ''
        
        if ylim is not None and np.abs(ylim[0] + np.pi) < 0.1 and np.abs(ylim[1] - np.pi) < 0.1:
            y_unit = ' (rad)'
        else:
            y_unit = ''
        
        ax.set_xlabel(f'CV X{x_unit}', fontsize=12)
        ax.set_ylabel(f'CV Y{y_unit}', fontsize=12)
        ax.set_title(f'2D Histogram of CV Values - Generation {gen_num}', 
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        
        # Set axis limits if range provided
        if xlim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
    else:
        # 1D histogram
        cv_values = data[:, 1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determine range for hist
        if cv_range is not None:
            range_arg = cv_range
        else:
            range_arg = None
        
        ax.hist(cv_values, bins=50, range=range_arg, alpha=0.7, color='blue', density=True)
        ax.set_xlabel('Collective Variable (CV)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Histogram of CV Values - Generation {gen_num}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set axis limits if range provided
        if cv_range is not None:
            ax.set_xlim(cv_range)
    
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
    
    For 2D CVs, creates two stacked subplots (X vs time, Y vs time).
    For 1D CVs, creates a single plot (CV vs time).
    
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
        scores = gen_data.get('all_scores', [])
        if scores:
            # Robustly select best individual: handle nan, inf, and both min/max cases
            scores_arr = np.array(scores)
            # Remove invalid scores
            valid_mask = np.isfinite(scores_arr)
            if np.any(valid_mask):
                valid_scores = scores_arr[valid_mask]
                valid_indices = np.where(valid_mask)[0]
                # Try to detect if lower or higher is better
                # If all scores are positive, assume lower is better
                # If all scores are negative, assume higher is better
                # If mixed, default to lower is better
                if np.all(valid_scores >= 0):
                    best_valid_idx = np.argmin(valid_scores)
                elif np.all(valid_scores <= 0):
                    best_valid_idx = np.argmax(valid_scores)
                else:
                    best_valid_idx = np.argmin(valid_scores)
                best_idx = valid_indices[best_valid_idx]
            else:
                best_idx = 0
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

    # Auto-detect if 2D by checking first COLVAR file
    is_2d = False
    test_ind_dir = ind_dirs[individuals_to_plot[0]]
    test_colvar = list(test_ind_dir.glob('COLVAR')) or list(test_ind_dir.glob('replica_*/COLVAR'))
    if test_colvar:
        is_2d, n_cvs = _detect_cv_dimensionality(test_colvar[0])
    
    # Create figure with appropriate subplots
    if is_2d:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        axes = [ax]
    
    # Get colormap for individuals
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(individuals_to_plot))))
    
    # Get colormap for replicas (for when single individual has multiple replicas)
    replica_colors = plt.cm.Set2(np.linspace(0, 1, 8))
    
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
                
                # Choose color based on context
                if individual == 'all':
                    # Multiple individuals: use individual color
                    plot_color = colors[plot_idx]
                    label = f'Ind {ind_idx}' if replica_idx == 0 else None
                    alpha = 0.4
                    marker_size = 1
                elif len(colvar_paths) > 1:
                    # Multiple replicas for single individual: use replica colors
                    plot_color = replica_colors[replica_idx % len(replica_colors)]
                    label = f'Replica {replica_idx}'
                    alpha = 0.6
                    marker_size = 3
                else:
                    # Single replica for single individual
                    plot_color = colors[plot_idx]
                    label = None
                    alpha = 0.7
                    marker_size = 3
                
                if is_2d and data.shape[1] >= 3:
                    # Plot both dimensions
                    cv_x = data[:, 1]
                    cv_y = data[:, 2]
                    
                    axes[0].scatter(time, cv_x, color=plot_color, alpha=alpha, 
                                   s=marker_size, label=label, rasterized=True)
                    axes[1].scatter(time, cv_y, color=plot_color, alpha=alpha, 
                                   s=marker_size, rasterized=True)
                else:
                    # Plot single dimension
                    cv_values = data[:, 1]
                    axes[0].scatter(time, cv_values, color=plot_color, alpha=alpha, 
                                   s=marker_size, label=label, rasterized=True)
            except Exception as e:
                print(f"Warning: Could not read {colvar_path}: {e}")
                continue
    
    # Configure axes
    gen_num = gen_data['generation']
    
    if is_2d:
        # Auto-detect units for labeling (angular if values in ±π range)
        cv_x_sample = data[:, 1] if 'data' in locals() else None
        cv_y_sample = data[:, 2] if 'data' in locals() and data.shape[1] >= 3 else None
        
        # Check if CVs are angular (values near ±π)
        x_unit = ''
        y_unit = ''
        if cv_x_sample is not None:
            if np.abs(np.max(cv_x_sample)) <= np.pi + 0.1 and np.abs(np.min(cv_x_sample)) >= -np.pi - 0.1:
                x_unit = ' (rad)'
        if cv_y_sample is not None:
            if np.abs(np.max(cv_y_sample)) <= np.pi + 0.1 and np.abs(np.min(cv_y_sample)) >= -np.pi - 0.1:
                y_unit = ' (rad)'
        
        axes[0].set_ylabel(f'CV X{x_unit}', fontsize=12)
        axes[1].set_ylabel(f'CV Y{y_unit}', fontsize=12)
        axes[1].set_xlabel('Time', fontsize=12)
        axes[0].set_title(f'2D CV Time Series - Generation {gen_num}{plot_title_suffix}', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        
        # Show legend only on top plot if useful
        if individual == 'all' and len(individuals_to_plot) <= 10:
            axes[0].legend(loc='best', markerscale=3, fontsize=8, ncol=2)
        elif len(colvar_paths) > 1 and individual != 'all':
            axes[0].legend(loc='best', markerscale=3, fontsize=10)
    else:
        # Auto-detect units for 1D
        cv_sample = data[:, 1] if 'data' in locals() else None
        cv_unit = ''
        if cv_sample is not None:
            if np.abs(np.max(cv_sample)) <= np.pi + 0.1 and np.abs(np.min(cv_sample)) >= -np.pi - 0.1:
                cv_unit = ' (rad)'
        
        axes[0].set_xlabel('Time', fontsize=12)
        axes[0].set_ylabel(f'Collective Variable{cv_unit}', fontsize=12)
        axes[0].set_title(f'CV Time Series - Generation {gen_num}{plot_title_suffix}', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Show legend if useful
        if individual == 'all' and len(individuals_to_plot) <= 10:
            axes[0].legend(loc='best', markerscale=3, fontsize=8, ncol=2)
        elif len(colvar_paths) > 1 and individual != 'all':
            axes[0].legend(loc='best', markerscale=3, fontsize=10)
    
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
                                 show_uniform=True, cv_range=None, individual='best', initial_position=None):
    """Plot CV histograms for multiple generations showing evolution of sampling.
    
    Creates individual histogram plots for each generation and saves raw data.
    Automatically handles both 1D and 2D collective variables.
    
    Args:
        result: Optimization result dictionary with 'history'
        output_dir: Directory to save plots and data
        generations: 'all' or list of generation indices to plot
        n_bins: Number of bins for histogram (per dimension for 2D)
        show_uniform: If True, overlay uniform distribution for reference
        cv_range: CV range for uniform reference
            - 1D: (min, max)
            - 2D: ((x_min, x_max), (y_min, y_max))
        individual: Which individual(s) to plot:
            - 'best': Only the best individual from the generation (default)
            - 'all': All individuals combined
            - int: Specific individual index
        initial_position: (x, y) tuple of starting position to mark on 2D plots
        
    Returns:
        List of figure objects
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history = result['history']
    
    # Determine CV dimensionality from cv_range
    is_2d = False
    if cv_range is not None:
        if isinstance(cv_range[0], (tuple, list)):
            is_2d = True
            print(f"  Detected 2D CVs: X={cv_range[0]}, Y={cv_range[1]}")
        else:
            print(f"  Detected 1D CV: {cv_range}")
    
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
        first_cv_position = None  # Track the first CV value as initial position
        
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
                    
                    # For 2D: extract both columns [1] and [2]
                    # For 1D: extract column [1]
                    if is_2d and data.shape[1] >= 3:
                        cv_values = data[:, 1:3]  # X and Y columns
                        # Capture first position from first COLVAR file
                        if first_cv_position is None and len(cv_values) > 0:
                            first_cv_position = cv_values[0].copy()
                        all_cv_values.append(cv_values)
                    else:
                        cv_values = data[:, 1]  # Single CV column
                        if first_cv_position is None and len(cv_values) > 0:
                            first_cv_position = cv_values[0]
                        all_cv_values.extend(cv_values)
                except Exception as e:
                    print(f"  ⚠️  Could not read {colvar_path}: {e}")
                    continue
        
        if not all_cv_values:
            print(f"  ⚠️  Generation {gen_num} has no valid CV data, skipping")
            continue
        
        if is_2d:
            all_cv_values = np.vstack(all_cv_values) if len(all_cv_values) > 0 else np.array([])
        else:
            all_cv_values = np.array(all_cv_values)
        
        if len(all_cv_values) == 0:
            print(f"  ⚠️  Generation {gen_num} has no valid CV data, skipping")
            continue
        
        # Create histogram plot - different for 1D vs 2D
        if is_2d:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 2D histogram
            x_vals = all_cv_values[:, 0]
            y_vals = all_cv_values[:, 1]
            
            # Define bins
            if cv_range is not None:
                x_bins = np.linspace(cv_range[0][0], cv_range[0][1], n_bins + 1)
                y_bins = np.linspace(cv_range[1][0], cv_range[1][1], n_bins + 1)
                range_arg = [[cv_range[0][0], cv_range[0][1]], [cv_range[1][0], cv_range[1][1]]]
            else:
                x_bins = n_bins
                y_bins = n_bins
                range_arg = None
            
            counts, x_edges, y_edges, im = ax.hist2d(x_vals, y_vals, bins=[x_bins, y_bins],
                                                      range=range_arg, density=True, 
                                                      cmap='viridis', cmin=1e-10)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Density', fontsize=11)
            
            # Add uniform density reference line on colorbar
            if show_uniform and cv_range is not None:
                area = (cv_range[0][1] - cv_range[0][0]) * (cv_range[1][1] - cv_range[1][0])
                uniform_density = 1.0 / area
                cbar.ax.axhline(uniform_density, color='red', linestyle='--', linewidth=2)
                cbar.ax.text(1.05, uniform_density, f'Uniform\n{uniform_density:.4f}', 
                           va='center', fontsize=8, color='red')
            
            # Mark starting cell using the first CV value from this generation
            if first_cv_position is not None and is_2d:
                from matplotlib.patches import Rectangle
                
                start_x, start_y = first_cv_position
                
                # Find which bin the starting position belongs to using the actual edges
                # x_edges and y_edges are returned by hist2d
                bin_x_idx = np.searchsorted(x_edges, start_x) - 1
                bin_y_idx = np.searchsorted(y_edges, start_y) - 1
                
                # Ensure indices are within bounds
                bin_x_idx = np.clip(bin_x_idx, 0, len(x_edges) - 2)
                bin_y_idx = np.clip(bin_y_idx, 0, len(y_edges) - 2)
                
                # Get bin boundaries from the actual edges
                start_bin_x = x_edges[bin_x_idx]
                start_bin_y = y_edges[bin_y_idx]
                bin_width_x = x_edges[bin_x_idx + 1] - x_edges[bin_x_idx]
                bin_width_y = y_edges[bin_y_idx + 1] - y_edges[bin_y_idx]
                
                # Draw rectangle around starting cell
                rect = Rectangle((start_bin_x, start_bin_y), bin_width_x, bin_width_y,
                                linewidth=2.5, edgecolor='lime', facecolor='none',
                                zorder=10)
                ax.add_patch(rect)
                
                # Mark center with a star
                ax.scatter(start_x, start_y, c='lime', s=150, marker='*',
                          edgecolors='black', linewidths=1.5, zorder=11)
            
            ax.set_xlabel('CV X (nm)', fontsize=12)
            ax.set_ylabel('CV Y (nm)', fontsize=12)
            ax.set_title(f'2D CV Sampling Distribution - Generation {gen_num}{title_suffix}', 
                        fontsize=14, fontweight='bold')
            
            # Add stats text box
            stats_text = f'Samples: {len(all_cv_values)}\n'
            stats_text += f'X: {np.mean(x_vals):.3f} ± {np.std(x_vals):.3f}\n'
            stats_text += f'Y: {np.mean(y_vals):.3f} ± {np.std(y_vals):.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        else:
            # 1D histogram
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
        if is_2d:
            np.savez(data_path, cv_values=all_cv_values, counts=counts, 
                    x_edges=x_edges, y_edges=y_edges)
        else:
            np.savez(data_path, cv_values=all_cv_values, counts=counts, bins=bins)
        
        figures.append(fig)
        print(f"  ✓ Gen {gen_num:3d}: {len(all_cv_values):6d} samples")
    
    print("="*60)
    print(f"✅ Generated {len(figures)} CV histogram plots")
    print(f"   Saved to: {output_dir}")
    print("="*60)
    
    return figures