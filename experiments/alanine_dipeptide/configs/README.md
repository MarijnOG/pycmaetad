# Alanine Dipeptide 2D Configuration Files

This directory contains configuration files for different experimental setups of the alanine dipeptide 2D CMA-ES optimization.

## Available Configurations

### `config_default.py` (Default)
- **Description**: Balanced configuration for 2D Ramachandran sampling
- **Hills**: 2×2 = 4 Gaussians
- **Simulation**: 20,000 steps (40 ps)
- **Bins**: 25×25 = 625 bins
- **Purpose**: Good balance between speed and quality for typical use cases
- **Use case**: Default choice for most optimization runs

### `config_fast.py`
- **Description**: Fast configuration for testing (lower quality sampling)
- **Hills**: 2×2 = 4 Gaussians
- **Simulation**: 10,000 steps (20 ps) - **REDUCED**
- **Bins**: 20×20 = 400 bins - **REDUCED**
- **Purpose**: Quick testing and debugging
- **Use case**: Initial exploration, parameter testing, or when time is limited

### `config_thorough.py`
- **Description**: High-quality configuration with longer simulations
- **Hills**: 2×2 = 4 Gaussians
- **Simulation**: 50,000 steps (100 ps) - **INCREASED**
- **Bins**: 30×30 = 900 bins - **INCREASED**
- **Replicas**: 3 - **INCREASED** (averages over 3 random starting positions)
- **Purpose**: Production-quality results
- **Use case**: Final runs, publications, when quality matters more than speed

### `config_3x3hills.py`
- **Description**: Configuration with 6 Gaussian hills (3 per dimension)
- **Hills**: 3 per CV dimension = 6 Gaussians total - **INCREASED**
- **Simulation**: 30,000 steps (60 ps)
- **Bins**: 25×25 = 625 bins
- **Parameters**: 36 total (6 hills × 6 params)
- **Purpose**: More flexible bias potential with finer-grained control per CV
- **Use case**: When you need more granular control over bias in each dimension

## Usage

Run with a specific config:
```bash
python alanine_dipeptide_2d_example.py --config configs/config_default.py run
python alanine_dipeptide_2d_example.py --config configs/config_fast.py both
python alanine_dipeptide_2d_example.py --config configs/config_thorough.py run
```

Default (uses `config_default.py`):
```bash
python alanine_dipeptide_2d_example.py run
```

Resume from checkpoint:
```bash
python alanine_dipeptide_2d_example.py --config configs/config_default.py resume
```

Generate plots only:
```bash
python alanine_dipeptide_2d_example.py --config configs/config_default.py plot
```

## Creating New Configs

To create a new configuration:

1. Copy an existing config file
2. Modify the parameters in the `CONFIG` dictionary
3. Update the `name` and `description` fields
4. Use it with `--config path/to/your_config.py`

## Configuration Parameters

Each config file defines a `CONFIG` dictionary with:

### Sampler parameters
- `temperature`: Simulation temperature (K)
- `time_step`: Integration timestep (ps)
- `friction`: Langevin friction coefficient (/ps)
- `simulation_steps`: Steps per evaluation
- `report_interval`: Data collection frequency

### Bias parameters (2D)
- `hills_per_d`: Hills per CV dimension (total = hills_per_d × n_cvs = hills_per_d × 2)
- `hills_space`: CV ranges ((phi_min, phi_max), (psi_min, psi_max))
- `hills_height`: Maximum height (kJ/mol)
- `hills_width`: Initial widths [phi_width, psi_width] (radians)
- `min_width`: Minimum width constraint (radians)
- `multivariate`: Enable correlation between phi and psi

### Evaluator parameters
- `n_bins`: Histogram bins per dimension (total = n_bins²)

### Optimizer parameters
- `sigma`: CMA-ES step size
- `population_size`: Individuals per generation
- `max_generations`: Total generations
- `n_workers`: Parallel workers
- `n_replicas`: MD replicas per evaluation (for averaging)
- `early_stop_patience`: Convergence patience (0=disabled)
- `early_stop_threshold`: Minimum improvement threshold

### Visualization
- `cv_range`: Plotting range ((phi_range), (psi_range))

## Performance Considerations

**Fast config** (~10-20 min per generation × 100 gen = **16-33 hours total**):
- Best for: Quick iterations, debugging, parameter exploration
- Trade-off: Noisier evaluations, may not converge as well

**Default config** (~20-30 min per generation × 200 gen = **66-100 hours total**):
- Best for: Standard production runs
- Trade-off: Good balance of quality and time

**Thorough config** (~50-75 min per generation × 300 gen = **250-375 hours total**):
- Best for: Publication-quality results, final optimizations
- Trade-off: Significantly longer runtime

**3 hills per dimension config** (~30-45 min per generation × 300 gen = **150-225 hours total**):
- Best for: When you need more control over individual CV dimensions
- Trade-off: 36 parameters require more generations to converge

Note: Times are rough estimates and depend heavily on hardware (CPU cores, speed).
