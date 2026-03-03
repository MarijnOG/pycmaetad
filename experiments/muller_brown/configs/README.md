# Muller-Brown Configuration Files

This directory contains configuration files for different experimental setups of the Muller-Brown CMA-ES optimization.

## Available Configurations

### `config_tight.py` (Default)
- **Description**: Restricted search space focused on well regions
- **Center ranges**: X: (-0.9, 1.3), Y: (-0.3, 1.8)
- **Purpose**: Keeps Gaussians near the three Muller-Brown minima, reducing wasted exploration in high-energy regions
- **Use case**: Better convergence when you want to constrain the optimization to physically meaningful regions

### `config_wide.py`
- **Description**: Original wide search space covering full Muller-Brown landscape
- **Center ranges**: X: (-1.5, 1.5), Y: (-0.5, 2.5)
- **Purpose**: Allows full exploration of the potential energy surface
- **Use case**: Comparison baseline, exploring different starting conditions

### `config_left_well.py`
- **Description**: Search space focused on the left well region of the Muller-Brown landscape
- **Purpose**: Targets the optimization towards a specific metastable region
- **Use case**: Single-well isolation experiments, studying convergence in a restricted region

## Usage

Run with a specific config:
```bash
python mb_example.py --config configs/config_tight.py run
python mb_example.py --config configs/config_wide.py both
```

Default (uses `config_tight.py`):
```bash
python mb_example.py run
```

## Creating New Configs

To create a new configuration:

1. Copy an existing config file
2. Modify the parameters in the `CONFIG` dictionary
3. Update the `name` and `description` fields
4. Use it with `--config path/to/your_config.py`

## Configuration Parameters

Each config file defines a `CONFIG` dictionary with:

**Bias parameters:**
- `num_gaussians`: Number of Gaussian hills
- `height_range`: Height bounds (kJ/mol)
- `center_x_range`, `center_y_range`: Position bounds (nm)
- `log_variance_x_range`, `log_variance_y_range`: Variance bounds

**Initial conditions:**
- `well_positions`: Starting positions for Gaussian centers
- `initial_height`: Initial hill height (kJ/mol)
- `initial_sigma`: Initial width (nm)

**Sampler parameters:**
- `temperature`: Simulation temperature (K)
- `time_step`: Integration timestep (ps)
- `friction`: Langevin friction coefficient (/ps)
- `simulation_steps`: Steps per evaluation
- `report_interval`: Data collection frequency

**Evaluator parameters:**
- `evaluation_x_range`, `evaluation_y_range`: Evaluation region
- `n_bins`: Histogram bins per dimension

**Optimizer parameters:**
- `sigma`: CMA-ES step size
- `population_size`: Individuals per generation
- `max_generations`: Total generations
- `n_workers`: Parallel workers
- `n_replicas`: MD replicas per evaluation
- `early_stop_patience`: Convergence patience (0=disabled)

**Visualization:**
- `cv_range`: Plotting range
