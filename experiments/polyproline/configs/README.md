# Polyproline Configuration Files

This directory contains configuration files for different experimental setups of the Polyproline CMA-ES optimization.

## Available Configurations

### `config_default.py` (Default)
- **Description**: Standard polyproline optimization parameters
- **Simulation time**: 200 ps per evaluation (50k steps × 4 fs)
- **Hills**: 2 hills per dimension, height 150 kJ/mol, width 1.5 nm
- **Population**: 20 individuals, 2 replicas per evaluation
- **Use case**: Baseline configuration for polyproline conformational sampling using both `pp1.pdb` and `pp2.pdb` as starting structures.

### `config_pp2_only.py` (Single Structure)
- **Description**: Polyproline optimization using only `pp2.pdb` as the starting structure
- **Simulation time**: 200 ps per evaluation (50k steps × 4 fs)
- **Hills**: 2 hills per dimension, height 150 kJ/mol, width 1.6 nm
- **Population**: 20 individuals, 1 replica per evaluation
- **Use case**: Isolates the effect of a single starting structure on optimization and sampling behavior; useful for single-structure convergence and bias analysis

## Usage

Run with a specific config:
```bash
python polyproline_example.py --config configs/config_default.py run
python polyproline_example.py --config configs/config_custom.py both
python polyproline_example.py --config configs/config_pp2_only.py run
```

Default (uses `config_default.py`):
```bash
python polyproline_example.py run
```

## Creating New Configs

To create a new configuration:

1. Copy an existing config file
2. Modify the parameters in the `CONFIG` dictionary
3. Update the `name` and `description` fields
4. Use it with `--config path/to/your_config.py`

## Notes on Provided Configurations

- `config_default.py` uses two starting structures (`pp1.pdb`, `pp2.pdb`) and evaluates each individual with both, providing a baseline for conformational sampling diversity.
- `config_pp2_only.py` restricts sampling to a single starting structure (`pp2.pdb`) and uses only one replica per evaluation, allowing analysis of convergence and bias effects from a single initial conformation.

## Configuration Parameters

Each config file defines a `CONFIG` dictionary with:

**Sampler parameters:**
- `temperature`: Simulation temperature (K)
- `time_step`: Integration timestep (ps)
- `friction`: Langevin friction coefficient (/ps)
- `simulation_steps`: Steps per evaluation
- `report_interval`: Data collection frequency
- `pdb_files`: List of starting PDB structures

**Bias parameters:**
- `hills_per_d`: Number of hills per dimension
- `hills_space`: CV space bounds (min, max)
- `hills_height`: Initial hill height (kJ/mol)
- `hills_width`: Initial hill width (nm)

**Evaluator parameters:**
- `bin_edges`: Histogram bin edges for KL divergence
- `is_2d`: Whether evaluation is 2D (False for 1D)

**Optimizer parameters:**
- `initial_mean`: Initial mean for CMA-ES in [0,1] space
- `sigma`: CMA-ES step size
- `population_size`: Individuals per generation
- `max_generations`: Total generations
- `n_workers`: Parallel workers
- `n_replicas`: MD replicas per evaluation
- `early_stop_patience`: Convergence patience (0=disabled)

**Visualization:**
- `cv_range`: CV range for plotting

**Files:**
- `plumed_template`: PLUMED template file name
- `forcefield_files`: OpenMM forcefield files
