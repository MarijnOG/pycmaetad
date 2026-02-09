# PyCMAETAD - CMA-ES Optimized Metadynamics for Enhanced Sampling

## Overview

**PyCMAETAD** implements a novel approach to enhanced sampling in molecular dynamics simulations by optimizing static metadynamics bias potentials using the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm.

Instead of traditional metadynamics that gradually deposits Gaussian hills during simulation, we treat the bias potential as a **static, optimizable object**. The CMA-ES algorithm searches for hill parameters (centers, widths, heights) that produce the most uniform sampling of the collective variable (CV) space.

### Key Concept: Pseudo-Metadynamics

Traditional metadynamics:
- Hills deposited online during simulation
- Adaptive but can overflood or leave gaps
- Limited control over final bias shape

Our approach:
1. **Define**: A parameterized set of Gaussian hills
2. **Simulate**: Run MD with static bias potential
3. **Evaluate**: Measure sampling uniformity (e.g., KL divergence from uniform distribution)
4. **Optimize**: Use CMA-ES to adjust hill parameters
5. **Iterate**: Repeat until optimal uniform sampling achieved

## Features

- 🎯 **CMA-ES Optimization**: Adaptive parameter search for bias potentials
- 🔬 **OpenMM Integration**: Fast MD simulations with PLUMED bias support
- 📊 **Multiple Evaluators**:
  - KL divergence (uniform sampling)
  - Potential variance (landscape flattening)
  - Trajectory-based metrics
- 🔄 **Parallel Execution**: Multi-worker support for population-based optimization
- 💾 **Checkpoint/Resume**: Save progress and resume interrupted runs
- 📈 **Visualization Tools**: Bias landscapes, convergence plots, CV distributions
- 🧪 **Analytical Potentials**: Test on Muller-Brown, double-well, custom 2D surfaces

## Installation

### Prerequisites

- Python 3.8+
- OpenMM 8.0+
- PLUMED 2.8+ (for MD bias integration)
- OpenMM-PLUMED plugin

### Setup

```bash
# Clone repository
git clone <repository-url>
cd python_cma_metadynamics

# Create conda environment
conda env create -f environment.yaml
conda activate pycmaetad

# Install package
pip install -e .
```

### Verify Installation

```bash
# Run a quick test
python test_file_generation.py

# Run minimal optimization (2 generations)
cd examples/dipeptide
python alanine_dipeptide_example.py run --generations 2
```

## Quick Start

### Basic Workflow

```python
from pycmaetad.bias import PlumedHillBias
from pycmaetad.sampler import OpenMMPlumedSampler
from pycmaetad.evaluator import UniformKLEvaluator
from pycmaetad.optimizer import CMAESOptimizer

# Define bias potential
bias = PlumedHillBias(
    plumed_template="plumed_template.dat",
    hills_per_d=2,
    hills_space=(-3.14, 3.14),
    hills_height=70.0,
    hills_width=1.5
)

# Configure MD sampler
sampler = OpenMMPlumedSampler(
    pdb_file="system.pdb",
    forcefield_files=["amber14-all.xml"],
    temperature=300.0,
    time_step=0.004,
    friction=1.0,
    simulation_steps=25000
)

# Set up evaluator
evaluator = UniformKLEvaluator(
    bin_edges=np.linspace(-np.pi, np.pi, 31),
    is_2d=False
)

# Run optimization
optimizer = CMAESOptimizer(
    bias=bias,
    sampler=sampler,
    evaluator=evaluator,
    population_size=10,
    max_generations=50,
    n_workers=4
)

result = optimizer.optimize(output_dir="output")
```

## Examples

### Alanine Dipeptide

See [`examples/dipeptide/alanine_dipeptide_example.py`](examples/dipeptide/alanine_dipeptide_example.py):

```bash
# Run full optimization
python alanine_dipeptide_example.py run

# Generate plots from results
python alanine_dipeptide_example.py plot

# Resume from checkpoint
python alanine_dipeptide_example.py resume
```

