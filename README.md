# PyCMAETAD — CMA-ES Optimised Metadynamics

## Overview

**PyCMAETAD** optimises static metadynamics bias potentials for molecular dynamics
simulations using the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm.

Rather than depositing Gaussian hills adaptively during a simulation, the bias is treated
as a **static, parameterised object** whose hill positions, heights, and widths are
optimised by CMA-ES.  Each generation of candidates is evaluated in parallel; the score
for each candidate measures how uniformly the biased simulation samples collective
variable (CV) space (e.g. KL divergence from a uniform distribution).

### Optimisation loop

1. **Propose** — CMA-ES samples a population of bias parameter vectors
2. **Simulate** — each candidate runs an MD simulation (OpenMM) with that bias applied
3. **Evaluate** — a scoring function measures sampling quality (lower = better)
4. **Update** — scores are fed back to CMA-ES to guide the next generation
5. **Repeat** until convergence or a generation budget is exhausted


## Repository layout

```
pycmaetad/          Core library
  bias/             Bias potential classes (PlumedHillBias, MultiGaussian2DForceBias, …)
  evaluator/        Scoring functions (KL divergence, trajectory length, …)
  optimizer/        CMAESWorkflow — the main optimisation driver
  sampler/          MD wrappers (OpenMMPlumedSampler, MullerBrownSampler)
  potentials.py     Analytical 2D potentials (Muller-Brown, double-well, …)
  visualization/    Convergence and bias-landscape plots

experiments/
  minimal_example/  Quick end-to-end smoke test (start here)
  muller_brown/     2D Muller-Brown analytical potential
  alanine_dipeptide/  1D backbone dihedral of alanine dipeptide
  polyproline/      Polyproline II helix backbone CVs
  sampling_convergence/  Convergence diagnostics
  evaluator_comparison/  Evaluator benchmarks
```


## Installation

```bash
# Clone the repository
git clone <repository-url>
cd python_cma_metadynamics

# Create and activate the conda environment
conda env create -f environment.yml
conda activate pycmaetad
```

The environment installs OpenMM, the OpenMM-PLUMED plugin, PLUMED, NumPy, SciPy,
Matplotlib, and the `cma` Python package, then installs `pycmaetad` itself in
editable mode (`pip install -e .`).


## Verifying the installation

Run the minimal end-to-end test, which optimises a 2-Gaussian bias on the 2D
Muller-Brown potential for 10 generations (takes roughly 1–2 minutes):

```bash
cd experiments/minimal_example
python mb_test.py
```

A successful run prints a best score and writes a convergence plot to
`experiments/minimal_example/test_output/convergence.png`.  No PLUMED files or
external force-field parameters are required.


## Usage example

```python
import numpy as np
from pycmaetad.bias import MultiGaussian2DForceBias
from pycmaetad.sampler import MullerBrownSampler
from pycmaetad.evaluator import UniformKLEvaluator2D
from pycmaetad.optimizer import CMAESWorkflow

bias = MultiGaussian2DForceBias(
    n_gaussians=2,
    height_range=(0, 1000),
    center_x_range=(-1.5, 1.5),
    center_y_range=(-0.5, 2.5),
    log_variance_x_range=(np.log(0.01**2), np.log(0.5**2)),
    log_variance_y_range=(np.log(0.01**2), np.log(0.5**2)),
)

sampler = MullerBrownSampler(
    temperature=300.0,
    time_step=0.001,
    friction=1.0,
    simulation_steps=10000,
    report_interval=10,
)

evaluator = UniformKLEvaluator2D.from_ranges(
    ranges=((-1.5, 1.5), (-0.5, 2.5)),
    n_bins=25,
)

workflow = CMAESWorkflow(
    bias=bias,
    sampler=sampler,
    evaluator=evaluator,
    initial_mean=np.ones(12) * 0.5,
    sigma=0.3,
    population_size=8,
    max_generations=50,
    n_workers=4,
)

result = workflow.optimize("output/")
```

For PLUMED-based systems (e.g. alanine dipeptide) replace `MultiGaussian2DForceBias`
with `PlumedHillBias` / `PlumedHillBias2D` and `MullerBrownSampler` with
`OpenMMPlumedSampler`.  Refer to the experiment scripts for complete working examples.


## Experiments

| Folder | System | Bias type |
|--------|--------|-----------|
| `minimal_example/` | Muller-Brown (2D) | `MultiGaussian2DForceBias` |
| `muller_brown/` | Muller-Brown (2D) | `MultiGaussian2DForceBias` |
| `alanine_dipeptide/` | Alanine dipeptide φ/ψ | `PlumedHillBias2D` |
| `polyproline/` | Polyproline backbone CVs | `PlumedHillBias` / `PlumedHillBias2D` |
| `sampling_convergence/` | Convergence diagnostics | — |
| `evaluator_comparison/` | Evaluator benchmarks | — |

Each experiment folder contains its own `README.md` and `configs/` directory with
ready-to-use configuration files.


