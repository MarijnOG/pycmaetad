"""Minimal Muller-Brown CMA-ES test experiment.

This script runs a very short CMA-ES optimization on the 2D Muller-Brown potential using a simple multi-Gaussian bias.
The goal is to verify that the optimization workflow runs end-to-end without errors and produces compatible outputs.

It serves as an easily understadable example of how to set up and run a CMA-ES optimization using the PyCMAETAD library, 
and can be used for testing installations or as a template for more complex experiments. This to contrast with the more
sophisticated other examples, where auxiliary functionality like checkpointing, logging, and advanced visualization may 
make it harder to quickly verify that the core optimization workflow is functioning correctly.

Usage:
    python mb_test.py
"""
from pathlib import Path
import numpy as np
from pycmaetad.sampler import MullerBrownSampler
from pycmaetad.bias import MultiGaussian2DForceBias
from pycmaetad.evaluator import UniformKLEvaluator2D
from pycmaetad.optimizer import CMAESWorkflow
from pycmaetad.visualization import plot_convergence

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "test_output"

# Setup bias (2 Gaussians)
bias = MultiGaussian2DForceBias(
    n_gaussians=2,
    height_range=(0, 1000),
    center_x_range=(-1.5, 1.5),
    center_y_range=(-0.5, 2.5),
    log_variance_x_range=(np.log(0.01**2), np.log(0.5**2)),
    log_variance_y_range=(np.log(0.01**2), np.log(0.5**2))
)

# Setup sampler (short simulation)
sampler = MullerBrownSampler(
    temperature=300.0,
    time_step=0.001,
    friction=1.0,
    simulation_steps=5000,
    report_interval=10,
    initial_position=None,
    cv_range=((-1.5, 1.5), (-0.5, 2.5))
)

# Setup evaluator
evaluator = UniformKLEvaluator2D.from_ranges(
    ranges=((-1.5, 1.5), (-0.5, 2.5)),
    n_bins=25 # 25x25 grid for KL divergence estimation
)

# Initial parameters (seed near well positions)
initial_mean = np.ones(12) * 0.5  # 2 Gaussians × 6 params

# Run optimization
workflow = CMAESWorkflow(
    bias=bias,
    sampler=sampler,
    evaluator=evaluator,
    initial_mean=initial_mean,
    sigma=0.3,
    population_size=8,
    max_generations=10,
    n_workers=4,
    n_replicas=1
)

print(f"\n{'='*60}")
print("RUNNING MINIMAL MULLER-BROWN TEST")
print(f"{'='*60}\n")

result = workflow.optimize(str(OUTPUT_DIR))

if result:
    print(f"\n✅ Test complete!")
    print(f"   Best score: {result['best_score']:.4f}")
    print(f"   Results: {OUTPUT_DIR}")

    print("\n📊 Plotting convergence...")
    plot_convergence(result, OUTPUT_DIR / "convergence.png")
    print(f"   Convergence plot saved to: {OUTPUT_DIR / 'convergence.png'}")
else:
    print("\n❌ Test failed: No result returned from optimization.")