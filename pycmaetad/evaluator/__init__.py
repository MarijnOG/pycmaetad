"""Evaluator classes for scoring CMA-ES bias parameter candidates.

Provides several families of evaluators, each computing a scalar score
(lower = better) for a proposed bias potential:

- ``KLDivEvaluator`` / ``UniformKLEvaluator1D`` / ``UniformKLEvaluator2D``:
  measure how close the sampled collective variable distribution is to
  uniform, using KL divergence.
- ``FESMatchingEvaluator``: scores how well the bias matches a target FES.
- ``HybridUniformEvaluator``: combines KL divergence with a coverage penalty
  so that sparsely-sampled regions are also penalised.
- ``TrajectoryLengthEvaluator``: rewards long, diffusive trajectories
  (no histogram needed).
- ``PotentialVarianceEvaluator``: purely analytical — measures the variance
  of (V_true + V_bias) on a grid, requiring no MD simulation at all.

All concrete evaluators inherit from ``Evaluator`` or ``ColvarEvaluator``.
"""

from .base import Evaluator, ColvarEvaluator
from .KLDiv import (
    KLDivEvaluator,
    UniformKLEvaluator,
    UniformKLEvaluator1D,
    UniformKLEvaluator2D,
    FESMatchingEvaluator
)
from .trajectory_length import TrajectoryLengthEvaluator
from .hybrid_uniform import HybridUniformEvaluator
from .potential_variance import PotentialVarianceEvaluator

__all__ = [
    # Abstract base classes
    "Evaluator",
    "ColvarEvaluator",
    # KL-divergence-based evaluators
    "KLDivEvaluator",
    "UniformKLEvaluator",
    "UniformKLEvaluator1D",
    "UniformKLEvaluator2D",
    "FESMatchingEvaluator",
    # Trajectory-based evaluators
    "TrajectoryLengthEvaluator",
    # Hybrid / combined evaluators
    "HybridUniformEvaluator",
    # Analytical evaluators (no simulation required)
    "PotentialVarianceEvaluator",
]