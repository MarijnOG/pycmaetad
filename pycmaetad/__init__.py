"""PyCMAETAD: CMA-ES optimisation of metadynamics bias parameters.

Top-level convenience imports — everything else is available under the
sub-packages ``pycmaetad.bias``, ``pycmaetad.evaluator``,
``pycmaetad.optimizer``, ``pycmaetad.sampler``, and ``pycmaetad.potentials``.
"""

__version__ = "0.1.0"

from .bias import PlumedHillBias, PlumedHillBias2D, MultiGaussian2DForceBias
from .evaluator import Evaluator, UniformKLEvaluator1D, UniformKLEvaluator2D
from .optimizer import CMAESWorkflow
from .sampler import Sampler, OpenMMPlumedSampler, MullerBrownSampler
from .potentials import AnalyticalPotential, MullerBrownPotential

__all__ = [
    # Bias potentials
    "PlumedHillBias",
    "PlumedHillBias2D",
    "MultiGaussian2DForceBias",
    # Evaluators
    "Evaluator",
    "UniformKLEvaluator1D",
    "UniformKLEvaluator2D",
    # Optimizer
    "CMAESWorkflow",
    # Samplers
    "Sampler",
    "OpenMMPlumedSampler",
    "MullerBrownSampler",
    # Analytical potentials
    "AnalyticalPotential",
    "MullerBrownPotential",
]