"""PyCMAETAD: CMA-ES optimization of metadynamics bias parameters."""

__version__ = "0.1.0"

# Import main classes for convenient access
from .bias import PlumedHillBias
# from .evaluator import Evaluator
# from .optimizer import CMAESOptimizer
from .sampler import Sampler

__all__ = [
    "PlumedHillBias",
    "Evaluator", 
    "CMAESWorkflow",
    "Sampler",
]