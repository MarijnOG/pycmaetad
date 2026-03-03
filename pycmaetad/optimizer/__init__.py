"""Optimizer classes for CMA-ES metadynamics workflows.

Currently exposes a single high-level entry point:

- ``CMAESWorkflow``: orchestrates iterative CMA-ES optimisation of bias
  parameters.  Each generation samples *population_size* candidate bias
  potentials in parallel (via ``ProcessPoolExecutor``), evaluates each
  with the configured ``Evaluator``, and feeds the scores back into
  ``SepCMA`` to generate the next generation.  Supports checkpointing,
  resume-from-disk, and both analytical (no-simulation) and
  simulation-based evaluation modes.
"""

from .cmaes import CMAESWorkflow

__all__ = ["CMAESWorkflow"]