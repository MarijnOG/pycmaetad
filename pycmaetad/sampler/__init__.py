"""Sampler classes for running biased MD simulations.

Provides the bridge between CMA-ES parameter candidates and actual
molecular dynamics trajectories:

- ``Sampler`` — abstract base defining the ``run`` / ``get_colvar_values``
  interface that all samplers must implement.
- ``OpenMMLangevinSampler`` — shared OpenMM setup (Langevin integrator,
  energy minimisation, COLVAR parsing) inherited by concrete samplers.
- ``OpenMMPlumedSampler`` — full MD sampler that accepts any ``Bias``
  subclass (both PLUMED-based and ``CustomExternalForce``-based), supports
  multiple PDB starting structures, and writes per-replica COLVAR files.
- ``MullerBrownSampler`` — lightweight sampler tailored to the analytical
  Muller-Brown potential; uses a ``CustomExternalForce`` directly and
  writes CV positions via ``ColvarReporter``.
"""

from .base import Sampler, OpenMMLangevinSampler
from .openmmplumed import OpenMMPlumedSampler, MullerBrownSampler

__all__ = [
    "Sampler",
    "OpenMMLangevinSampler",
    "OpenMMPlumedSampler",
    "MullerBrownSampler",
]