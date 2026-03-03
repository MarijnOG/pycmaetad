"""Bias potential classes for CMA-ES metadynamics.

Provides three families of bias potentials:
- PlumedBias (via PlumedHillBias / PlumedHillBias2D): write HILLS files consumed
  by the PLUMED OpenMM plugin.
- CustomForceBias (via MultiGaussian2DForceBias): analytical Gaussian potentials
  evaluated directly by OpenMM's CustomExternalForce — no file I/O needed.
- Harmonic restraints (Harmonic2DForceBias / HarmonicBias) for testing.

All concrete classes inherit from ``Bias`` and expose a normalized [0,1]
parameter interface that CMA-ES optimizers consume.
"""

from .base import Bias, PlumedBias, CustomForceBias
from .plumed import PlumedHillBias, PlumedHillBias2D
from .customforce import MultiGaussian2DForceBias, Harmonic2DForceBias, HarmonicBias

__all__ = [
    # Abstract base classes
    "Bias",
    "PlumedBias",
    "CustomForceBias",
    # PLUMED-based biases
    "PlumedHillBias",
    "PlumedHillBias2D",
    # OpenMM CustomExternalForce-based biases
    "MultiGaussian2DForceBias",
    "Harmonic2DForceBias",
    "HarmonicBias",
]