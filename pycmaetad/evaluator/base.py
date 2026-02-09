"""Abstract base classes for sampling evaluators."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class Evaluator(ABC):
    """Base class for all evaluators."""

    @property
    def requires_simulation(self) -> bool:
        """Whether this evaluator requires MD simulation.
        
        Returns:
            True if evaluator needs trajectory files (default),
            False if evaluator can work directly with bias parameters (analytical).
        """
        return True  # Default: most evaluators need simulation

    @abstractmethod
    def evaluate(self) -> float:
        """Evaluate the quality of sampled collective variable values.
            
        Returns:
            A scalar score indicating the quality of the sampling.
        """
        pass

#TODO: maybe make ABC
class ColvarEvaluator(Evaluator):
    """Base class for evaluators that use collective variable values."""

    @abstractmethod
    def evaluate(self, colvar_values: np.ndarray) -> float:
        """Evaluate the quality of sampled collective variable values.
        
        Args:
            colvar_values: Array of collective variable values from sampling.
            
        Returns:
            A scalar score indicating the quality of the sampling.
        """
        pass

# TODO: possibly not needed
class FESEvaluator(Evaluator):
    """Base class for evaluators that use free energy surfaces."""

    @abstractmethod
    def evaluate(self, fes_grid: np.ndarray, fes_values: np.ndarray) -> float:
        """Evaluate the quality of the free energy surface.
        
        Args:
            fes_grid: Grid points where the FES is evaluated.
            fes_values: Free energy values at the grid points.
            
        Returns:
            A scalar score indicating the quality of the FES.
        """
        pass