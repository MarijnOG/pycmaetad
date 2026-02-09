"""Base class for analytical 2D potentials.

Provides a clean interface for defining analytical potential energy surfaces
that can be used with PotentialVarianceEvaluator.
"""

from abc import ABC, abstractmethod
import numpy as np


class AnalyticalPotential(ABC):
    """Base class for 2D analytical potentials."""
    
    @abstractmethod
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate potential at given coordinates.
        
        Args:
            x: X coordinates (scalar or array)
            y: Y coordinates (scalar or array)
            
        Returns:
            Potential energy values (kJ/mol)
        """
        pass
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Allow calling potential as a function."""
        return self.evaluate(x, y)
    
    @property
    def name(self) -> str:
        """Potential name for logging/visualization."""
        return self.__class__.__name__


class GaussianPotential(AnalyticalPotential):
    """Single 2D Gaussian well potential.
    
    V(x,y) = -height * exp(-0.5 * ((x-cx)²/σx² + (y-cy)²/σy²))
    """
    
    def __init__(self, height=150.0, center=(0.0, 0.0), sigma=(0.5, 0.5)):
        """Initialize Gaussian potential.
        
        Args:
            height: Depth of the well (kJ/mol)
            center: (x, y) center position (nm)
            sigma: (σx, σy) widths (nm)
        """
        self.height = height
        self.center = center
        self.sigma = sigma
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate Gaussian potential."""
        cx, cy = self.center
        sx, sy = self.sigma
        
        exponent = -0.5 * (((x - cx) / sx)**2 + ((y - cy) / sy)**2)
        return -self.height * np.exp(exponent)
    
    @property
    def name(self) -> str:
        return f"Gaussian(h={self.height}, c={self.center}, σ={self.sigma})"


class MullerBrownPotential(AnalyticalPotential):
    """Muller-Brown potential energy surface.
    
    Classic test potential with multiple minima and saddle points.
    Used extensively in metadynamics literature.
    """
    
    def __init__(self):
        """Initialize Muller-Brown potential with standard parameters."""
        # Standard Muller-Brown parameters
        self.A = np.array([-200, -100, -170, 15])
        self.a = np.array([-1, -1, -6.5, 0.7])
        self.b = np.array([0, 0, 11, 0.6])
        self.c = np.array([-10, -10, -6.5, 0.7])
        self.x0 = np.array([1, 0, -0.5, -1])
        self.y0 = np.array([0, 0.5, 1.5, 1])
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate Muller-Brown potential.
        
        V = sum_i A_i * exp(a_i*(x-x0_i)² + b_i*(x-x0_i)*(y-y0_i) + c_i*(y-y0_i)²)
        """
        V = np.zeros_like(x)
        
        for i in range(4):
            dx = x - self.x0[i]
            dy = y - self.y0[i]
            exponent = self.a[i] * dx**2 + self.b[i] * dx * dy + self.c[i] * dy**2
            V += self.A[i] * np.exp(exponent)
        
        return V
    
    @property
    def name(self) -> str:
        return "Muller-Brown"


class DoubleWellPotential(AnalyticalPotential):
    """Symmetric double-well potential along X axis.
    
    V(x,y) = (x² - 1)² + y²
    
    Has two minima at x = ±1, y = 0.
    """
    
    def __init__(self, barrier_height=4.0, well_separation=2.0):
        """Initialize double well.
        
        Args:
            barrier_height: Height of central barrier (kJ/mol)
            well_separation: Distance between wells (nm)
        """
        self.barrier_height = barrier_height
        self.well_separation = well_separation
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate double-well potential."""
        a = self.well_separation / 2.0
        V_x = self.barrier_height * ((x**2 - a**2) / a**2)**2
        V_y = 0.5 * y**2  # Harmonic in y
        return V_x + V_y
    
    @property
    def name(self) -> str:
        return f"DoubleWell(barrier={self.barrier_height}, sep={self.well_separation})"
