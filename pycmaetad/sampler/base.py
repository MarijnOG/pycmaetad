"""Abstract base classes for molecular dynamics samplers."""

from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pycmaetad.bias.base import Bias


class Sampler(ABC):
    """Base class for all samplers.
    
    A sampler runs molecular dynamics simulations (or equivalent)
    with a given bias potential and extracts collective variable values.
    """
    
    @abstractmethod
    def run(self, output_path: str, bias: 'Bias') -> None:
        """Run sampling with the given bias.
        
        Args:
            output_path: Directory to write output files.
            bias: Bias potential to apply during sampling.
        """
        pass
    
    @abstractmethod
    def get_colvar_values(self, colvar_file: str) -> np.ndarray:
        """Extract collective variable values from output.
        
        Args:
            colvar_file: Path to file containing CV timeseries.
            
        Returns:
            Array of CV values from the trajectory.
        """
        pass


class OpenMMSampler(Sampler):
    """Base class for OpenMM-based samplers.
    
    Provides common OpenMM setup and utilities that all OpenMM samplers share.
    """
    
    def __init__(
        self,
        temperature: float,
        time_step: float,
        friction: float,
        simulation_steps: int,
        report_interval: int = 1000
    ):
        """Initialize OpenMM sampler.
        
        Args:
            temperature: Temperature in Kelvin.
            time_step: Integration timestep in picoseconds.
            friction: Friction coefficient in 1/ps.
            simulation_steps: Total number of MD steps to run.
            report_interval: Frequency for writing trajectory frames.
        """
        # Import OpenMM once at initialization
        try:
            import openmm
            import openmm.app as app
            from openmm import unit
            self.openmm = openmm
            self.app = app
            self.unit = unit
        except ImportError as e:
            raise RuntimeError(
                "OpenMMSampler requires OpenMM to be installed. "
                "Install with: conda install -c conda-forge openmm"
            ) from e
        
        self.temperature = temperature
        self.time_step = time_step
        self.friction = friction
        self.simulation_steps = simulation_steps
        self.report_interval = report_interval
    
    def _setup_integrator(self): # TODO: make integrator configurable
        """Create Langevin integrator with configured parameters.
        
        Returns:
            Configured LangevinMiddleIntegrator.
        """
        from openmm import LangevinMiddleIntegrator
        
        return LangevinMiddleIntegrator(
            self.temperature * self.unit.kelvin,
            self.friction / self.unit.picosecond,
            self.time_step * self.unit.picoseconds
        )
    
    def _create_simulation(self, topology, system, integrator):
        """Create OpenMM Simulation object."""
        simulation = self.app.Simulation(topology, system, integrator)
        return simulation
    
    def _minimize_energy(self, simulation, max_iterations: int = 100):
        """Minimize energy of the system.
        
        Args:
            simulation: OpenMM Simulation object.
            max_iterations: Maximum number of minimization iterations.
        """
        print(f"Minimizing energy (max {max_iterations} iterations)...")
        simulation.minimizeEnergy(maxIterations=max_iterations)
        print("Energy minimization complete.")
    
    def get_colvar_values(self, colvar_file: str) -> np.ndarray:
        """Parse Plumed COLVAR file.
        
        Args:
            colvar_file: Path to COLVAR file.
            
        Returns:
            Array of CV values (second column).
        """
        # Use numpy to read, skip comment lines
        data = np.loadtxt(colvar_file, comments='#')
        
        # Handle both 1D and 2D arrays
        if data.ndim == 1:
            return data[1] if len(data) > 1 else np.array([data[1]])
        else:
            return data[:, 1]  # Second column is CV value
        
    def run(self, output_path: str, bias: 'Bias') -> None:
        """Run OpenMM simulation with given bias.
        
        Args:
            output_path: Directory to write output files.
            bias: Bias potential to apply during sampling.
        """
        raise NotImplementedError(
            "Subclasses must implement the run method."
        )