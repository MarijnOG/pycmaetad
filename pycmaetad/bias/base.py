"""Abstract base classes for bias potentials."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openmm import Force


class Bias(ABC):
    """Base class for all bias potentials.
    
    A bias represents a set of parameters that modify the energy landscape
    during molecular dynamics sampling.
    
    The minimal interface requires:
    1. Getting/setting parameters
    2. Knowing the parameter space dimensionality
    3. Providing default parameters
    4. Converting to an OpenMM Force object
    """
    
    @abstractmethod
    def get_parameter_space_size(self) -> int:
        """Get the number of parameters defining this bias.
        
        Returns:
            Dimensionality of the parameter space.
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: np.ndarray) -> None:
        """Set bias parameters.
        
        Args:
            parameters: Parameter vector (unnormalized, in physical units).
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> np.ndarray:
        """Get current bias parameters.
        
        Returns:
            Parameter vector (unnormalized, in physical units).
        """
        pass
    
    def get_parameter_bounds(self) -> Optional[np.ndarray]:
        """Get bounds for each parameter.
        
        Returns:
            Array of shape (n_params, 2) with (min, max) for each parameter,
            or None if parameters are unbounded.
        """
        return None
    
    def get_default_parameters(self) -> np.ndarray:
        """Get sensible default parameters.
        
        Returns:
            Default parameter vector (unnormalized).
        """
        size = self.get_parameter_space_size()
        bounds = self.get_parameter_bounds()
        
        if bounds is not None:
            # Default to midpoint of bounds
            return (bounds[:, 0] + bounds[:, 1]) / 2
        
        return np.zeros(size)
    
    # Optimizer-friendly interface (normalized [0,1])
    def set_normalized_parameters(self, normalized: np.ndarray) -> None:
        """Set parameters from normalized [0,1] vector.
        
        Convenience method for optimizers (CMA-ES, etc.) that work
        in normalized space. By default, this uses linear scaling
        based on parameter bounds. Override in subclasses for custom behavior.
        
        Args:
            normalized: Parameter vector with values in [0,1].
        """
        bounds = self.get_parameter_bounds()
        
        if bounds is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} has no bounds, "
                "cannot use normalized parameters"
            )
        
        # Denormalize: [0,1] -> [min, max]
        parameters = bounds[:, 0] + normalized * (bounds[:, 1] - bounds[:, 0])
        self.set_parameters(parameters)
    
    def get_normalized_parameters(self) -> np.ndarray:
        """Get parameters as normalized [0,1] vector.
        
        Convenience method for optimizers.
        
        Returns:
            Normalized parameter vector with values in [0,1].
        """
        parameters = self.get_parameters()
        bounds = self.get_parameter_bounds()
        
        if bounds is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} has no bounds, "
                "cannot use normalized parameters"
            )
        
        # Normalize: [min, max] -> [0,1]
        return (parameters - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    
    def get_normalized_default_parameters(self) -> np.ndarray:
        """Get default parameters as normalized [0,1] vector.
        
        Returns:
            Default normalized parameter vector.
        """
        default = self.get_default_parameters()
        bounds = self.get_parameter_bounds()
        
        if bounds is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} has no bounds"
            )
        
        return (default - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


class PlumedBias(Bias):
    """Base for Plumed-based bias potentials.
    
    These biases write Plumed configuration files and return a PlumedForce
    for OpenMM integration.
    
    Args:
        plumed_template: Path to Plumed template file (defines CVs, etc.).
    """
    
    def __init__(self, plumed_template: str):
        super().__init__()
        self.plumed_template = plumed_template
        self._output_path: Optional[str] = None
    
    @property
    def output_path(self) -> Optional[str]:
        """Get current output path."""
        return self._output_path
    
    def set_output_path(self, path: str) -> None:
        """Set output directory for bias files.
        
        Args:
            path: Directory path where files should be written.
        """
        self._output_path = path
    
    @abstractmethod
    def get_plumed_script(self) -> str:
        """Generate Plumed script content.
        
        Returns:
            String content for plumed.dat file.
        """
        pass
    
    def write_files(self) -> dict[str, str]:
        """Write plumed.dat script file.
        
        Subclasses can override to write additional files (e.g., HILLS).
        
        Returns:
            Dictionary with at least "script" key pointing to plumed.dat.
        """
        if self.output_path is None:
            raise RuntimeError(
                "Must call set_output_path() before writing files"
            )
        
        from pathlib import Path
        
        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        script_path = output_dir / "plumed.dat"
        with open(script_path, "w") as f:
            f.write(self.get_plumed_script())
        
        return {"script": str(script_path)}
    
    def get_openmm_force(self) -> "Force":
        """Create PlumedForce from written files.
        
        Returns:
            PlumedForce object referencing plumed.dat.
            
        Raises:
            RuntimeError: If files not written yet.
        """
        try:
            from openmmplumed import PlumedForce
        except ImportError:
            raise RuntimeError(
                "PlumedBias requires OpenMM-Plumed plugin. "
                "Install with: conda install -c conda-forge openmm-plumed"
            )
        
        if self.output_path is None:
            raise RuntimeError(
                "Must call set_output_path() and write_files() before get_openmm_force()"
            )
        
        # Write files if not already done
        files = self.write_files()
        plumed_script_path = files["script"]
        
        # Read the script content (PlumedForce expects content, not file path!)
        with open(plumed_script_path, 'r') as f:
            plumed_script_content = f.read()
        
        return PlumedForce(plumed_script_content)


class CustomForceBias(Bias):
    """Base for biases using OpenMM CustomExternalForce.
    
    These biases define an analytical potential energy function that
    OpenMM evaluates directly (no file I/O needed).
    
    Subclasses implement:
    1. get_energy_expression() - OpenMM-compatible math expression
    2. get_force_parameters() - Parameters for the force
    3. get_particle_indices() - Which particles the force acts on
    """
    
    @abstractmethod
    def get_energy_expression(self) -> str:
        """Get OpenMM energy expression string.
        
        Returns:
            String with OpenMM-compatible math expression.
            Example: "k*(x-x0)^2 + k*(y-y0)^2"
        """
        pass
    
    @abstractmethod
    def get_force_parameters(self) -> dict[str, float]:
        """Get current parameter values for the force.
        
        Returns:
            Dictionary mapping parameter names to values.
            Example: {"k": 100.0, "x0": 0.5, "y0": 1.5}
        """
        pass
    
    @abstractmethod
    def get_particle_indices(self) -> list[int]:
        """Get indices of particles this force acts on.
        
        Returns:
            List of particle indices.
            Example: [0] for single particle, [0, 5, 7, 9] for torsion.
        """
        pass
    
    def get_openmm_force(self) -> "Force":
        """Create CustomExternalForce from current parameters.
        
        Returns:
            CustomExternalForce object with current parameters.
        """
        try:
            from openmm import CustomExternalForce
        except ImportError:
            raise RuntimeError(
                "CustomForceBias requires OpenMM. "
                "Install with: conda install -c conda-forge openmm"
            )
        
        # Create force with energy expression
        force = CustomExternalForce(self.get_energy_expression())
        
        # Add global parameters
        for name, value in self.get_force_parameters().items():
            force.addGlobalParameter(name, value)
        
        # Add particles
        for particle_idx in self.get_particle_indices():
            force.addParticle(particle_idx, [])
        
        return force