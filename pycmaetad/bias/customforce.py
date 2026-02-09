from .base import CustomForceBias
import numpy as np


class MultiGaussian2DForceBias(CustomForceBias):
    """Multiple 2D Gaussian hills for multi-well flattening.
    
    Sum of N Gaussian biases, each with 6 parameters:
    V(x,y) = sum_i height_i * exp(-0.5 * (r - μ_i)ᵀ Σ_i⁻¹ (r - μ_i))
    
    Total parameters: N × 6 = [height_1, cx_1, cy_1, log_var_x_1, ρ_1, log_var_y_1,
                                 height_2, cx_2, cy_2, log_var_x_2, ρ_2, log_var_y_2,
                                 ...]
    
    Args:
        n_gaussians: Number of Gaussian hills
        height_range: (min, max) for each Gaussian amplitude (kJ/mol)
        center_x_range: (min, max) for x center positions (nm)
        center_y_range: (min, max) for y center positions (nm)
        log_variance_x_range: (min, max) for log(variance_x)
        log_variance_y_range: (min, max) for log(variance_y)
        correlation_range: (min, max) for correlation coefficient ρ
    """
    
    def __init__(
        self,
        n_gaussians: int = 1,
        height_range: tuple[float, float] = (0.0, 200.0),
        center_x_range: tuple[float, float] = (-1.5, 1.5),
        center_y_range: tuple[float, float] = (-0.5, 2.5),
        log_variance_x_range: tuple[float, float] = (np.log(0.01), np.log(1.0)),
        log_variance_y_range: tuple[float, float] = (np.log(0.01), np.log(1.0)),
        correlation_range: tuple[float, float] = (-0.99, 0.99)
    ):
        self.n_gaussians = n_gaussians
        self.height_range = height_range
        self.center_x_range = center_x_range
        self.center_y_range = center_y_range
        self.log_variance_x_range = log_variance_x_range
        self.log_variance_y_range = log_variance_y_range
        self.correlation_range = correlation_range
        
        self._parameters = None
        # Cache for each Gaussian
        self._gaussians = []  # List of parameter dicts
    
    def get_parameter_space_size(self) -> int:
        """6 parameters per Gaussian."""
        return 6 * self.n_gaussians
    
    def set_parameters(self, parameters: np.ndarray) -> None:
        """Set parameters for all Gaussians.
        
        Args:
            parameters: Flattened array of N×6 parameters
        """
        if len(parameters) != self.get_parameter_space_size():
            raise ValueError(
                f"Expected {self.get_parameter_space_size()} parameters "
                f"for {self.n_gaussians} Gaussians, got {len(parameters)}"
            )
        
        self._parameters = parameters.copy()
        self._gaussians = []
        
        # Process each Gaussian
        for i in range(self.n_gaussians):
            idx = i * 6
            height, cx, cy, log_var_x, rho, log_var_y = parameters[idx:idx+6]
            
            # Convert and validate
            var_x = np.exp(log_var_x)
            var_y = np.exp(log_var_y)
            rho = np.clip(rho, -1 + 1e-5, 1 - 1e-5)
            cov_xy = rho * np.sqrt(var_x * var_y)
            
            # Compute inverse covariance
            det = var_x * var_y - cov_xy**2
            if det <= 0:
                raise ValueError(f"Gaussian {i}: non-positive definite covariance")
            
            inv_11 = var_y / det
            inv_12 = -cov_xy / det
            inv_22 = var_x / det
            
            self._gaussians.append({
                'height': height,
                'cx': cx,
                'cy': cy,
                'inv_11': inv_11,
                'inv_12': inv_12,
                'inv_22': inv_22
            })
    
    def get_parameters(self) -> np.ndarray:
        """Get current parameters."""
        if self._parameters is None:
            return self.get_default_parameters()
        return self._parameters.copy()
    
    def get_parameter_bounds(self) -> np.ndarray:
        """Bounds for all N Gaussians."""
        single_bounds = np.array([
            self.height_range,
            self.center_x_range,
            self.center_y_range,
            self.log_variance_x_range,
            self.correlation_range,
            self.log_variance_y_range
        ])
        # Repeat for each Gaussian
        return np.tile(single_bounds, (self.n_gaussians, 1))
    
    def get_default_parameters(self) -> np.ndarray:
        """Default: spread Gaussians across the domain."""
        bounds = self.get_parameter_bounds()
        params = []
        
        for i in range(self.n_gaussians):
            # Default height
            height = (self.height_range[0] + self.height_range[1]) / 2
            
            # Spread centers across domain
            frac = (i + 1) / (self.n_gaussians + 1)
            cx = self.center_x_range[0] + frac * (self.center_x_range[1] - self.center_x_range[0])
            cy = self.center_y_range[0] + frac * (self.center_y_range[1] - self.center_y_range[0])
            
            # Moderate variances
            log_var_x = (self.log_variance_x_range[0] + self.log_variance_x_range[1]) / 2
            log_var_y = (self.log_variance_y_range[0] + self.log_variance_y_range[1]) / 2
            
            # No correlation
            rho = 0.0
            
            params.extend([height, cx, cy, log_var_x, rho, log_var_y])
        
        return np.array(params)
    
    def get_energy_expression(self) -> str:
        """OpenMM energy expression for sum of Gaussians."""
        if self.n_gaussians == 1:
            # Special case for single Gaussian - simpler expression
            return ("height*exp(-0.5*(dx*dx*inv_11 + 2*dx*dy*inv_12 + dy*dy*inv_22)); "
                   "dx=x-cx; dy=y-cy")
        
        terms = []
        for i in range(self.n_gaussians):
            # Each Gaussian gets its own parameters
            terms.append(
                f"h{i}*exp(-0.5*(dx{i}*dx{i}*i11_{i} + "
                f"2*dx{i}*dy{i}*i12_{i} + dy{i}*dy{i}*i22_{i}))"
            )
        
        # Define displacement terms
        displacements = []
        for i in range(self.n_gaussians):
            displacements.append(f"dx{i}=x-cx{i}; dy{i}=y-cy{i}")
        
        # Energy expression MUST come first, then variable definitions
        return " + ".join(terms) + "; " + "; ".join(displacements)
    
    def get_force_parameters(self) -> dict[str, float]:
        """Get force parameters for all Gaussians."""
        if self._parameters is None:
            raise RuntimeError("Must set parameters first")
        
        if self.n_gaussians == 1:
            # Special case for single Gaussian - simpler parameter names
            gauss = self._gaussians[0]
            return {
                'height': gauss['height'],
                'cx': gauss['cx'],
                'cy': gauss['cy'],
                'inv_11': gauss['inv_11'],
                'inv_12': gauss['inv_12'],
                'inv_22': gauss['inv_22']
            }
        
        params = {}
        for i, gauss in enumerate(self._gaussians):
            params[f'h{i}'] = gauss['height']
            params[f'cx{i}'] = gauss['cx']
            params[f'cy{i}'] = gauss['cy']
            params[f'i11_{i}'] = gauss['inv_11']
            params[f'i12_{i}'] = gauss['inv_12']
            params[f'i22_{i}'] = gauss['inv_22']
        
        return params
    
    def get_particle_indices(self) -> list[int]:
        """Acts on particle 0."""
        return [0]
    
    def evaluate_numpy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate sum of Gaussians on a grid (for visualization).
        
        Args:
            x: X coordinates (array or scalar)
            y: Y coordinates (array or scalar)
            
        Returns:
            Sum of all Gaussian biases
        """
        if self._parameters is None:
            raise RuntimeError("Must set parameters first")
        
        V_total = np.zeros_like(x, dtype=float)
        
        for gauss in self._gaussians:
            dx = x - gauss['cx']
            dy = y - gauss['cy']
            
            exponent = -0.5 * (
                dx * dx * gauss['inv_11'] +
                2 * dx * dy * gauss['inv_12'] +
                dy * dy * gauss['inv_22']
            )
            
            V_total += gauss['height'] * np.exp(exponent)
        
        return V_total
    
    def get_all_gaussians(self):
        """Get all individual Gaussians as separate objects.
        
        Returns:
            List of GaussianInfo objects, each with get_parameters() method.
        """
        if self._parameters is None:
            raise RuntimeError("Must set parameters first")
        
        gaussians = []
        for i in range(self.n_gaussians):
            idx = i * 6
            params = self._parameters[idx:idx+6]
            gaussians.append(GaussianInfo(params))
        
        return gaussians
    
    def denormalize_parameters(self, normalized_params: np.ndarray) -> np.ndarray:
        """Convert normalized parameters [0, 1] to physical values.
        
        Args:
            normalized_params: Parameters in [0, 1] range
            
        Returns:
            Physical parameter values
        """
        bounds = self.get_parameter_bounds()
        return bounds[:, 0] + normalized_params * (bounds[:, 1] - bounds[:, 0])


class GaussianInfo:
    """Helper class to wrap individual Gaussian parameters.
    
    Provides a get_parameters() method for compatibility with plotting code.
    """
    
    def __init__(self, parameters: np.ndarray):
        """Initialize with 6-element parameter array."""
        if len(parameters) != 6:
            raise ValueError(f"Expected 6 parameters, got {len(parameters)}")
        self._parameters = parameters.copy()
    
    def get_parameters(self) -> np.ndarray:
        """Return [height, cx, cy, log_var_x, rho, log_var_y]."""
        return self._parameters.copy()


# Backward compatibility alias: single Gaussian is just MultiGaussian2DForceBias with n=1
Gaussian2DForceBias = MultiGaussian2DForceBias


class Harmonic2DForceBias(CustomForceBias):
    """2D harmonic restraint centered at (x0, y0).
    
    V(x,y) = k/2 * ((x-x0)^2 + (y-y0)^2)
    
    Parameters: [k, x0, y0]
    
    Args:
        k_range: (min, max) for spring constant (kJ/mol/nm²).
        x_range: (min, max) for x center position (nm).
        y_range: (min, max) for y center position (nm).
    """
    
    def __init__(
        self,
        k_range: tuple[float, float] = (0.0, 500.0),
        x_range: tuple[float, float] = (-2.0, 2.0),
        y_range: tuple[float, float] = (-2.0, 2.0)
    ):
        self.k_range = k_range
        self.x_range = x_range
        self.y_range = y_range
        
        self._parameters = None
    
    def get_parameter_space_size(self) -> int:
        return 3  # k, x0, y0
    
    def set_parameters(self, parameters: np.ndarray) -> None:
        if len(parameters) != 3:
            raise ValueError(f"Expected 3 parameters, got {len(parameters)}")
        self._parameters = parameters.copy()
    
    def get_parameters(self) -> np.ndarray:
        if self._parameters is None:
            return self.get_default_parameters()
        return self._parameters.copy()
    
    def get_parameter_bounds(self) -> np.ndarray:
        return np.array([
            self.k_range,   # k bounds
            self.x_range,   # x0 bounds
            self.y_range    # y0 bounds
        ])
    
    def get_default_parameters(self) -> np.ndarray:
        """Default: moderate spring constant at center."""
        bounds = self.get_parameter_bounds()
        return np.array([
            (bounds[0, 0] + bounds[0, 1]) / 2,  # k
            (bounds[1, 0] + bounds[1, 1]) / 2,  # x0
            (bounds[2, 0] + bounds[2, 1]) / 2   # y0
        ])
    
    # ===== CustomForceBias interface =====
    
    def get_energy_expression(self) -> str:
        """OpenMM energy expression."""
        return "0.5*k*((x-x0)^2 + (y-y0)^2)"
    
    def get_force_parameters(self) -> dict[str, float]:
        """Get force parameters."""
        if self._parameters is None:
            raise RuntimeError("Must set parameters first")
        
        params = self._parameters
        return {
            "k": params[0],
            "x0": params[1],
            "y0": params[2]
        }
    
    def get_particle_indices(self) -> list[int]:
        """Acts on particle 0."""
        return [0]


class HarmonicBias(CustomForceBias):
    """Simple 2D harmonic restraint bias (legacy name).
    
    V(x,y) = k/2 * ((x-x0)^2 + (y-y0)^2)
    
    Useful for testing and simple restraints.
    
    Parameters: [k, x0, y0]
    """
    
    def __init__(
        self,
        k_range: tuple[float, float] = (0.0, 500.0),
        x_range: tuple[float, float] = (-2.0, 2.0),
        y_range: tuple[float, float] = (-2.0, 2.0)
    ):
        self.k_range = k_range
        self.x_range = x_range
        self.y_range = y_range
        
        self._parameters = None
    
    def get_parameter_space_size(self) -> int:
        return 3  # k, x0, y0
    
    def set_parameters(self, parameters: np.ndarray) -> None:
        if len(parameters) != 3:
            raise ValueError(f"Expected 3 parameters, got {len(parameters)}")
        self._parameters = parameters.copy()
    
    def get_parameters(self) -> np.ndarray:
        if self._parameters is None:
            return self.get_default_parameters()
        return self._parameters.copy()
    
    def get_parameter_bounds(self) -> np.ndarray:
        return np.array([
            self.k_range,   # k bounds
            self.x_range,   # x0 bounds
            self.y_range    # y0 bounds
        ])
    
    def get_default_parameters(self) -> np.ndarray:
        """Default: moderate spring constant at center."""
        bounds = self.get_parameter_bounds()
        return np.array([
            (bounds[0, 0] + bounds[0, 1]) / 2,  # k
            (bounds[1, 0] + bounds[1, 1]) / 2,  # x0
            (bounds[2, 0] + bounds[2, 1]) / 2   # y0
        ])
    
    # ===== CustomForceBias interface =====
    
    def get_energy_expression(self) -> str:
        """OpenMM energy expression."""
        return "0.5*k*((x-x0)^2 + (y-y0)^2)"
    
    def get_force_parameters(self) -> dict[str, float]:
        """Get force parameters."""
        if self._parameters is None:
            raise RuntimeError("Must set parameters first")
        
        params = self._parameters
        return {
            "k": params[0],
            "x0": params[1],
            "y0": params[2]
        }
    
    def get_particle_indices(self) -> list[int]:
        """Acts on particle 0."""
        return [0]
