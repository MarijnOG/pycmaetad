"""PLUMED-based bias potentials for 1D and 2D collective variables.

Provides two concrete ``PlumedBias`` subclasses:

- ``PlumedHillBias``  — 1D static Gaussian bias read by PLUMED via a HILLS file.
  CMA-ES optimizes *hills_per_d* Gaussians, each described by a center, height,
  and width.  Parameter vector layout (length = hills_per_d * 3)::

      [center_0, ..., center_N, height_0, ..., height_N, width_0, ..., width_N]

- ``PlumedHillBias2D`` — 2D extension supporting correlated Gaussians via
  PLUMED's multivariate HILLS format.  Each hill has six parameters:
  center_x, center_y, height, width_x, width_y, correlation.

Both classes write a HILLS file and a plumed.dat script (generated from a
Plumed template) and return a ``PlumedForce`` for OpenMM integration.
"""

from .base import Bias, PlumedBias
from string import Template
from typing import Optional, Dict
from pathlib import Path
import numpy as np


class PlumedHillBias(PlumedBias):
    """1D PLUMED hill bias for a single collective variable.

    Represents the bias as a static sum of Gaussian hills passed to PLUMED
    via a HILLS file.  CMA-ES optimizes the positions, heights, and widths of
    ``hills_per_d`` Gaussians within the specified CV space.

    Args:
        plumed_template: Path to a PLUMED template file that defines the CV
            and METAD action (uses ``$hills_file`` / ``$colvar_file`` placeholders).
        hills_per_d: Number of Gaussian hills to place along the CV axis.
        hills_space: ``(min, max)`` bounds for hill center positions.
        hills_height: Maximum allowed hill height (kJ/mol).
        hills_width: Maximum allowed hill width (CV units).
        template_vars: Extra substitution variables for the PLUMED template.
    """

    def __init__(
        self,
        plumed_template: str,
        hills_per_d: int,
        hills_space: tuple[float, float],
        hills_height: float,
        hills_width: float,
        template_vars: Optional[Dict[str, str]] = None
    ):
        super().__init__(plumed_template)
        self.hills_per_d = hills_per_d
        self.hills_space = hills_space
        self.hills_height = hills_height
        self.hills_width = hills_width
        self.template_vars = template_vars if template_vars is not None else {}
    
        # Current hill parameters (set by CMA-ES)
        self._centers: Optional[np.ndarray] = None
        self._widths: Optional[np.ndarray] = None
        self._heights: Optional[np.ndarray] = None

    def get_parameter_space_size(self) -> int:
        """MTD has 3 parameters per hill: center, width, height."""
        return self.hills_per_d * 3
    
    def set_parameters(self, parameters: np.ndarray) -> None:
        """Set from unnormalized parameter vector."""
        if len(parameters) != self.get_parameter_space_size():
            raise ValueError(
                f"Expected {self.get_parameter_space_size()} parameters, "
                f"got {len(parameters)}"
            )
        
        self._centers = parameters[:self.hills_per_d]
        self._heights = parameters[self.hills_per_d:2*self.hills_per_d]
        self._widths = parameters[2*self.hills_per_d:]
    
    def get_parameters(self) -> np.ndarray:
        """Get unnormalized parameter vector."""
        if self._centers is None:
            return self.get_default_parameters()
        return np.concatenate([self._centers, self._heights, self._widths])
    
    def get_default_parameters(self) -> np.ndarray:
        """Evenly-spaced centers, mid-range widths/heights."""
        centers = np.linspace(
            self.hills_space[0], 
            self.hills_space[1], 
            self.hills_per_d,
            endpoint=False
        )
        heights = np.ones(self.hills_per_d) * (self.hills_height / 2)
        widths = np.ones(self.hills_per_d) * (self.hills_width / 2)
        
        return np.concatenate([centers, heights, widths])
    
    def get_parameter_bounds(self) -> np.ndarray:
        """Bounds for unnormalized parameters."""
        bounds = np.zeros((self.get_parameter_space_size(), 2))
        bounds[:self.hills_per_d] = [self.hills_space[0], self.hills_space[1]]
        bounds[self.hills_per_d:2*self.hills_per_d] = [0.0, self.hills_height]
        bounds[2*self.hills_per_d:] = [0.0, self.hills_width]
        return bounds
    
    # CMA-ES convenience (normalized [0,1] interface)
    def set_normalized_parameters(self, normalized: np.ndarray) -> None:
        """Set from CMA-ES [0,1] vector."""
        bounds = self.get_parameter_bounds()
        parameters = bounds[:, 0] + normalized * (bounds[:, 1] - bounds[:, 0])
        self.set_parameters(parameters)
    
    def get_normalized_parameters(self) -> np.ndarray:
        """Get normalized [0,1] vector for CMA-ES."""
        parameters = self.get_parameters()
        bounds = self.get_parameter_bounds()
        return (parameters - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    
    def _get_default_hills_header(self) -> str:
        """Generate standard Plumed HILLS header."""
        # Format min/max values - use 'pi' or '-pi' if values match np.pi
        min_val = "-pi" if abs(self.hills_space[0] + np.pi) < 1e-10 else str(self.hills_space[0])
        max_val = "pi" if abs(self.hills_space[1] - np.pi) < 1e-10 else str(self.hills_space[1])
        
        header = (
            "#! FIELDS time w1 sigma_w1 height biasf\n"
            "#! SET multivariate false\n"
            "#! SET kerneltype stretched-gaussian\n"
            f"#! SET min_w1 {min_val}\n"
            f"#! SET max_w1 {max_val}\n"
        )
        return header
    
    @classmethod
    def from_hills_file(
        cls,
        hills_file: str,
        plumed_template: str,
        hills_space: tuple[float, float],
        template_vars: Optional[Dict[str, str]] = None
    ) -> "PlumedHillBias":
        """
        Construct a PlumedHillBias directly from a PLUMED HILLS file.
        Assumes 1D CV.
        """
        centers = []
        widths = []
        heights = []

        with open(hills_file) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.split()
                centers.append(float(fields[1]))
                widths.append(float(fields[2]))
                heights.append(float(fields[3]))

        centers = np.array(centers)
        widths = np.array(widths)
        heights = np.array(heights)

        bias = cls(
            plumed_template=plumed_template,
            hills_per_d=len(centers),
            hills_space=hills_space,
            hills_height=max(heights),
            hills_width=max(widths),
            template_vars=template_vars,
        )

        # Directly set internal state
        bias._centers = centers
        bias._widths = widths
        bias._heights = heights

        return bias

    def write_hills_file(self) -> str:
        """Write HILLS file from current parameters."""
        if self.output_path is None:
            raise RuntimeError("Must set output_path before writing files")
        
        if self._centers is None:
            raise RuntimeError("Must set parameters before writing HILLS")
                
        hills_path = Path(self.output_path) / "HILLS"
        hills_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(hills_path, 'w') as f:
            f.write(self._get_default_hills_header())
            
            # Write all hills at time=0 (static bias)
            for center, width, height in zip(self._centers, self._widths, self._heights):
                f.write(f"0 {center} {width} {height} 1\n")

        
        return str(hills_path)

    def get_plumed_script(self) -> str:
        """Generate Plumed script from template."""
        if self.output_path is None:
            raise RuntimeError("Must set output_path before generating script")
        
        with open(self.plumed_template, 'r') as f:
            template_content = f.read()
        
        output_dir = Path(self.output_path)
        
        # Use paths relative to current working directory for PLUMED
        # PLUMED runs inside OpenMM and interprets paths relative to Python's cwd
        try:
            rel_output = output_dir.relative_to(Path.cwd())
        except ValueError:
            # If output_dir is not relative to cwd, use absolute path as fallback
            rel_output = output_dir
        
        # Generate file paths (relative if possible, absolute otherwise)
        hills_path = str(rel_output / "HILLS")
        colvar_path = str(rel_output / "COLVAR")
        
        # Use both uppercase and lowercase variants for compatibility
        template_vars = {
            'hills_file': hills_path,
            'colvar_file': colvar_path,
            'HILLS_FILE': hills_path,
            'COLVAR_FILE': colvar_path,
            'STRIDE': '10',  # Default stride
            **self.template_vars
        }
        
        template = Template(template_content)
        return template.substitute(template_vars)
    
    def write_files(self) -> Dict[str, str]:
        """Write both HILLS and plumed.dat files."""
        output_files = super().write_files()  # Writes plumed.dat
        hills_file = self.write_hills_file()
        output_files["hills"] = hills_file
        return output_files
    
    def sum_hills(self, cv_range: tuple[float, float], n_points: int = 500, 
                  periodic: bool = True, mintozero: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Compute free energy surface from bias (like PLUMED's sum_hills).
        
        Evaluates FES = -V_bias, where V_bias is the sum of all Gaussian hills.
        This inverts the bias potential to obtain the free energy surface.
        For periodic CVs (like dihedral angles), accounts for periodic boundary conditions.
        
        Args:
            cv_range: (min, max) range for CV values
            n_points: Number of points to evaluate
            periodic: If True, use periodic boundaries (for dihedral angles)
            mintozero: If True, shift minimum to zero (PLUMED --mintozero)
            
        Returns:
            (cv_values, fes_values): Arrays of CV coordinates and corresponding FES
        """
        if self._centers is None:
            raise RuntimeError("Must set parameters before computing sum_hills")
        
        cv_values = np.linspace(cv_range[0], cv_range[1], n_points)
        bias_values = np.zeros(n_points)
        
        # Sum all Gaussian hills: V_bias(s) = Σ h_i * exp(-0.5 * ((s - c_i) / σ_i)²)
        for center, width, height in zip(self._centers, self._widths, self._heights):
            if periodic:
                # For periodic CVs, use continuous modulo to avoid discontinuities
                delta = cv_values - center
                delta = np.arctan2(np.sin(delta), np.cos(delta))
                bias_values += height * np.exp(-0.5 * (delta / width)**2)
            else:
                bias_values += height * np.exp(-0.5 * ((cv_values - center) / width)**2)
        
        # Invert bias to get FES: FES = -V_bias (matches PLUMED convention)
        fes_values = -bias_values
        
        # Shift minimum to zero if requested (PLUMED --mintozero)
        if mintozero:
            fes_values = fes_values - np.min(fes_values)
        
        return cv_values, fes_values
    
    def save_fes(self, output_path: str, cv_range: tuple[float, float], 
                 n_points: int = 500, mintozero: bool = True) -> str:
        """Save free energy surface to file (mimics PLUMED's sum_hills output).
        
        Args:
            output_path: Path to save FES file
            cv_range: (min, max) range for CV values
            n_points: Number of grid points
            mintozero: If True, shift minimum to zero
            
        Returns:
            Path to saved file
        """
        cv_values, bias_values = self.sum_hills(cv_range, n_points)
        
        # Optionally shift minimum to zero (standard for FES visualization)
        if mintozero:
            bias_values = bias_values - np.min(bias_values)
        
        # Save in PLUMED format (CV value, bias/FES value)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("#! FIELDS cv bias\n")
            for cv, bias in zip(cv_values, bias_values):
                f.write(f"{cv:.8f} {bias:.8f}\n")
        
        return str(output_path)

class PlumedHillBias2D(PlumedBias):
    """2D PLUMED hill bias for two collective variables.
    
    Supports optimization of 2D Gaussian hills placed on a grid in 2D CV space.
    Each hill has 6 parameters: center_x, center_y, height, width_x, width_y, correlation.
    
    Uses PLUMED's multivariate format to support correlated Gaussians with full covariance matrix.
    
    Args:
        plumed_template: Path to PLUMED template file
        hills_per_d: Number of hills per dimension (total hills = hills_per_d^2)
        hills_space: Tuple of (min, max) ranges for each CV: ((min_x, max_x), (min_y, max_y))
        hills_height: Maximum height for hills (kJ/mol)
        hills_width: Maximum width for each dimension [width_x, width_y]
        multivariate: If True, use PLUMED multivariate format (supports correlation)
        template_vars: Additional template variables
    """
    def __init__(
        self,
        plumed_template: str,
        hills_per_d: int,
        hills_space: tuple[tuple[float, float], tuple[float, float]],
        hills_height: float,
        hills_width: list[float],
        multivariate: bool = True,
        template_vars: Optional[Dict[str, str]] = None
    ):
        super().__init__(plumed_template)
        self.hills_per_d = hills_per_d
        self.hills_space = hills_space  # ((min_x, max_x), (min_y, max_y))
        self.hills_height = hills_height
        self.hills_width = hills_width  # [width_x, width_y]
        self.multivariate = multivariate
        self.template_vars = template_vars if template_vars is not None else {}
        
        # Current hill parameters
        self._centers_x: Optional[np.ndarray] = None
        self._centers_y: Optional[np.ndarray] = None
        self._heights: Optional[np.ndarray] = None
        self._widths_x: Optional[np.ndarray] = None
        self._widths_y: Optional[np.ndarray] = None
        self._correlations: Optional[np.ndarray] = None
    
    def get_parameter_space_size(self) -> int:
        """2D MTD has 6 parameters per hill: center_x, center_y, height, width_x, width_y, correlation."""
        n_hills = self.hills_per_d ** 2
        return n_hills * 6
    
    def set_parameters(self, parameters: np.ndarray) -> None:
        """Set from unnormalized parameter vector.
        
        Parameters are organized as: for each hill (in row-major order):
        [center_x, center_y, height, width_x, width_y, correlation]
        """
        expected_size = self.get_parameter_space_size()
        if len(parameters) != expected_size:
            raise ValueError(
                f"Expected {expected_size} parameters, got {len(parameters)}"
            )
        
        n_hills = self.hills_per_d ** 2
        params_reshaped = parameters.reshape(n_hills, 6)
        
        self._centers_x = params_reshaped[:, 0]
        self._centers_y = params_reshaped[:, 1]
        self._heights = params_reshaped[:, 2]
        self._widths_x = params_reshaped[:, 3]
        self._widths_y = params_reshaped[:, 4]
        self._correlations = params_reshaped[:, 5]
    
    def get_parameters(self) -> np.ndarray:
        """Get unnormalized parameter vector."""
        if self._centers_x is None:
            return self.get_default_parameters()
        
        n_hills = self.hills_per_d ** 2
        params = np.zeros((n_hills, 6))
        params[:, 0] = self._centers_x
        params[:, 1] = self._centers_y
        params[:, 2] = self._heights
        params[:, 3] = self._widths_x
        params[:, 4] = self._widths_y
        params[:, 5] = self._correlations
        
        return params.flatten()
    
    def get_default_parameters(self) -> np.ndarray:
        """Grid-based initialization with moderate heights/widths."""
        n_hills = self.hills_per_d ** 2
        params = np.zeros((n_hills, 6))
        
        # Create grid of centers
        x_centers = np.linspace(
            self.hills_space[0][0], self.hills_space[0][1],
            self.hills_per_d, endpoint=False
        )
        y_centers = np.linspace(
            self.hills_space[1][0], self.hills_space[1][1],
            self.hills_per_d, endpoint=False
        )
        
        grid_x, grid_y = np.meshgrid(x_centers, y_centers)
        params[:, 0] = grid_x.flatten()
        params[:, 1] = grid_y.flatten()
        
        # Default heights and widths
        params[:, 2] = self.hills_height / 2
        params[:, 3] = self.hills_width[0] / 2
        params[:, 4] = self.hills_width[1] / 2
        params[:, 5] = 0.0  # No correlation
        
        return params.flatten()
    
    def get_parameter_bounds(self) -> np.ndarray:
        """Bounds for unnormalized parameters."""
        n_hills = self.hills_per_d ** 2
        bounds = np.zeros((n_hills * 6, 2))
        
        for i in range(n_hills):
            idx = i * 6
            bounds[idx:idx+6] = [
                [self.hills_space[0][0], self.hills_space[0][1]],  # center_x
                [self.hills_space[1][0], self.hills_space[1][1]],  # center_y
                [0.0, self.hills_height],  # height
                [0.0, self.hills_width[0]],  # width_x
                [0.0, self.hills_width[1]],  # width_y
                [-1.0, 1.0],  # correlation
            ]
        
        return bounds
    
    def set_normalized_parameters(self, normalized: np.ndarray) -> None:
        """Set from CMA-ES [0,1] vector."""
        bounds = self.get_parameter_bounds()
        parameters = bounds[:, 0] + normalized * (bounds[:, 1] - bounds[:, 0])
        self.set_parameters(parameters)
    
    def get_normalized_parameters(self) -> np.ndarray:
        """Get normalized [0,1] vector for CMA-ES."""
        parameters = self.get_parameters()
        bounds = self.get_parameter_bounds()
        return (parameters - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    
    def _get_default_hills_header(self) -> str:
        """Generate 2D PLUMED HILLS header."""
        # Format min/max values
        def format_val(val):
            if abs(val + np.pi) < 1e-10:
                return "-pi"
            elif abs(val - np.pi) < 1e-10:
                return "pi"
            else:
                return str(val)
        
        if self.multivariate:
            # Multivariate format: supports full covariance matrix
            header = (
                "#! FIELDS time phi psi sigma_phi_phi sigma_psi_phi sigma_psi_psi height biasf\n"
                "#! SET multivariate true\n"
                "#! SET kerneltype stretched-gaussian\n"
                f"#! SET min_phi {format_val(self.hills_space[0][0])}\n"
                f"#! SET max_phi {format_val(self.hills_space[0][1])}\n"
                f"#! SET min_psi {format_val(self.hills_space[1][0])}\n"
                f"#! SET max_psi {format_val(self.hills_space[1][1])}\n"
            )
        else:
            # Non-multivariate format: diagonal covariance only
            header = (
                "#! FIELDS time phi psi sigma_phi sigma_psi height biasf\n"
                "#! SET multivariate false\n"
                "#! SET kerneltype stretched-gaussian\n"
                f"#! SET min_phi {format_val(self.hills_space[0][0])}\n"
                f"#! SET max_phi {format_val(self.hills_space[0][1])}\n"
                f"#! SET min_psi {format_val(self.hills_space[1][0])}\n"
                f"#! SET max_psi {format_val(self.hills_space[1][1])}\n"
            )
        return header
    
    def write_hills_file(self) -> str:
        """Write 2D HILLS file from current parameters."""
        if self.output_path is None:
            raise RuntimeError("Must set output_path before writing files")
        
        if self._centers_x is None:
            raise RuntimeError("Must set parameters before writing HILLS")
        
        hills_path = Path(self.output_path) / "HILLS"
        hills_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(hills_path, 'w') as f:
            f.write(self._get_default_hills_header())
            
            # Write all hills at time=0 (static bias)
            for cx, cy, h, wx, wy, corr in zip(
                self._centers_x, self._centers_y, self._heights,
                self._widths_x, self._widths_y, self._correlations
            ):
                if self.multivariate:
                    # Multivariate format: time cx cy var_xx cov_xy var_yy height biasf
                    # Convert widths and correlation to covariance matrix
                    var_xx = wx * wx  # variance_phi = sigma_phi^2
                    var_yy = wy * wy  # variance_psi = sigma_psi^2
                    cov_xy = corr * wx * wy  # covariance = rho * sigma_phi * sigma_psi
                    f.write(f"0 {cx} {cy} {var_xx} {cov_xy} {var_yy} {h} 1\n")
                else:
                    # Non-multivariate format: time cx cy sigma_x sigma_y height biasf
                    # Correlation is ignored
                    f.write(f"0 {cx} {cy} {wx} {wy} {h} 1\n")
        
        return str(hills_path)
    
    def get_plumed_script(self) -> str:
        """Generate PLUMED script from template."""
        if self.output_path is None:
            raise RuntimeError("Must set output_path before generating script")
        
        with open(self.plumed_template, 'r') as f:
            template_content = f.read()
        
        output_dir = Path(self.output_path)
        
        try:
            rel_output = output_dir.relative_to(Path.cwd())
        except ValueError:
            rel_output = output_dir
        
        hills_path = str(rel_output / "HILLS")
        colvar_path = str(rel_output / "COLVAR")
        
        template_vars = {
            'hills_file': hills_path,
            'colvar_file': colvar_path,
            'HILLS_FILE': hills_path,
            'COLVAR_FILE': colvar_path,
            'STRIDE': '10',
            **self.template_vars
        }
        
        template = Template(template_content)
        return template.substitute(template_vars)
    
    def write_files(self) -> Dict[str, str]:
        """Write both HILLS and plumed.dat files."""
        output_files = super().write_files()
        hills_file = self.write_hills_file()
        output_files["hills"] = hills_file
        return output_files
    
    def compute_bias_landscape(self, n_points: tuple[int, int] = (100, 100), 
                               periodic: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 2D bias potential landscape on a grid.
        
        Args:
            n_points: Number of grid points (nx, ny)
            periodic: If True, use periodic boundaries for torsion angles
            
        Returns:
            (X, Y, V): Meshgrid coordinates and bias values
                X, Y: 2D arrays of coordinates
                V: 2D array of bias potential values (kJ/mol)
        """
        if self._centers_x is None:
            raise RuntimeError("Must set parameters before computing bias landscape")
        
        # Create grid
        x = np.linspace(self.hills_space[0][0], self.hills_space[0][1], n_points[0])
        y = np.linspace(self.hills_space[1][0], self.hills_space[1][1], n_points[1])
        X, Y = np.meshgrid(x, y)  # Use 'xy' indexing (default) for matplotlib compatibility
        
        # Initialize bias potential
        V = np.zeros_like(X)
        
        # Sum all Gaussian hills
        for cx, cy, h, wx, wy, corr in zip(
            self._centers_x, self._centers_y, self._heights,
            self._widths_x, self._widths_y, self._correlations
        ):
            if periodic:
                # Use periodic replication: sum all periodic images
                # This avoids discontinuities from wrapping
                period = 2 * np.pi
                
                # Add original hill + 8 periodic neighbors (3x3 grid)
                for shift_x in [-period, 0, period]:
                    for shift_y in [-period, 0, period]:
                        cx_shifted = cx + shift_x
                        cy_shifted = cy + shift_y
                        
                        dx = X - cx_shifted
                        dy = Y - cy_shifted
                        
                        # Compute Gaussian (no wrapping!)
                        var_x = wx * wx
                        var_y = wy * wy
                        cov_xy = corr * wx * wy
                        
                        det = var_x * var_y - cov_xy * cov_xy
                        if det > 1e-10:
                            inv_xx = var_y / det
                            inv_xy = -cov_xy / det
                            inv_yy = var_x / det
                            mahalanobis = inv_xx * dx * dx + 2 * inv_xy * dx * dy + inv_yy * dy * dy
                            V += h * np.exp(-0.5 * mahalanobis)
                        else:
                            V += h * np.exp(-0.5 * ((dx/wx)**2 + (dy/wy)**2))
            else:
                # Non-periodic: just compute distance directly
                dx = X - cx
                dy = Y - cy
                
                var_x = wx * wx
                var_y = wy * wy
                cov_xy = corr * wx * wy
                
                det = var_x * var_y - cov_xy * cov_xy
                if det > 1e-10:
                    inv_xx = var_y / det
                    inv_xy = -cov_xy / det
                    inv_yy = var_x / det
                    mahalanobis = inv_xx * dx * dx + 2 * inv_xy * dx * dy + inv_yy * dy * dy
                    V += h * np.exp(-0.5 * mahalanobis)
                else:
                    V += h * np.exp(-0.5 * ((dx/wx)**2 + (dy/wy)**2))
        
        return X, Y, V
    
    def sum_hills(self, cv_ranges: tuple[tuple[float, float], tuple[float, float]], 
                  n_points: int = 100, periodic: bool = True, mintozero: bool = False) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Compute 2D free energy surface from bias (like PLUMED's sum_hills).
        
        Evaluates FES = -V_bias, where V_bias is the sum of all Gaussian hills.
        This inverts the bias potential to obtain the free energy surface.
        
        Args:
            cv_ranges: Tuple of (min, max) ranges for each CV: ((min_x, max_x), (min_y, max_y))
            n_points: Number of points to evaluate per dimension
            periodic: If True, use periodic boundaries (for dihedral angles)
            mintozero: If True, shift minimum to zero (PLUMED --mintozero)
            
        Returns:
            (cv_values, fes_values): Tuple of ((x_values, y_values), fes_2d_array)
                cv_values: Tuple of 1D arrays for x and y coordinates
                fes_values: 2D array of FES values on the grid
        """
        if self._centers_x is None:
            raise RuntimeError("Must set parameters before computing sum_hills")
        
        # Create 1D coordinate arrays
        x_values = np.linspace(cv_ranges[0][0], cv_ranges[0][1], n_points)
        y_values = np.linspace(cv_ranges[1][0], cv_ranges[1][1], n_points)
        
        # Compute bias landscape using existing method
        X, Y, V = self.compute_bias_landscape(n_points=(n_points, n_points), periodic=periodic)
        
        # Invert bias to get FES: FES = -V_bias (matches PLUMED convention)
        fes_values = -V
        
        # Shift minimum to zero if requested (PLUMED --mintozero)
        if mintozero:
            fes_values = fes_values - np.min(fes_values)
        
        return (x_values, y_values), fes_values