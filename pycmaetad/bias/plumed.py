from .base import Bias, PlumedBias
from string import Template
from typing import Optional, Dict
from pathlib import Path
import numpy as np


class PlumedHillBias(PlumedBias):
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

    def write_hills_file(self) -> str:
        """Write HILLS file from current parameters."""
        if self.output_path is None:
            raise RuntimeError("Must set output_path before writing files")
        
        if self._centers is None:
            raise RuntimeError("Must set parameters before writing HILLS")
                
        hills_path = Path(self.output_path) / "HILLS"
        hills_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(hills_path, 'w') as f:
                        # Test for somewhat optimized hills
#             hills_file_test = """#! FIELDS time w1 sigma_w1 height biasf
#                 #! SET multivariate false
#                 #! SET kerneltype stretched-gaussian
#                 #! SET min_w1 -pi
#                 #! SET max_w1 pi
# 0 -3.1391832551092547 0.7326422242925802 51.00861898914101 1
# 0 0.2227831890635663 0.5421032375956504 39.551776464408604 1
# """
#            f.write(hills_file_test)
#             return str(hills_path)

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
                # For periodic CVs, compute minimum distance considering wrapping
                period = cv_range[1] - cv_range[0]
                # Distance considering periodic boundaries
                delta = cv_values - center
                delta = delta - period * np.round(delta / period)
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
