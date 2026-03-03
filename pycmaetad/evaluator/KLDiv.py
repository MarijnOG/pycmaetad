"""KL-divergence-based evaluators for collective variable distributions.

All evaluators here compare a sampled CV histogram against a target
distribution using KL divergence and return a scalar score (lower = better)
for minimisation by CMA-ES.

Classes
-------
KLDivEvaluator
    General-purpose KL divergence against an arbitrary target distribution.
UniformKLEvaluator1D
    Specialisation for 1D CVs targeting a uniform distribution.
UniformKLEvaluator2D
    Specialisation for 2D CV spaces (e.g., Muller-Brown x/y coordinates).
UniformKLEvaluator
    Alias kept for backward compatibility.
FESMatchingEvaluator
    Scores how well the bias matches a reference free energy surface.
"""

from .base import ColvarEvaluator
import numpy as np
from scipy.stats import entropy
from scipy.special import kl_div
from pathlib import Path


class KLDivEvaluator(ColvarEvaluator):
    """Evaluator that computes the KL divergence between sampled and target distributions."""
    
    def __init__(self, target_distribution: np.ndarray, bin_edges: np.ndarray):
        """
        Args:
            target_distribution: The target probability distribution (normalized).
            bin_edges: The edges of the bins used for histogramming.
        """
        self.target_distribution = target_distribution / np.sum(target_distribution)
        self.bin_edges = bin_edges
    
    def evaluate(self, colvar_values: np.ndarray) -> float:
        """Compute the KL divergence between sampled and target distributions.
        
        Args:
            colvar_values: Array of collective variable values from sampling.
            
        Returns:
            KL divergence value.
        """
        sampled_hist, _ = np.histogram(colvar_values, bins=self.bin_edges, density=True)
        sampled_distribution = sampled_hist / np.sum(sampled_hist)
        
        # Add a small constant to avoid log(0)
        epsilon = np.finfo(float).tiny
        sampled_distribution += epsilon
        target_distribution = self.target_distribution + epsilon
        
        kl_div = entropy(sampled_distribution, target_distribution)
        return kl_div
    
    def evaluate_from_file(self, colvar_file: str) -> float:
        """Load colvar values from file and evaluate.
        
        Args:
            colvar_file: Path to COLVAR file or PDB trajectory.
            
        Returns:
            KL divergence value.
        """
        colvar_values = self._load_colvar(colvar_file)
        return self.evaluate(colvar_values)
    
    def _load_colvar(self, file_path: str) -> np.ndarray:
        """Load CV values from file.
        
        Supports COLVAR files and PDB trajectories.
        """
        path = Path(file_path)
        
        if path.suffix == '.pdb':
            return self._load_from_pdb(file_path)
        else:
            # Assume COLVAR format
            return self._load_from_colvar(file_path)
    
    def _load_from_colvar(self, colvar_file: str) -> np.ndarray:
        """Load from PLUMED COLVAR file.
        
        Assumes column 0 is time, column 1 is the CV value.
        For multi-column COLVAR files, only loads the first CV (column 1).
        """
        data = np.loadtxt(colvar_file, comments='#')
        if data.ndim == 1:
            # Single row - unlikely but handle it
            return np.array([data[1]])
        else:
            # Return only column 1 (the CV), not all columns after time
            return data[:, 1]
    
    def _load_from_pdb(self, pdb_file: str) -> np.ndarray:
        """Load positions from multi-frame PDB trajectory.
        
        OpenMM's PDBReporter writes multiple MODEL records.
        We need to parse them manually since openmm.app.PDBFile only reads the last one.
        """
        positions = []
        
        with open(pdb_file, 'r') as f:
            current_frame_atoms = []
            
            for line in f:
                if line.startswith('MODEL'):
                    # Start of new frame
                    current_frame_atoms = []
                elif line.startswith('ATOM') or line.startswith('HETATM'):
                    # Parse atom line
                    # PDB format: columns 31-38 (x), 39-46 (y), 47-54 (z)
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    # For 2D, we only need x and y
                    current_frame_atoms.append([x, y])
                elif line.startswith('ENDMDL'):
                    # End of frame - save first atom's position
                    if current_frame_atoms:
                        positions.append(current_frame_atoms[0])  # First atom only
        
        if not positions:
            raise ValueError(f"No trajectory data found in {pdb_file}")
        
        # OpenMM's PDBReporter writes in Angstroms, convert to nm
        # (Our simulation is in nm internally, but PDB format uses Angstroms)
        positions_array = np.array(positions) / 10.0
        
        # For 2D evaluators, return just [x, y] (drop z if present)
        if positions_array.shape[1] >= 2:
            positions_array = positions_array[:, :2]
        
        return positions_array


class UniformKLEvaluator1D(KLDivEvaluator):
    """1D Evaluator for uniform sampling (alanine dipeptide style).
    
    Computes KL divergence using density histograms.
    Used for 1D CV spaces (single dihedral angle, etc.)
    """
    
    def __init__(self, bin_edges: np.ndarray):
        """
        Args:
            bin_edges: 1D array of bin edges [x0, x1, ..., xn]
        """
        if isinstance(bin_edges, tuple):
            raise ValueError("For 1D evaluator, bin_edges must be array, not tuple")
        
        num_bins = len(bin_edges) - 1
        self.bin_edges = bin_edges
        
        # Create uniform density distribution
        # When density=True is used in histogram, values are densities not probabilities
        uniform_distribution = np.full(num_bins, 1.0 / num_bins)
        super().__init__(uniform_distribution, bin_edges)
    
    def evaluate(self, colvar_values: np.ndarray) -> float:
        """Compute KL divergence to uniform distribution.
        
        Args:
            colvar_values: 1D array of CV values
            
        Returns:
            KL divergence value (or large penalty if sampling failed).
        """
        # Check for insufficient data
        if len(colvar_values) == 0:
            return 1e6  # Penalty for no data
        
        # 1D histogram with density
        sampled_hist, _ = np.histogram(
            colvar_values, 
            bins=self.bin_edges, 
            density=True
        )
        
        # Check if histogram is valid
        hist_sum = np.sum(sampled_hist)
        if hist_sum == 0 or not np.isfinite(hist_sum):
            return 1e6
        
        # Use scipy.special.kl_div and sum absolute values (matches Alberto's approach)
        kl_divs = kl_div(sampled_hist, self.target_distribution)
        kl_divergence = np.sum(np.abs(kl_divs))
        
        if not np.isfinite(kl_divergence):
            return 1e6
        
        return kl_divergence


class UniformKLEvaluator2D(KLDivEvaluator):
    """2D Evaluator for uniform sampling (Muller-Brown style).
    
    Computes KL divergence using normalized probability histograms.
    Used for 2D position spaces (x,y coordinates).
    """
    
    def __init__(self, bin_edges: tuple):
        """
        Args:
            bin_edges: Tuple of (x_edges, y_edges) for 2D binning
        """
        if not isinstance(bin_edges, tuple):
            raise ValueError("For 2D evaluator, bin_edges must be tuple of (x_edges, y_edges)")
        
        num_bins = (len(bin_edges[0]) - 1) * (len(bin_edges[1]) - 1)
        self.bin_edges = bin_edges
        
        # Create uniform probability distribution (properly normalized)
        uniform_distribution = np.full(num_bins, 1.0 / num_bins)
        super().__init__(uniform_distribution, bin_edges)
    
    def evaluate(self, colvar_values: np.ndarray) -> float:
        """Compute KL divergence to uniform distribution.
        
        Args:
            colvar_values: Nx2 array of (x, y) positions
            
        Returns:
            KL divergence value (or large penalty if sampling failed).
        """
        # Check for insufficient data
        if len(colvar_values) == 0:
            return 1e6
        
        if colvar_values.ndim != 2 or colvar_values.shape[1] < 2:
            raise ValueError(f"Expected Nx2 array for 2D evaluation, got shape {colvar_values.shape}")
        
        # 2D histogram
        sampled_hist, _, _ = np.histogram2d(
            colvar_values[:, 0], 
            colvar_values[:, 1],
            bins=self.bin_edges,
            density=False  # Get counts, not density
        )
        
        # Normalize to get probabilities (sums to 1.0)
        sampled_flat = sampled_hist.flatten()
        total_counts = np.sum(sampled_flat)
        
        if total_counts == 0 or not np.isfinite(total_counts):
            return 1e6
        
        # Add epsilon to zero bins to avoid log(0)
        # This approach (epsilon before normalization) is mathematically sound
        # and numerically equivalent to scipy's handling of zeros
        epsilon = np.finfo(float).tiny
        sampled_flat_safe = np.where(sampled_flat == 0, epsilon, sampled_flat)
        sampled_prob = sampled_flat_safe / np.sum(sampled_flat_safe)
        
        # Use scipy for robust KL divergence calculation
        kl_divergence = entropy(sampled_prob, self.target_distribution)
        
        if not np.isfinite(kl_divergence):
            return 1e6
        
        return kl_divergence
    
    def _load_from_colvar(self, colvar_file: str) -> np.ndarray:
        """Load from PLUMED COLVAR file for 2D CVs.
        
        Assumes column 0 is time, columns 1 and 2 are the x,y CV values.
        """
        data = np.loadtxt(colvar_file, comments='#')
        if data.ndim == 1:
            # Single row - unlikely but handle it
            return np.array([[data[1], data[2]]])
        else:
            # Return columns 1 and 2 (x and y CVs)
            return data[:, 1:3]
    
    @classmethod
    def from_ranges(cls, ranges: tuple, n_bins: int = 50):
        """Convenience constructor from 2D CV ranges.
        
        Args:
            ranges: ((x_min, x_max), (y_min, y_max))
            n_bins: Number of bins per dimension
            
        Returns:
            UniformKLEvaluator2D instance
        """
        x_edges = np.linspace(ranges[0][0], ranges[0][1], n_bins + 1)
        y_edges = np.linspace(ranges[1][0], ranges[1][1], n_bins + 1)
        return cls((x_edges, y_edges))


# Backward compatibility: UniformKLEvaluator auto-detects 1D vs 2D
class UniformKLEvaluator(KLDivEvaluator):
    """Evaluator that computes KL divergence to a uniform distribution.
    
    Auto-detects 1D vs 2D based on bin_edges format.
    For new code, prefer using UniformKLEvaluator1D or UniformKLEvaluator2D directly.
    """
    
    def __init__(self, bin_edges: np.ndarray, is_2d: bool = None):
        """
        Args:
            bin_edges: Bin edges for histogramming.
                - 1D: array of edges [x0, x1, ..., xn]
                - 2D: tuple of (x_edges, y_edges)
            is_2d: Whether this is 2D (auto-detected if None)
        """
        # Auto-detect if not specified
        if is_2d is None:
            is_2d = isinstance(bin_edges, tuple)
        
        # Delegate to specific evaluator
        if is_2d:
            self._evaluator = UniformKLEvaluator2D(bin_edges)
        else:
            self._evaluator = UniformKLEvaluator1D(bin_edges)
        
        # Copy attributes for compatibility
        self.bin_edges = self._evaluator.bin_edges
        self.target_distribution = self._evaluator.target_distribution
        self.is_2d = is_2d
    
    def evaluate(self, colvar_values: np.ndarray) -> float:
        """Delegate to specific evaluator."""
        return self._evaluator.evaluate(colvar_values)
    
    def evaluate_from_file(self, colvar_file: str) -> float:
        """Delegate to specific evaluator."""
        return self._evaluator.evaluate_from_file(colvar_file)
    
    @classmethod
    def from_ranges(cls, ranges: tuple, n_bins: int = 50):
        """Convenience constructor from CV ranges.
        
        Args:
            ranges: CV range(s)
                - 1D: (min, max)
                - 2D: ((x_min, x_max), (y_min, y_max))
            n_bins: Number of bins per dimension
            
        Returns:
            UniformKLEvaluator instance (wraps 1D or 2D evaluator)
        """
        if isinstance(ranges[0], tuple):
            # 2D
            return cls.__new__(cls)
            wrapped = cls.__new__(cls)
            wrapped._evaluator = UniformKLEvaluator2D.from_ranges(ranges, n_bins)
            wrapped.bin_edges = wrapped._evaluator.bin_edges
            wrapped.target_distribution = wrapped._evaluator.target_distribution
            wrapped.is_2d = True
            return wrapped
        else:
            # 1D
            bin_edges = np.linspace(ranges[0], ranges[1], n_bins + 1)
            return cls(bin_edges, is_2d=False)


class FESMatchingEvaluator(ColvarEvaluator):
    """Evaluator that matches a bias potential to a target FES.
    
    The goal is to find a bias V(x) such that FES(x) + V(x) ≈ constant,
    which makes the biased sampling uniform. This is equivalent to V(x) ≈ -FES(x).
    
    This evaluator does NOT run simulations - it directly compares the bias landscape
    to the target FES on a grid.
    """
    
    @property
    def requires_simulation(self) -> bool:
        """This evaluator does not require MD simulation."""
        return False
    
    def __init__(self, fes_file: str, bias, temperature: float = 300.0):
        """
        Args:
            fes_file: Path to FES file (PLUMED format: phi psi FES ...)
            bias: Bias object to evaluate (will be modified with different params)
            temperature: Temperature in Kelvin (for kT units)
        """
        self.temperature = temperature
        self.kT = 8.314462e-3 * temperature  # kJ/mol (gas constant × T)
        self.bias = bias
        
        # Load FES data
        self.phi_grid, self.psi_grid, self.fes_grid = self._load_fes(fes_file)
        
        # Shift FES so minimum is 0 (for numerical stability)
        self.fes_min = np.min(self.fes_grid)
        self.fes_grid_shifted = self.fes_grid - self.fes_min
        
        # Compute target bias: V_target = -FES (shifted)
        self.target_bias = -self.fes_grid_shifted
        
    def _load_fes(self, fes_file: str) -> tuple:
        """Load FES from PLUMED-style free energy file.
        
        Expected format: phi psi FES [derivatives...]
        
        Returns:
            (phi_grid, psi_grid, fes_grid) as 2D arrays
        """
        # Load data (skip comment lines)
        data = np.loadtxt(fes_file, comments='#')
        
        # Columns: phi, psi, FES, dF/dphi, dF/dpsi
        phi = data[:, 0]
        psi = data[:, 1]
        fes = data[:, 2]
        
        # Determine grid shape (assuming regular grid)
        # Count unique values
        unique_phi = np.unique(phi)
        unique_psi = np.unique(psi)
        
        n_phi = len(unique_phi)
        n_psi = len(unique_psi)
        
        # Reshape to 2D grid
        phi_grid = phi.reshape((n_phi, n_psi))
        psi_grid = psi.reshape((n_phi, n_psi))
        fes_grid = fes.reshape((n_phi, n_psi))
        
        return phi_grid, psi_grid, fes_grid
    
    def evaluate(self, params: np.ndarray) -> float:
        """Evaluate how well the bias matches the target FES.
        
        This method is called by the analytical worker with denormalized parameters.
        
        Args:
            params: Denormalized bias parameters to evaluate
            
        Returns:
            KL divergence between (FES + bias) distribution and uniform
        """
        # Set parameters on bias object
        self.bias.set_parameters(params)
        
        # Compute bias directly on FES grid points using periodic replication
        V_bias = np.zeros_like(self.fes_grid_shifted)
        period = 2 * np.pi
        
        for cx, cy, h, wx, wy, corr in zip(
            self.bias._centers_x, self.bias._centers_y, self.bias._heights,
            self.bias._widths_x, self.bias._widths_y, self.bias._correlations
        ):
            # Add periodic images: original + 8 neighbors (3x3 grid)
            for shift_x in [-period, 0, period]:
                for shift_y in [-period, 0, period]:
                    # Shifted hill center
                    cx_shifted = cx + shift_x
                    cy_shifted = cy + shift_y
                    
                    # Distance (no wrapping!)
                    dx = self.phi_grid - cx_shifted
                    dy = self.psi_grid - cy_shifted
                    
                    # Compute 2D Gaussian with correlation
                    var_x = wx * wx
                    var_y = wy * wy
                    cov_xy = corr * wx * wy
                    
                    det = var_x * var_y - cov_xy * cov_xy
                    if det > 1e-10:
                        inv_xx = var_y / det
                        inv_xy = -cov_xy / det
                        inv_yy = var_x / det
                        
                        mahalanobis = inv_xx * dx * dx + 2 * inv_xy * dx * dy + inv_yy * dy * dy
                        V_bias += h * np.exp(-0.5 * mahalanobis)
                    else:
                        # Fallback to diagonal
                        V_bias += h * np.exp(-0.5 * ((dx/wx)**2 + (dy/wy)**2))
        
        # Combined landscape: FES + bias (should be flat)
        combined = self.fes_grid_shifted + V_bias
        
        # Convert to probability distributions
        # P(x) ∝ exp(-F(x)/kT)
        # For uniform sampling: exp(-(FES + V)/kT) should be constant
        
        # Compute Boltzmann weights
        combined_prob = np.exp(-combined / self.kT)
        combined_prob = combined_prob / np.sum(combined_prob)
        
        # Target: uniform distribution
        n_bins = combined_prob.size
        uniform_prob = np.ones(n_bins) / n_bins
        
        # KL divergence
        kl_div = entropy(combined_prob.flatten(), uniform_prob)
        
        return kl_div