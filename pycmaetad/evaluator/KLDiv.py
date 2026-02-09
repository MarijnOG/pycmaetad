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
        epsilon = 1e-10
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
        
        # 2D histogram - use density=False and normalize manually
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
        
        sampled_prob = sampled_flat / total_counts
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        sampled_prob = np.clip(sampled_prob, epsilon, 1.0)
        target_prob = np.clip(self.target_distribution, epsilon, 1.0)
        
        # Compute KL divergence: sum(p * log(p/q))
        kl_divergence = np.sum(sampled_prob * np.log(sampled_prob / target_prob))
        
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