"""Hybrid evaluator combining KL divergence with coverage metrics."""

import numpy as np
from scipy.special import kl_div
from .base import ColvarEvaluator


class HybridUniformEvaluator(ColvarEvaluator):
    """Hybrid evaluator that combines KL divergence with coverage metrics.
    
    Addresses KL divergence limitation: it doesn't penalize missing bins, only
    unevenness in bins that ARE sampled. This can make sparse but uniform sampling
    score better than dense but slightly uneven sampling.
    
    This evaluator combines:
    1. KL divergence (penalizes unevenness)
    2. Coverage penalty (penalizes unsampled bins)
    3. Optional: Entropy bonus (rewards spread)
    
    Score = α·KL + β·coverage_penalty - γ·entropy
    """
    
    def __init__(
        self,
        bin_edges: np.ndarray,
        is_2d: bool = None,
        kl_weight: float = 1.0,
        coverage_weight: float = 0.5,
        entropy_weight: float = 0.0,
        coverage_target: float = 0.9
    ):
        """
        Args:
            bin_edges: Bin edges for histogramming
                - 1D: array of edges [x0, x1, ..., xn]
                - 2D: tuple of (x_edges, y_edges)
            is_2d: Whether this is 2D (auto-detected if None)
            kl_weight: Weight for KL divergence term (default: 1.0)
            coverage_weight: Weight for coverage penalty (default: 0.5)
            entropy_weight: Weight for entropy bonus (default: 0.0, disabled)
            coverage_target: Target fraction of bins to sample (default: 0.9)
        """
        # Auto-detect if not specified
        if is_2d is None:
            is_2d = isinstance(bin_edges, tuple)
        
        self.is_2d = is_2d
        self.kl_weight = kl_weight
        self.coverage_weight = coverage_weight
        self.entropy_weight = entropy_weight
        self.coverage_target = coverage_target
        
        if self.is_2d:
            if not isinstance(bin_edges, tuple):
                raise ValueError("For 2D, bin_edges must be tuple of (x_edges, y_edges)")
            self.num_bins = (len(bin_edges[0]) - 1) * (len(bin_edges[1]) - 1)
            self.bin_edges = bin_edges
        else:
            if isinstance(bin_edges, tuple):
                raise ValueError("For 1D, bin_edges must be array, not tuple")
            self.num_bins = len(bin_edges) - 1
            self.bin_edges = bin_edges
        
        # Uniform density (matches Alberto's 1/n_bins)
        self.uniform_density = 1.0 / self.num_bins
    
    def evaluate(self, colvar_values: np.ndarray) -> float:
        """Compute hybrid score combining KL divergence and coverage.
        
        Args:
            colvar_values: CV values (1D array or Nx2 array for 2D)
            
        Returns:
            Hybrid score (lower is better)
        """
        # Check for insufficient data
        if len(colvar_values) == 0:
            return 1e6
        
        # Compute histogram
        if self.is_2d and colvar_values.ndim == 2:
            hist, _, _ = np.histogram2d(
                colvar_values[:, 0],
                colvar_values[:, 1],
                bins=self.bin_edges,
                density=True
            )
            hist = hist.flatten()
        else:
            hist, _ = np.histogram(
                colvar_values,
                bins=self.bin_edges,
                density=True
            )
        
        # Validate histogram
        if np.sum(hist) == 0 or not np.isfinite(np.sum(hist)):
            return 1e6
        
        # 1. KL divergence term (penalizes unevenness)
        uniform = np.full(self.num_bins, self.uniform_density)
        kl_divs = kl_div(hist, uniform)
        kl_divergence = np.sum(np.abs(kl_divs))
        
        if not np.isfinite(kl_divergence):
            return 1e6
        
        # 2. Coverage penalty (penalizes unsampled bins)
        # Count bins with non-negligible sampling
        threshold = self.uniform_density * 0.01  # 1% of uniform
        bins_sampled = np.sum(hist > threshold)
        coverage_fraction = bins_sampled / self.num_bins
        
        # Penalty increases as coverage drops below target
        if coverage_fraction < self.coverage_target:
            coverage_penalty = (self.coverage_target - coverage_fraction) ** 2
        else:
            coverage_penalty = 0.0
        
        # 3. Entropy bonus (optional, rewards spread)
        # Normalized entropy: H(P) / H(uniform)
        # Higher entropy = more spread out = better
        epsilon = 1e-10
        hist_prob = hist / (np.sum(hist) + epsilon)
        entropy = -np.sum(hist_prob * np.log(hist_prob + epsilon))
        max_entropy = np.log(self.num_bins)  # Entropy of uniform
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Combine terms
        score = (
            self.kl_weight * kl_divergence +
            self.coverage_weight * coverage_penalty -
            self.entropy_weight * normalized_entropy
        )
        
        return score
    
    def evaluate_from_file(self, colvar_file: str) -> float:
        """Load colvar values from file and evaluate.
        
        Args:
            colvar_file: Path to COLVAR file or PDB trajectory
            
        Returns:
            Hybrid score
        """
        colvar_values = self._load_colvar(colvar_file)
        return self.evaluate(colvar_values)
    
    def _load_colvar(self, file_path: str) -> np.ndarray:
        """Load CV values from COLVAR file."""
        from pathlib import Path
        path = Path(file_path)
        
        if path.suffix == '.pdb':
            raise NotImplementedError("PDB loading not implemented for HybridEvaluator yet")
        
        # Load COLVAR file
        data = np.loadtxt(file_path, comments='#')
        if data.ndim == 1:
            return np.array([data[1]])
        else:
            if self.is_2d:
                # Return columns 1 and 2 (first two CVs)
                return data[:, 1:3]
            else:
                # Return only column 1
                return data[:, 1]
    
    @classmethod
    def from_ranges(
        cls,
        ranges: tuple,
        n_bins: int = 50,
        kl_weight: float = 1.0,
        coverage_weight: float = 0.5,
        entropy_weight: float = 0.0,
        coverage_target: float = 0.9
    ):
        """Convenience constructor from CV ranges.
        
        Args:
            ranges: CV range(s)
                - 1D: (min, max)
                - 2D: ((x_min, x_max), (y_min, y_max))
            n_bins: Number of bins per dimension
            kl_weight: Weight for KL divergence
            coverage_weight: Weight for coverage penalty
            entropy_weight: Weight for entropy bonus
            coverage_target: Target coverage fraction
            
        Returns:
            HybridUniformEvaluator instance
        """
        if isinstance(ranges[0], tuple):
            # 2D
            x_edges = np.linspace(ranges[0][0], ranges[0][1], n_bins + 1)
            y_edges = np.linspace(ranges[1][0], ranges[1][1], n_bins + 1)
            bin_edges = (x_edges, y_edges)
            is_2d = True
        else:
            # 1D
            bin_edges = np.linspace(ranges[0], ranges[1], n_bins + 1)
            is_2d = False
        
        return cls(
            bin_edges,
            is_2d=is_2d,
            kl_weight=kl_weight,
            coverage_weight=coverage_weight,
            entropy_weight=entropy_weight,
            coverage_target=coverage_target
        )
