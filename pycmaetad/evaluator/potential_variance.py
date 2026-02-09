"""Evaluator based on direct potential variance calculation.

No MD simulation needed - computes variance of (V_true + V_bias) on a grid.
Fast and deterministic evaluation for analytical potentials.
"""

from .base import Evaluator
import numpy as np


class PotentialVarianceEvaluator(Evaluator):
    """Evaluator that computes variance of the biased potential on a grid.
    
    Measures how flat the combined potential (V_true + V_bias) is.
    Lower variance = flatter potential = better bias cancellation.
    
    No MD simulation required - pure analytical evaluation.
    """
    
    @property
    def requires_simulation(self) -> bool:
        """This evaluator works analytically - no simulation needed."""
        return False
    
    def __init__(
        self, 
        potential_func, 
        x_range, 
        y_range, 
        n_grid=100,
        energy_clip=None,
        use_robust_variance=False,
        boltzmann_weight=False,
        temperature=300.0
    ):
        """Initialize potential variance evaluator.
        
        Args:
            potential_func: Function(x, y) -> V_true that computes true potential
            x_range: (x_min, x_max) tuple for evaluation grid
            y_range: (y_min, y_max) tuple for evaluation grid
            n_grid: Number of grid points per dimension
            energy_clip: If set, clip V_total to [-clip, +clip] to reduce
                        impact of extreme barriers (e.g., 200.0 kJ/mol)
            use_robust_variance: Use median absolute deviation instead of variance
                                (more robust to outliers)
            boltzmann_weight: Weight grid points by exp(-V/kT) to focus on
                             thermodynamically accessible regions
            temperature: Temperature in K for Boltzmann weighting (default 300K)
        """
        self.potential_func = potential_func
        self.x_range = x_range
        self.y_range = y_range
        self.n_grid = n_grid
        self.energy_clip = energy_clip
        self.use_robust_variance = use_robust_variance
        self.boltzmann_weight = boltzmann_weight
        self.temperature = temperature
        
        # Pre-compute grid
        self.x = np.linspace(x_range[0], x_range[1], n_grid)
        self.y = np.linspace(y_range[0], y_range[1], n_grid)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Pre-compute true potential on grid
        self.V_true = potential_func(self.X, self.Y)
        
        # Pre-compute Boltzmann weights if requested
        self._boltzmann_weights = None
        if boltzmann_weight:
            kT = 8.314e-3 * temperature  # kJ/mol (R in kJ/mol/K)
            # Shift to prevent numerical issues
            V_shifted = self.V_true - self.V_true.min()
            weights = np.exp(-V_shifted / kT)
            self._boltzmann_weights = weights / weights.sum()  # Normalize
    
    def evaluate(self, bias_params) -> float:
        """Compute variance of biased potential.
        
        Args:
            bias_params: Array of bias parameters. Can be:
                - Single Gaussian: [height, cx, cy, log_var_x, rho, log_var_y]
                - Multi Gaussian: N × 6 parameters flattened
            
        Returns:
            Variance of (V_true + V_bias) on the grid.
            Lower is better (flatter potential).
        """
        # Determine if single or multi-Gaussian based on parameter count
        n_params = len(bias_params)
        
        if n_params == 6:
            # Single Gaussian
            V_bias = self._compute_single_gaussian_bias(bias_params)
        elif n_params % 6 == 0:
            # Multi Gaussian
            n_gaussians = n_params // 6
            V_bias = self._compute_multi_gaussian_bias(bias_params, n_gaussians)
        else:
            raise ValueError(f"Invalid number of parameters: {n_params} (expected multiple of 6)")
        
        # Combined potential
        V_total = self.V_true + V_bias
        
        # Clip extreme values if requested
        if self.energy_clip is not None:
            V_total = np.clip(V_total, -self.energy_clip, self.energy_clip)
        
        # Compute variance metric
        if self.use_robust_variance:
            # Median Absolute Deviation (robust to outliers)
            median = np.median(V_total)
            mad = np.median(np.abs(V_total - median))
            metric = mad ** 2  # Square for consistency with variance units
        elif self.boltzmann_weight:
            # Weighted variance (focus on accessible regions)
            mean = np.sum(self._boltzmann_weights * V_total)
            metric = np.sum(self._boltzmann_weights * (V_total - mean)**2)
        else:
            # Standard variance
            metric = np.var(V_total)
        
        if not np.isfinite(metric):
            return 1e6
        
        return metric
    
    def _compute_single_gaussian_bias(self, params):
        """Compute single Gaussian bias on grid."""
        height, cx, cy, log_var_x, rho, log_var_y = params
        
        # Compute bias on grid
        var_x = np.exp(log_var_x)
        var_y = np.exp(log_var_y)
        cov_xy = rho * np.sqrt(var_x * var_y)
        
        # Compute inverse of covariance matrix
        det = var_x * var_y - cov_xy**2
        
        if det <= 0:
            # Invalid covariance matrix
            return np.zeros_like(self.X)
        
        i11 = var_y / det
        i12 = -cov_xy / det
        i22 = var_x / det
        
        # Compute Gaussian bias
        dx = self.X - cx
        dy = self.Y - cy
        exponent = -0.5 * (dx*dx*i11 + 2*dx*dy*i12 + dy*dy*i22)
        return height * np.exp(exponent)
    
    def _compute_multi_gaussian_bias(self, params, n_gaussians):
        """Compute sum of Gaussian biases on grid."""
        V_bias = np.zeros_like(self.X)
        
        for i in range(n_gaussians):
            idx = i * 6
            gauss_params = params[idx:idx+6]
            V_bias += self._compute_single_gaussian_bias(gauss_params)
        
        return V_bias
    
    def evaluate_from_file(self, trajectory_file: str) -> float:
        """Not used for analytical evaluation.
        
        This evaluator doesn't use trajectory files.
        Included for compatibility with base class interface.
        """
        raise NotImplementedError(
            "PotentialVarianceEvaluator uses analytical evaluation, "
            "not trajectory files. Call evaluate(bias_params) directly."
        )
