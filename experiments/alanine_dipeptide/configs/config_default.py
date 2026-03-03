"""Default configuration for alanine dipeptide 2D CMA-ES optimization.

Balanced configuration with moderate computational cost and good sampling quality.
"""

import numpy as np

CONFIG = {
    "name": "default",
    "description": "Balanced configuration for 2D Ramachandran sampling",
    
    # Sampler parameters
    "temperature": 300.0,  # Kelvin
    "time_step": 0.002,  # ps (2 fs)
    "friction": 1.0,  # /ps
    "simulation_steps": 20000,  # steps (40 ps total)
    "report_interval": 100,
    
    # Bias parameters - 2D configuration
    "hills_per_d": 2,  # Hills per CV dimension (2 per CV × 2 CVs = 4 total hills for 2D)
    "hills_space": ((-np.pi, np.pi), (-np.pi, np.pi)),  # 2D space: ((phi_min, phi_max), (psi_min, psi_max))
    "hills_height": 50.0,  # kJ/mol - Lower than 1D (energy spread over 2D)
    "hills_width": [1.2, 1.2],  # radians - Width initial guess for each dimension [phi_width, psi_width]
    "min_width": 0.3,  # radians - Minimum width to prevent degenerate narrow Gaussians
    "multivariate": True,  # Use PLUMED multivariate format (enables correlation)
    
    # Evaluator parameters - 2D configuration
    "n_bins": 25,  # 25x25 = 625 bins
    
    # Optimizer parameters
    # For 2D with hills_per_d=2: 4 hills × 6 params/hill = 24 parameters
    "sigma": 0.2,  # CMA-ES step size
    "population_size": 32,  # Auto-calculated if None: max(16, int(4 + 3 * np.log(n_params)))
    "max_generations": 200,
    "n_workers": 16,  # Fewer workers due to longer simulations
    "n_replicas": 1,  # Number of replicas per evaluation
    "early_stop_patience": 50,  # Stop if no improvement for N generations
    "early_stop_threshold": 1e-5,  # Minimum improvement to reset patience
    
    # CV range for plotting
    "cv_range": ((-np.pi, np.pi), (-np.pi, np.pi)),  # 2D: (phi_range, psi_range)
}
