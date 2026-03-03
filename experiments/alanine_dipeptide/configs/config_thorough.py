"""Thorough configuration for alanine dipeptide 2D CMA-ES optimization.

Higher computational cost with longer simulations and more refined sampling.
Best for production runs where quality matters more than speed.
"""

import numpy as np

CONFIG = {
    "name": "thorough",
    "description": "High-quality configuration with longer simulations",
    
    # Sampler parameters
    "temperature": 300.0,  # Kelvin
    "time_step": 0.002,  # ps (2 fs)
    "friction": 1.0,  # /ps
    "simulation_steps": 50000,  # steps (100 ps total) - INCREASED for better sampling
    "report_interval": 100,
    
    # Bias parameters - 2D configuration
    "hills_per_d": 2,  # Hills per CV dimension (2 per CV × 2 CVs = 4 total hills for 2D)
    "hills_space": ((-np.pi, np.pi), (-np.pi, np.pi)),  # 2D space: ((phi_min, phi_max), (psi_min, psi_max))
    "hills_height": 50.0,  # kJ/mol - Lower than 1D (energy spread over 2D)
    "hills_width": [1.2, 1.2],  # radians - Width initial guess for each dimension [phi_width, psi_width]
    "min_width": 0.3,  # radians - Minimum width to prevent degenerate narrow Gaussians
    "multivariate": True,  # Use PLUMED multivariate format (enables correlation)
    
    # Evaluator parameters - 2D configuration
    "n_bins": 30,  # 30x30 = 900 bins - INCREASED for finer resolution
    
    # Optimizer parameters
    # For 2D with hills_per_d=2: 4 hills × 6 params/hill = 24 parameters
    "sigma": 0.2,  # CMA-ES step size
    "population_size": 40,  # INCREASED for better exploration
    "max_generations": 300,  # INCREASED for convergence
    "n_workers": 16,
    "n_replicas": 3,  # INCREASED: average over 3 runs for more robust evaluation
    "early_stop_patience": 75,  # INCREASED patience
    "early_stop_threshold": 1e-5,
    
    # CV range for plotting
    "cv_range": ((-np.pi, np.pi), (-np.pi, np.pi)),  # 2D: (phi_range, psi_range)
}
