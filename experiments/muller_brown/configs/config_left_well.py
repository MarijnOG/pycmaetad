"""Tight search space configuration.

Restricted configuration with parameter ranges focused on the well regions.
Centers are constrained to stay near the three Muller-Brown minima,
reducing wasted exploration in high-energy regions.
"""

import numpy as np

CONFIG = {
    "name": "left_well",
    "description": "Restricted search space focused on well regions",
    
    # Bias parameters
    "num_gaussians": 3,
    "height_range": (0.0, 250.0),                           # kJ/mol
    "center_x_range": (-0.9, 1.3),                          # nm - restricted to well region with margin
    "center_y_range": (-0.3, 1.8),                          # nm - restricted to well region with margin
    "log_variance_x_range": (np.log(0.15**2), np.log(0.5**2)),  # sigma: 0.15-0.5 nm
    "log_variance_y_range": (np.log(0.15**2), np.log(0.5**2)),  # sigma: 0.15-0.5 nm
    
    # Well positions for initial seeding
    "well_positions": [(-0.5, 1.5), (0.6, 0.0), (1.0, 0.0)],
    "initial_height": 150.0,                                # kJ/mol
    "initial_sigma": 0.2,                                   # nm
    
    # Sampler parameters
    "temperature": 300.0,                                   # Kelvin
    "time_step": 0.001,                                     # ps (1 fs)
    "friction": 5.0,                                        # /ps
    "simulation_steps": 40000,                              # steps (40 ps total)
    "report_interval": 100,
    "initial_position": (-0.5, 1.5),                        # Start in left well
    
    # Evaluator parameters
    "evaluation_x_range": (-0.9, 1.3),                      # nm
    "evaluation_y_range": (-0.3, 1.8),                      # nm
    "n_bins": 50,                                           # 50x50 = 2500 bins
    
    # Optimizer parameters
    "sigma": 0.2,                                           # CMA-ES step size
    "population_size": 24,                                  # 2× default for 18 params
    "max_generations": 100,
    "n_workers": 12,
    "n_replicas": 1,                                        # Average over 2 replicas per individual
    "early_stop_patience": 0,                               # 0 = disabled
    
    # CV range for plotting
    "cv_range": ((-0.9, 1.3), (-0.3, 1.8)),
}
