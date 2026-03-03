"""Tight search space configuration.

Restricted configuration with parameter ranges focused on the well regions.
Centers are constrained to stay near the three Muller-Brown minima,
reducing wasted exploration in high-energy regions.
"""

import numpy as np

CONFIG = {
    "name": "tight_ranges",
    "description": "Restricted search space focused on well regions",
    
    # Bias parameters
    "num_gaussians": 3,
    "height_range": (0.0, 200.0),  # kJ/mol
    # Wells span approximately x: -0.6 to 1.0, y: 0.0 to 1.5
    "center_x_range": (-0.9, 1.3),  # nm - restricted to well region with margin
    "center_y_range": (-0.3, 1.8),  # nm - restricted to well region with margin
    "log_variance_x_range": (np.log(0.15**2), np.log(0.5**2)),  # sigma: 0.15-0.5 nm
    "log_variance_y_range": (np.log(0.15**2), np.log(0.5**2)),  # sigma: 0.15-0.5 nm
    
    # Well positions for initial seeding (closer to actual minima)
    "well_positions": [(-0.5, 1.5), (0.6, 0.0), (1.0, 0.0)],
    "initial_height": 150.0,  # kJ/mol
    "initial_sigma": 0.2,  # nm
    
    # Sampler parameters
    "temperature": 300.0,  # Kelvin
    "time_step": 0.0005,  # ps (0.5 fs)
    "friction": 5.0,  # /ps
    "simulation_steps": 100000,  # steps (50 ps total) - needs longer to explore with bias
    "report_interval": 100,
    "initial_position": None,  # None = randomized per evaluation using seed
    
    # Evaluator parameters (match center ranges)
    "evaluation_x_range": (-0.9, 1.3),  # nm
    "evaluation_y_range": (-0.3, 1.8),  # nm
    "n_bins": 50,  # 50x50 = 2500 bins
    
    # Optimizer parameters
    "sigma": 0.2,  # CMA-ES step size (0.1 = ±0.22 nm, keeps hills near initial wells)
    "population_size": 24,  # 2× default for 18 params - good balance of exploration vs speed
    "max_generations": 200,
    "n_workers": 20,
    "n_replicas": 3,  # Average over 3 random starting positions per individual for robust evaluation
    "early_stop_patience": 0,  # 0 = disabled
    
    # CV range for plotting (match evaluation ranges)
    "cv_range": ((-0.9, 1.3), (-0.3, 1.8)),
}
