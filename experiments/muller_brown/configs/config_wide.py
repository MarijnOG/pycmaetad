"""Wide search space configuration.

Original configuration with broad parameter ranges that allow
exploration across the entire Muller-Brown potential landscape.
"""

import numpy as np

CONFIG = {
    "name": "wide_ranges",
    "description": "Original wide search space covering full MB landscape",
    
    # Bias parameters
    "num_gaussians": 3,
    "height_range": (0.0, 200.0),  # kJ/mol
    "center_x_range": (-1.5, 1.5),  # nm
    "center_y_range": (-0.5, 2.5),  # nm
    "log_variance_x_range": (np.log(0.15**2), np.log(0.5**2)),  # sigma: 0.15-0.5 nm
    "log_variance_y_range": (np.log(0.15**2), np.log(0.5**2)),  # sigma: 0.15-0.5 nm
    
    # Well positions for initial seeding
    "well_positions": [(-0.5, 1.5), (0.6, 0.0), (1.0, 0.0)],
    "initial_height": 500.0,  # kJ/mol
    "initial_sigma": 0.2,  # nm
    
    # Sampler parameters
    "temperature": 300.0,  # Kelvin
    "time_step": 0.0005,  # ps (0.5 fs)
    "friction": 5.0,  # /ps
    "simulation_steps": 10000,  # steps (12.5 ps total)
    "report_interval": 10,
    "initial_position": None,  # None = randomized per evaluation using seed
    
    # Evaluator parameters
    "evaluation_x_range": (-1.5, 1.5),  # nm
    "evaluation_y_range": (-0.5, 2.5),  # nm
    "n_bins": 50,  # 50x50 = 2500 bins
    
    # Optimizer parameters
    "sigma": 0.3,  # CMA-ES step size
    "population_size": 36,  # Large population for 18 parameters
    "max_generations": 150,
    "n_workers": 20,
    "n_replicas": 1,  # Single starting position for Muller-Brown
    "early_stop_patience": 0,  # 0 = disabled
    
    # CV range for plotting
    "cv_range": ((-1.5, 1.5), (-0.5, 2.5)),
}
