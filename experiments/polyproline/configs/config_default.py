"""Default configuration for polyproline optimization.

Standard parameters for CMA-ES optimization of bias parameters
for uniform sampling of the polyproline conformational space.
"""

import numpy as np

# Calculate INITIAL_MEAN based on HILLS_PER_D
def _compute_initial_mean(hills_per_d):
    """Compute initial mean with evenly spaced centers."""
    initial_mean = np.ones(hills_per_d * 3) * 0.5  # Heights and widths at midpoint
    initial_mean[:hills_per_d] = np.linspace(0, 1, hills_per_d, endpoint=False)  # Centers evenly spaced
    return initial_mean

CONFIG = {
    "name": "default",
    "description": "Standard polyproline optimization parameters",
    
    # Sampler parameters
    "temperature": 300.0,  # Kelvin
    "time_step": 0.004,  # ps (4 fs)
    "friction": 1.0,  # /ps
    "simulation_steps": 50000,  # 50k steps = 200 ps
    "report_interval": 1000,
    "pdb_files": ["pp1.pdb", "pp2.pdb"],  # Two different starting structures
    
    # Bias parameters
    "hills_per_d": 2,
    "hills_space": (-np.pi, np.pi),
    "hills_height": 150.0,  # kJ/mol
    "hills_width": 1.5,  # nm - broader, smoother hills
    
    # Evaluator parameters
    "bin_edges": np.linspace(-np.pi, np.pi, 31),
    "is_2d": False,
    
    # Optimizer parameters
    "initial_mean": _compute_initial_mean(2),  # Computed based on hills_per_d
    "sigma": 0.1,  # CMA-ES step size
    "population_size": 20,
    "max_generations": 500,
    "n_workers": 20,
    "n_replicas": 2,  # Each individual evaluated with 2 different starting structures
    "early_stop_patience": 0,  # 0 = disabled
    
    # CV range for plotting
    "cv_range": (-np.pi, np.pi),
    
    # Files
    "plumed_template": "plumed_template.dat",
    "forcefield_files": ["amber14-all.xml", "amber14/tip3pfb.xml"],
}
