


"""
Polyproline Fixed-Bias Production Simulation
============================================

This script runs a production molecular dynamics simulation of polyproline using a fixed bias
potential defined by a PLUMED HILLS file. It uses OpenMM and PLUMED via pycmaetad to apply the bias,
and generates diagnostic plots of the resulting conformational sampling.

Usage:
    python polyproline_production.py --hills HILLS_FILE [--steps N] [--output DIR]

Arguments:
    --hills   Path to PLUMED HILLS file (required)
    --steps   Number of MD steps to run (default: 10,000,000)
    --output  Output directory for results (default: output_production)

Outputs:
    - Simulated trajectory and COLVAR file
    - Diagnostic plots: bias landscape, CV histogram, CV time series
    - Pickled result and bias objects for further analysis
"""

import numpy as np
import pickle
import argparse
from pathlib import Path

# Import pycmaetad modules for bias, sampling, evaluation, and visualization
from pycmaetad.bias import PlumedHillBias
from pycmaetad.sampler import OpenMMPlumedSampler
from pycmaetad.evaluator import UniformKLEvaluator1D
from pycmaetad.visualization.colvar import (
    plot_cv_histogram,
    plot_cv_time_series
)
from pycmaetad.visualization.bias import plot_bias_landscape_1d


# =================== CONSTANTS ===================
# Script directory and simulation parameters
SCRIPT_DIR = Path(__file__).parent.resolve()

# Input files and forcefield
PDB_FILES = [str(SCRIPT_DIR / "pp2.pdb")]  # Use one starting structure for production
FORCEFIELD_FILES = ["amber14-all.xml", "amber14/tip3pfb.xml"]

# MD simulation parameters
TEMPERATURE = 300.0
TIME_STEP = 0.002
FRICTION = 1.0
REPORT_INTERVAL = 1000

# PLUMED template for bias
PLUMED_TEMPLATE = SCRIPT_DIR / "plumed_template.dat"

# Histogram/evaluator parameters
BIN_EDGES = np.linspace(-np.pi, np.pi, 40)
IS_2D = False



# =================== UTILITY FUNCTIONS ===================

def load_hills_file(hills_file):
    """
    Load centers, widths, heights from a PLUMED HILLS file (1D).
    Args:
        hills_file: Path to HILLS file
    Returns:
        centers, widths, heights (np.ndarray)
    """
    centers, widths, heights = [], [], []
    with open(hills_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            centers.append(float(fields[1]))
            widths.append(float(fields[2]))
            heights.append(float(fields[3]))
    return np.array(centers), np.array(widths), np.array(heights)



def make_fake_result(output_dir):
    """
    Create a minimal optimizer-style result dictionary for plotting compatibility.
    Args:
        output_dir: Output directory Path
    Returns:
        result dict
    """
    return {
        "best_generation": 0,
        "history": [{
            "generation": 0,
            "output_dir": output_dir / "gen000",
            "all_scores": [0.0]
        }],
    }



# =================== MAIN PRODUCTION FUNCTION ===================

def run_production(hills_file, steps, output_dir):
    """
    Run a fixed-bias production simulation using a PLUMED HILLS file.
    Args:
        hills_file: Path to HILLS file
        steps: Number of MD steps
        output_dir: Output directory name
    """
    output_dir = Path(SCRIPT_DIR) / output_dir

    # If output directory exists, clear it for a fresh run
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create generation and individual subdirectories
    gen_dir = output_dir / "gen000"
    ind_dir = gen_dir / "ind000"
    gen_dir.mkdir(exist_ok=True)
    ind_dir.mkdir(exist_ok=True)

    # ---- Logging: Run info ----
    print("=" * 60)
    print("FIXED-BIAS PRODUCTION RUN")
    print("=" * 60)
    print(f"HILLS file: {hills_file}")
    print(f"Steps: {steps:,}")
    print(f"Output: {output_dir}")

    # ---- Load HILLS file ----
    centers, widths, heights = load_hills_file(hills_file)
    print("\nLoaded hills:")
    print(f"  Count:   {len(centers)}")
    print(f"  Centers: {centers[:5]} ...")
    print(f"  Widths:  {widths[:5]} ...")
    print(f"  Heights:{heights[:5]} ...")

    # ---- Create bias object from HILLS ----
    bias = PlumedHillBias.from_hills_file(
        hills_file=hills_file,
        plumed_template=PLUMED_TEMPLATE,
        hills_space=(-np.pi, np.pi),
    )

    # ---- Set up MD sampler ----
    sampler = OpenMMPlumedSampler(
        pdb_file=PDB_FILES,
        forcefield_files=FORCEFIELD_FILES,
        temperature=TEMPERATURE,
        time_step=TIME_STEP,
        friction=FRICTION,
        simulation_steps=steps,
        report_interval=REPORT_INTERVAL
    )

    # ---- Run simulation ----
    sampler.run(output_path=ind_dir, bias=bias)

    # ---- Save minimal result for plotting ----
    result = make_fake_result(output_dir)
    with open(output_dir / "production_result.pkl", "wb") as f:
        pickle.dump({"result": result, "bias": bias}, f)

    # ---- Evaluate KL divergence from COLVAR ----
    evaluator = UniformKLEvaluator1D(bin_edges=BIN_EDGES)
    colvar_file = ind_dir / "COLVAR"
    colvar_data = np.loadtxt(colvar_file, comments="#")
    if colvar_data.ndim == 1:
        cv_values = np.array([colvar_data[1]])
    else:
        cv_values = colvar_data[:, 1]
    kl_div = evaluator.evaluate(cv_values)

    # ---- Generate diagnostic plots ----
    plot_cv_histogram(result, output_path=output_dir / "cv_histogram.png")
    plot_cv_time_series(result, output_path=output_dir / "cv_time_series.png")

    all_params = [bias.get_parameters()]
    plot_bias_landscape_1d(
        bias,
        all_params=all_params,
        cv_range=(-np.pi, np.pi),
        output_path=output_dir / "bias_landscape.png",
        mintozero=True
    )

    # ---- Final logging ----
    print("\n" + "=" * 60)
    print("RUN COMPLETE")
    print(f"KL Divergence: {kl_div:.4f}")
    print("=" * 60)
    print(f"Simulated time: {steps * TIME_STEP:.2f} ps")



# =================== ENTRY POINT ===================

def main():
    """
    Parse command-line arguments and run the production simulation.
    """
    parser = argparse.ArgumentParser(
        description="Run fixed-bias production simulation from PLUMED HILLS file"
    )
    parser.add_argument("--hills", required=True, help="PLUMED HILLS file")
    parser.add_argument("--steps", type=int, default=10_000_000, help="Number of MD steps")
    parser.add_argument("--output", default="output_production", help="Output directory")

    args = parser.parse_args()
    run_production(args.hills, args.steps, args.output)


if __name__ == "__main__":
    main()