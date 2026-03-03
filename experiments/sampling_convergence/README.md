# Sampling Convergence Analysis

This experiment evaluates how well sampled trajectories converge to the true Boltzmann distribution as simulation length increases.

## Overview

The analytical evaluator computes the **ground truth Boltzmann distribution** of the (Muller-Brown + bias) landscape by evaluating on a fine grid, without running MD simulations. The sampled evaluator computes the **empirical distribution** from MD trajectory positions.

**KL divergence** measures how closely the sampled distribution matches the true Boltzmann distribution. Lower KLD indicates better sampling convergence.

This analysis uses a **strong bias** (heights 200-240 kJ/mol) designed to flatten the Muller-Brown surface and enable exploration. The test:
- Tests multiple sampling times with increasing length
- Runs 5 replicates per sampling time to compute mean and standard deviation
- Measures KLD between empirical distribution and analytical Boltzmann distribution
- Shows how sampling quality improves with longer trajectories

## Key Concept

Unlike measuring "flatness" (KLD to uniform), this directly measures **sampling quality**:
- Perfect sampling → empirical distribution = Boltzmann distribution → KLD = 0
- Poor sampling → empirical distribution ≠ Boltzmann distribution → KLD > 0
- Convergence: KLD should decrease with longer simulation times

## Usage

```bash
cd experiments/sampling_convergence
python convergence_analysis.py
```

## Output

All results are saved to the `output/` directory:
- `output/sampling_convergence_analysis.png` - Main convergence plot
- `output/convergence_data.npz` - Raw data (NumPy compressed format)
- `output/sim_*/` - Temporary directories for simulation outputs (auto-cleaned)

## Key Insights

This analysis demonstrates:

1. **Sampling convergence**: How empirical distribution approaches true Boltzmann with longer sampling
2. **Statistical uncertainty**: Variance across replicates shows stochastic nature of MD
3. **Quality metric**: KLD provides quantitative measure of sampling adequacy
4. **Analytical validation**: Ground truth allows objective assessment of sampling quality
5. **Practical guidance**: Determines simulation length needed for reliable statistics

## Configuration

- **System**: Muller-Brown potential with fixed 3-Gaussian bias
- **Temperature**: 300 K
- **Bins**: 50×50
- **Time step**: 0.001 ps
- **CV range**: X=(-1.5, 1.5), Y=(-0.5, 2.0)
