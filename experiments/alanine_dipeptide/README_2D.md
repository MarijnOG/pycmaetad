# Alanine Dipeptide 2D Optimization Example

This example demonstrates CMA-ES optimization for **2D collective variables** (phi and psi Ramachandran angles) to achieve uniform sampling in 2D space.

## Files

- `plumed_template_2d.dat` - PLUMED template defining phi and psi torsion angles
- `alanine_dipeptide_2d_example.py` - Main optimization script
- `../alanine-dipeptide-nowater.pdb` - Alanine dipeptide structure

## Running the Example

```bash
# Run optimization from scratch
python alanine_dipeptide_2d_example.py run

# Resume from checkpoint
python alanine_dipeptide_2d_example.py resume

# Generate plots only
python alanine_dipeptide_2d_example.py plot

# Run and plot
python alanine_dipeptide_2d_example.py both
```

## ⚠️ Special Considerations for 2D

### 1. **Larger Parameter Space**
- **1D**: With `hills_per_d=2`, you have 2 hills × 3 params = **6 parameters**
  - Each 1D hill: (center, height, width)
- **2D**: With `hills_per_d=2`, you have 2×2 = 4 hills × 6 params = **24 parameters**
  - Each 2D hill: (center_phi, center_psi, height, width_phi, width_psi, correlation)
- **Impact**: CMA-ES population size scales with parameters (~14-20 for 24 params vs ~6 for 6 params)

### 2. **Computational Cost**
- **Longer simulations needed**: 2D space requires more sampling
  - 1D: 250,000 steps (250 ps) may suffice
  - 2D: 500,000+ steps (500 ps - 1 ns) recommended
- **More expensive histograms**: 2D binning is O(n) per dimension
  - 1D: 30 bins
  - 2D: 25×25 = 625 bins (consider starting with fewer bins)
- **Longer optimization time**: Each generation takes ~2-4× longer

### 3. **Periodicity Handling**
Both phi and psi are **periodic** angles ranging from -π to π:
- Hills near boundaries wrap around
- Distance calculations must account for periodicity
- Evaluator should handle periodic boundaries correctly

### 4. **Initial Hill Placement**
For 2D, manual initialization is important:
```python
def create_2d_initial_mean(hills_per_d):
    """Place hills in a 2D grid pattern."""
    # For hills_per_d=2: place at (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)
    # This creates uniform coverage in normalized [0,1]² space
```

**Known secondary structures to seed near:**
- α-helix region: (φ ≈ -60°, ψ ≈ -45°)
- β-sheet region: (φ ≈ -120°, ψ ≈ +120°)
- PPII region: (φ ≈ -75°, ψ ≈ +145°)

### 5. **Visualization**
2D results are visualized as:
- **Ramachandran plots**: Histogram of (phi, psi) sampling
- **Free energy surfaces**: -kT log(P(phi, psi))
- **Trajectory overlays**: Simulation paths on Ramachandran space
- **Hill placement**: Contour plots showing Gaussian centers

### 6. **Convergence Criteria**
- 2D uniform sampling is **harder** to achieve
- KL divergence decreases more slowly
- May need more generations (30-50 vs 20-30 for 1D)
- Early stopping patience should be higher

### 7. **Memory and I/O**
- More COLVAR data: 2 CVs × N_frames per simulation
- Larger checkpoint files: More parameters and population
- More plots generated: Each generation produces 2D contour plots

## Recommended Settings

### Conservative (for testing)
```python
SIMULATION_STEPS = 250000    # 500 ps at 2 fs
N_BINS = 20                  # 20×20 = 400 bins
POPULATION_SIZE = 12
MAX_GENERATIONS = 20
N_WORKERS = 8
```

### Production (for publication)
```python
SIMULATION_STEPS = 500000    # 1 ns at 2 fs
N_BINS = 30                  # 30×30 = 900 bins
POPULATION_SIZE = 20
MAX_GENERATIONS = 50
N_WORKERS = 16
N_REPLICAS = 2               # Average over replicas for robustness
```

## Expected Outcomes

A successful 2D optimization should show:
1. **Convergence**: KL divergence decreases and plateaus
2. **Uniform Coverage**: Ramachandran plot shows even sampling across accessible regions
3. **Hill Spreading**: Hills distributed to fill low-barrier regions
4. **Stable Sampling**: Later generations maintain uniform distribution

## Common Issues

### Issue: Stuck in one region (e.g., only α-helix sampled)
**Solution**: 
- Increase simulation length
- Use higher bias heights
- Add more hills (increase `hills_per_d`)

### Issue: Very slow convergence
**Solution**:
- Check if simulations are long enough (should visit multiple regions)
- Reduce number of bins temporarily (20×20 instead of 30×30)
- Ensure adequate barrier crossing

### Issue: High variance in scores between replicas
**Solution**:
- Increase simulation length
- Use more replicas (`N_REPLICAS=3` or `4`)
- Average over longer trajectories

## Comparison: 1D vs 2D

| Aspect | 1D (Polyproline) | 2D (Alanine Dipeptide) |
|--------|------------------|------------------------|
| Parameters | 6 (2 hills × 3) | 24 (4 hills × 6) |
| Simulation length | 250k steps | 500k+ steps |
| Population size | ~6-8 | ~16-20 |
| Bins | 30 | 25×25 (625) |
| Convergence time | 15-25 gens | 30-50 gens |
| Computational cost | 1× (baseline) | ~4-6× |
| Main challenge | Barrier crossing | Coverage + barriers |

## Advanced: Adaptive Hill Placement

For more sophisticated 2D optimization, consider:
1. **Clustering-based initialization**: Seed hills near known metastable states
2. **Temperature-based seeding**: Different hills for different regions
3. **Adaptive binning**: More bins in well-sampled regions
4. **Enhanced sampling**: Combine with tempering or other methods

## References

- Alanine dipeptide as a test system: Lindorff-Larsen et al., Science 2011
- Ramachandran plot interpretation: Ramachandran & Sasisekharan, Adv. Protein Chem. 1968
- 2D metadynamics: Laio & Gervasio, Rep. Prog. Phys. 2008
