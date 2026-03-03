# Evaluator Comparison Tool

Visual comparison of different evaluators on synthetic distributions to understand their behavior and sensitivity.

## Overview

This tool compares **KL Divergence** and **Hybrid** evaluators by testing them on various synthetic distributions in both 1D and 2D.

## Usage

```bash
python compare_evaluators.py
```

This generates two visualization files:
- `evaluator_comparison_1d.png` - 1D comparison
- `evaluator_comparison_2d.png` - 2D comparison

## Test Distributions

### 1D (Dihedral-angle style)
- **Uniform**: Ideal uniform distribution
- **Gaussian**: Concentrated in middle
- **Bimodal**: Two peaks 
- **Sparse (70%)**: Uniform sampling from only 70% of the domain
- **Edge-biased**: More samples at boundaries
- **Very Sparse (40%)**: Uniform sampling from only 40% of the domain

### 2D (Position-space style)
- **Uniform**: Ideal uniform distribution
- **Gaussian Cluster**: Single concentrated cluster
- **Three Wells**: Three separate clusters (metastable states)
- **Sparse (70%)**: Uniform sampling from only 70% of each dimension
- **Corner-biased**: Samples concentrated in corners
- **Very Sparse (40%)**: Uniform sampling from only 40% of each dimension

## Evaluators Compared

### 1. KL Divergence (`UniformKLEvaluator1D` / `UniformKLEvaluator2D`)
- Measures how evenly samples are distributed **among bins that ARE sampled**
- **Does NOT penalize missing/empty bins**
- Low score when sampled bins have uniform density
- Can give good scores to sparse but uniform sampling

**Best for:** Comparing trajectory quality when you know space is well-explored

### 2. Hybrid (coverage_weight=1.0)
- Light coverage penalty
- Minimal impact on well-covered distributions  
- Sparse 40%: adds ~1.5% penalty
- Good for quick optimization

**Best for:** Fast convergence when coverage isn't critical

### 3. Hybrid (coverage_weight=5.0)
- Moderate coverage penalty
- Balances evenness and completeness
- Sparse 40%: adds ~7% penalty
- Sweet spot for most use cases

**Best for:** General-purpose metadynamics optimization

### 4. Hybrid (coverage_weight=20.0)
- Heavy coverage penalty
- Strongly penalizes incomplete sampling
- Sparse 40%: adds ~30% penalty!
- Ensures thorough exploration

**Best for:** Production runs where complete exploration is critical

## Interpreting Results

### What the visualizations show:

**Row 1: Distribution**
- 1D: Histogram of samples
- 2D: Heatmap of samples
- Red dashed line (1D): Ideal uniform density

**Row 2: Scores**
- Bar chart comparing four evaluators (KL, H(1.0), H(5.0), H(20.0))
- Lower scores = better (closer to uniform)
- Height differences show sensitivity

**Row 3: Coverage**
- Green: Fraction of bins sampled
- Red: Fraction of bins empty
- High coverage = good exploration

### Understanding the Penalty Scaling

The coverage penalty formula is:
```
penalty = coverage_weight × (target_coverage - actual_coverage)²
```

**Example: Very Sparse 40% distribution (target=90%)**
- Coverage gap: 0.90 - 0.40 = 0.50
- Base penalty: (0.50)² = 0.25
- Final penalties:
  - weight=1.0:  0.25 × 1  = 0.25  → adds ~1.5% to score
  - weight=5.0:  0.25 × 5  = 1.25  → adds ~7% to score
  - weight=20.0: 0.25 × 20 = 5.00  → adds ~30% to score!

The penalty is **quadratic** in the gap, so small coverage issues have minimal impact, but large gaps are heavily penalized.

### Why Different Weights for 1D vs 2D?

- **1D (50 bins)**: Uses higher weights (1.0, 5.0, 20.0) because fewer bins means smaller absolute penalties
- **2D (900 bins)**: Uses lower weights (0.5, 2.0) because more bins amplifies the penalty effect

Choose weights based on:
1. Number of bins (more bins → lower weights needed)
2. Coverage target (higher target → larger gaps → stronger penalties)
3. Optimization priority (speed vs completeness)

## Key Insights

### Coverage Detection Works!
The hybrid evaluator **correctly detects and penalizes** sparse coverage:

**1D Example (50 bins):**
```
Distribution         KL Div    Hybrid(1.0)  Hybrid(5.0)  Hybrid(20.0)
----------------------------------------------------------------------
Uniform               9.59        9.59         9.59         9.59
Sparse 70%           12.42       12.46        12.62        13.22  (+6.5%)
Very Sparse 40%      16.86       17.11        18.11        21.86  (+30%!)
```

**2D Example (900 bins):**
```
Distribution         KL Div    Hybrid(0.5)  Hybrid(2.0)
----------------------------------------------------------------------
Uniform               0.09      370.72       370.72
Very Sparse 40%       1.85      546.54       547.36  (+47%!)
```

### Why KL Divergence Alone Isn't Enough

KL divergence can miss incomplete exploration because it only compares densities in sampled bins. A distribution that samples only 40% of space uniformly can score similarly to full coverage with slight unevenness.

**The Hybrid Fix:**
- Adds explicit coverage tracking
- Penalizes empty/unsampled bins
- Quadratic penalty ensures large gaps are heavily penalized
- Weight parameter controls the balance

### Tuning Coverage Weight

**Rule of thumb:**
```python
# For N bins, KL scores are typically ~0.1*N to ~0.3*N
# Choose weight so penalty ~ 0.1-0.5 × KL_score for target gaps

weight ≈ (typical_KL_score / N_bins) × desired_penalty_fraction / (expected_gap²)
```

**Practical recommendations:**
- 1D (50 bins): weight = 5-20
- 2D (900 bins): weight = 0.5-2.0  
- 3D (>5000 bins): weight = 0.1-0.5

Increase weight if optimization converges before achieving good coverage.
Decrease weight if optimization gets stuck trying to fill difficult regions.

## Recommendations

### For Different Use Cases

**Development/Testing:**
- Use **KL divergence** for quick comparisons when coverage is already good
- Fast evaluation, interpretable scores

**General Optimization:**
- Use **Hybrid (5.0)** for 1D or **Hybrid (0.5-1.0)** for 2D
- Good balance between convergence speed and coverage
- Works well for most metadynamics applications

**Production/Publication:**
- Use **Hybrid (15.0-20.0)** for 1D or **Hybrid (2.0-5.0)** for 2D
- Ensures thorough exploration
- Reduces risk of missing important regions
- Scientifically rigorous

**Debugging Poor Coverage:**
- Start with high weights (20.0+ for 1D, 5.0+ for 2D)
- Optimization will strongly favor filling empty regions
- Gradually reduce weight once coverage improves

## Customization

You can adjust the evaluators' weights:

```python
from pycmaetad.evaluator.hybrid_uniform import HybridUniformEvaluator

# Light coverage penalty (faster convergence, may miss regions)
evaluator = HybridUniformEvaluator.from_ranges(
    ranges=(-3.14, 3.14),
    n_bins=50,
    kl_weight=1.0,
    coverage_weight=0.3,  # Light
    coverage_target=0.9   # 90% coverage target
)

# Heavy coverage penalty (thorough exploration, slower)
evaluator = HybridUniformEvaluator.from_ranges(
    ranges=(-3.14, 3.14),
    n_bins=50,
    kl_weight=1.0,
    coverage_weight=3.0,  # Heavy
    coverage_target=0.95  # 95% coverage target
)
```

## Technical Notes

### Normalization in KL Divergence
- **1D**: Uses density normalization (`density=True` in histogram)
- **2D**: Uses probability normalization (counts → probabilities)
- Both approaches are mathematically equivalent for KL divergence

### Coverage Threshold
Bins are counted as "sampled" if they contain > 1% of uniform density:
```python
threshold = uniform_density * 0.01
bins_sampled = np.sum(hist > threshold)
```

This avoids counting bins with insignificant sampling.

## Files Generated

- `evaluator_comparison_1d.png` - 1D analysis (16×10 inches, 150 DPI)
- `evaluator_comparison_2d.png` - 2D analysis (18×11 inches, 150 DPI)

Both are publication-quality and can be directly included in reports/papers.
