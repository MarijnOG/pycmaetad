"""Visualize the bias used in convergence analysis."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pycmaetad.bias import MultiGaussian2DForceBias


def muller_brown_potential(X, Y):
    """Muller-Brown potential (kJ/mol)."""
    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])
    
    V = np.zeros_like(X)
    for i in range(4):
        V += A[i] * np.exp(
            a[i] * (X - x0[i])**2 + 
            b[i] * (X - x0[i]) * (Y - y0[i]) + 
            c[i] * (Y - y0[i])**2
        )
    return V


# Create the same bias as in convergence_analysis.py
bias = MultiGaussian2DForceBias(
    n_gaussians=3,
    height_range=(0, 300),
    center_x_range=(-1.5, 1.5),
    center_y_range=(-0.5, 2.0),
    log_variance_x_range=(-6, 0),
    log_variance_y_range=(-6, 0),
)

# Stronger bias parameters matching convergence_analysis.py
# Parameter order: height, cx, cy, log_var_x, rho, log_var_y
params = np.array([
    220.0, 0.9, 0.0, -1.5, 0.0, -1.5,    # Well at (1.0, 0.0), no correlation
    200.0, 0.0, 0.5, -1.7, 0.0, -1.7,    # Well at (0.0, 0.5), no correlation
    240.0, -0.5, 1.5, -1.5, 0.0, -1.5,   # Well at (-0.5, 1.5), no correlation
])

bias.set_parameters(params)

print("Bias parameters:")
for i in range(3):
    idx = i * 6
    h, cx, cy, log_vx, rho, log_vy = params[idx:idx+6]
    sx, sy = np.exp(log_vx/2), np.exp(log_vy/2)
    print(f"  Gaussian {i+1}: h={h:.1f} kJ/mol, center=({cx:.2f}, {cy:.2f}), σ=({sx:.3f}, {sy:.3f}), ρ={rho:.2f}")

# Create grid
x_range = (-1.5, 1.5)
y_range = (-0.5, 2.0)
n_points = 200

x = np.linspace(x_range[0], x_range[1], n_points)
y = np.linspace(y_range[0], y_range[1], n_points)
X, Y = np.meshgrid(x, y)

# Compute potentials
V_mb = muller_brown_potential(X, Y)
V_bias = bias.evaluate_numpy(X, Y)
V_combined = V_mb + V_bias

# Create plot
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Muller-Brown
ax1 = axes[0, 0]
levels_mb = np.linspace(V_mb.min(), V_mb.min() + 150, 25)
im1 = ax1.contourf(X, Y, V_mb, levels=levels_mb, cmap='viridis', alpha=0.8)
ax1.contour(X, Y, V_mb, levels=levels_mb, colors='black', alpha=0.3, linewidths=0.5)
plt.colorbar(im1, ax=ax1, label='Energy (kJ/mol)')
ax1.set_xlabel('x (nm)', fontsize=11, fontweight='bold')
ax1.set_ylabel('y (nm)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Muller-Brown Potential', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Mark MB minima
mb_minima = [(1.0, 0.0), (0.0, 0.5), (-0.5, 1.5)]
for mx, my in mb_minima:
    ax1.scatter(mx, my, c='red', s=100, marker='*', edgecolors='black', linewidths=1.5, zorder=10)

# Plot 2: Bias
ax2 = axes[0, 1]
im2 = ax2.contourf(X, Y, V_bias, levels=20, cmap='plasma', alpha=0.8)
ax2.contour(X, Y, V_bias, levels=20, colors='black', alpha=0.3, linewidths=0.5)
plt.colorbar(im2, ax=ax2, label='Bias Energy (kJ/mol)')
ax2.set_xlabel('x (nm)', fontsize=11, fontweight='bold')
ax2.set_ylabel('y (nm)', fontsize=11, fontweight='bold')
ax2.set_title(f'(b) Bias Potential ({bias.n_gaussians} Gaussians)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Mark Gaussian centers
for i in range(bias.n_gaussians):
    g = bias._gaussians[i]
    ax2.scatter(g['cx'], g['cy'], c='yellow', s=150, marker='x', linewidths=3, zorder=10)
    ax2.annotate(f'G{i+1}', (g['cx'], g['cy']), xytext=(5, 5), 
                textcoords='offset points', fontsize=9, fontweight='bold', color='white')

# Plot 3: Combined
ax3 = axes[1, 0]
levels_comb = np.linspace(V_combined.min(), V_combined.min() + 150, 25)
im3 = ax3.contourf(X, Y, V_combined, levels=levels_comb, cmap='viridis', alpha=0.8)
ax3.contour(X, Y, V_combined, levels=levels_comb, colors='black', alpha=0.3, linewidths=0.5)
plt.colorbar(im3, ax=ax3, label='Energy (kJ/mol)')
ax3.set_xlabel('x (nm)', fontsize=11, fontweight='bold')
ax3.set_ylabel('y (nm)', fontsize=11, fontweight='bold')
ax3.set_title('(c) Combined Landscape (MB + Bias)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Overlay comparison
ax4 = axes[1, 1]
# Show MB contours in blue
ax4.contour(X, Y, V_mb, levels=15, colors='blue', alpha=0.6, linewidths=1.5, linestyles='solid')
# Show bias contours in red
ax4.contour(X, Y, V_bias, levels=15, colors='red', alpha=0.6, linewidths=1.5, linestyles='dashed')
# Show combined in green
ax4.contour(X, Y, V_combined, levels=15, colors='green', alpha=0.8, linewidths=2, linestyles='dotted')

# Mark important points
for mx, my in mb_minima:
    ax4.scatter(mx, my, c='blue', s=100, marker='*', edgecolors='black', linewidths=1.5, zorder=10)
for i in range(bias.n_gaussians):
    g = bias._gaussians[i]
    ax4.scatter(g['cx'], g['cy'], c='red', s=150, marker='x', linewidths=3, zorder=10)

ax4.set_xlabel('x (nm)', fontsize=11, fontweight='bold')
ax4.set_ylabel('y (nm)', fontsize=11, fontweight='bold')
ax4.set_title('(d) Contour Overlay', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='blue', linewidth=2, label='MB contours'),
    Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Bias contours'),
    Line2D([0], [0], color='green', linewidth=2, linestyle=':', label='Combined contours'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=10, label='MB minima'),
    Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=10, label='Bias centers'),
]
ax4.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.suptitle('Bias Analysis for Convergence Test', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

# Create output directory and save plot
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "bias_visualization.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_path}")

# Analyze the combined landscape
V_diff = V_combined.max() - V_combined.min()
V_mb_diff = V_mb.max() - V_mb.min()
print(f"\nLandscape analysis:")
print(f"  MB energy range: {V_mb.min():.1f} to {V_mb.min() + 150:.1f} kJ/mol (showing 150 kJ/mol range)")
print(f"  Bias max: {V_bias.max():.1f} kJ/mol")
print(f"  Combined energy range: {V_combined.min():.1f} to {V_combined.min() + 150:.1f} kJ/mol")
print(f"  Flattening achieved: {(1 - V_diff/V_mb_diff)*100:.1f}% (lower is flatter)")
