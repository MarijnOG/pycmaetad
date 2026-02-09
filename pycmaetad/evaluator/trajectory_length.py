"""Evaluator based on trajectory length (total distance traveled).

In a flat potential, particles diffuse freely and travel longer distances.
In a steep potential well, particles get trapped and travel shorter distances.
"""

from .base import Evaluator
import numpy as np
from pathlib import Path


class TrajectoryLengthEvaluator(Evaluator):
    """Evaluator that computes total trajectory length (distance traveled).
    
    Measures the total distance a particle travels during simulation:
        L = sum_t ||r_{t+1} - r_t||
    
    A flatter potential leads to more diffusion and longer trajectories.
    Since CMA-ES minimizes, we return the NEGATIVE trajectory length.
    """
    
    def __init__(self):
        """Initialize trajectory length evaluator."""
        pass
    
    def evaluate(self, positions: np.ndarray) -> float:
        """Compute total trajectory length.
        
        Args:
            positions: Nx2 or Nx3 array of particle positions (nm)
            
        Returns:
            Negative trajectory length (for minimization).
            Longer trajectories → more negative score → better.
        """
        if len(positions) < 2:
            return 1e6  # Penalty for insufficient data
        
        # Compute displacement vectors
        displacements = np.diff(positions, axis=0)
        
        # Compute distances
        distances = np.linalg.norm(displacements, axis=1)
        
        # Total length
        total_length = np.sum(distances)
        
        if not np.isfinite(total_length) or total_length == 0:
            return 1e6  # Penalty for invalid trajectory
        
        # Return NEGATIVE (CMA-ES minimizes, we want to maximize length)
        return -total_length
    
    def evaluate_from_file(self, trajectory_file: str) -> float:
        """Load trajectory from file and evaluate.
        
        Args:
            trajectory_file: Path to PDB trajectory file.
            
        Returns:
            Negative trajectory length.
        """
        positions = self._load_from_pdb(trajectory_file)
        return self.evaluate(positions)
    
    def _load_from_pdb(self, pdb_file: str) -> np.ndarray:
        """Load positions from multi-frame PDB trajectory.
        
        OpenMM's PDBReporter writes multiple MODEL records.
        We parse them manually to get all frames.
        """
        positions = []
        
        with open(pdb_file, 'r') as f:
            current_frame_atoms = []
            
            for line in f:
                if line.startswith('MODEL'):
                    # Start of new frame
                    current_frame_atoms = []
                elif line.startswith('ATOM') or line.startswith('HETATM'):
                    # Parse atom line
                    # PDB format: columns 31-38 (x), 39-46 (y), 47-54 (z)
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    current_frame_atoms.append([x, y, z])
                elif line.startswith('ENDMDL'):
                    # End of frame - save first atom's position
                    if current_frame_atoms:
                        positions.append(current_frame_atoms[0])  # First atom only
        
        if not positions:
            raise ValueError(f"No trajectory data found in {pdb_file}")
        
        # Convert from Angstroms to nm (OpenMM uses nm internally, PDB uses Angstroms)
        positions_array = np.array(positions) / 10.0
        
        return positions_array
