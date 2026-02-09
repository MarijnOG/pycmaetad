"""OpenMM samplers with bias integration."""

from .base import OpenMMSampler
from ..bias.base import Bias
from pathlib import Path
import numpy as np


class OpenMMPlumedSampler(OpenMMSampler):
    """OpenMM sampler for molecular systems with any bias type.
    
    Works with both PlumedBias and CustomForceBias.
    Supports multiple PDB files for different initial positions.
    """
    
    def __init__(
        self,
        pdb_file: str | list[str],
        forcefield_files: list[str],
        temperature: float,
        time_step: float,
        friction: float,
        simulation_steps: int,
        report_interval: int = 1000
    ):
        super().__init__(temperature, time_step, friction, simulation_steps, report_interval)
        # Store original parameter for reconstruction in workers
        self.pdb_file = pdb_file
        # Support both single PDB file and list of PDB files
        if isinstance(pdb_file, str):
            self._pdb_files = [pdb_file]
        else:
            self._pdb_files = list(pdb_file)
        self.forcefield_files = forcefield_files
    
    def _load_system(self, pdb_index=None):
        """Load PDB and create system from forcefield.
        
        Args:
            pdb_index: Optional index to select specific PDB file.
                      If None and multiple PDBs, selects randomly.
                      If provided, uses: pdb_files[pdb_index % len(pdb_files)]
        """
        # Select PDB file
        if pdb_index is not None:
            # Deterministic selection based on provided index
            pdb_file = self._pdb_files[pdb_index % len(self._pdb_files)]
        elif len(self._pdb_files) > 1:
            # Random selection if multiple PDBs and no index provided
            import random
            pdb_file = random.choice(self._pdb_files)
        else:
            # Single PDB file
            pdb_file = self._pdb_files[0]
        
        pdb = self.app.PDBFile(pdb_file)
        forcefield = self.app.ForceField(*self.forcefield_files)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=self.app.NoCutoff,
            nonbondedCutoff=1*self.unit.nanometer,
            constraints=self.app.HBonds
        )
        return pdb.topology, system, pdb.positions
    
    def run(self, output_path: str, bias: Bias = None, seed: int = None):
        """Run MD simulation with optional bias.
        
        Args:
            output_path: Directory for output files.
            bias: Optional Bias object (any type that implements get_openmm_force()).
            seed: Optional seed for deterministic PDB selection when multiple PDB files provided.
                  Used as: pdb_index = seed % len(pdb_files)
            
        Returns:
            OpenMM Simulation object after completion.
        """
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Load system (pass seed for deterministic PDB selection)
        topology, system, positions = self._load_system(pdb_index=seed)
        
        # Add bias if provided
        if bias is not None:
            # For PlumedBias, need to set output path first
            if hasattr(bias, 'set_output_path'):
                bias.set_output_path(output_path)
            
            # Get OpenMM force (works for any bias type!)
            bias_force = bias.get_openmm_force()
            system.addForce(bias_force)
        
        # Create integrator and simulation
        integrator = self._setup_integrator()
        simulation = self._create_simulation(topology, system, integrator)
        simulation.context.setPositions(positions)
        
        # Minimize energy
        self._minimize_energy(simulation)
        
        # Add reporters
        simulation.reporters.append(
            self.app.PDBReporter(f"{output_path}/output.pdb", self.report_interval)
        )
        simulation.reporters.append(
            self.app.StateDataReporter(
                f"{output_path}/state.log",
                self.report_interval,
                step=True,
                potentialEnergy=True,
                temperature=True
            )
        )
        
        # Run simulation
        print(f"Running {self.simulation_steps} steps of MD...")
        simulation.step(self.simulation_steps)
        print("Simulation complete.")
        
        # Ensure COLVAR file is fully written before returning
        # PLUMED may buffer writes, so we need to ensure the file is complete
        if bias is not None:
            import time
            time.sleep(0.1)  # Brief pause to let PLUMED finish writing
            # Also explicitly sync the COLVAR file if it exists
            colvar_file = Path(output_path) / "COLVAR"
            if colvar_file.exists():
                # Touch the file to ensure filesystem has flushed it
                colvar_file.touch()
        
        return simulation


class MullerBrownSampler(OpenMMSampler):
    """OpenMM sampler for Muller-Brown 2D potential."""
    
    def __init__(
        self,
        temperature: float,
        time_step: float,
        friction: float,
        simulation_steps: int,
        report_interval: int = 1000,
        initial_position: tuple[float, float] = (-0.5, 1.5)
    ):
        super().__init__(temperature, time_step, friction, simulation_steps, report_interval)
        self.initial_position = initial_position
    
    def _create_muller_brown_force(self):
        """Create Muller-Brown potential as CustomExternalForce."""
        from openmm import CustomExternalForce
        
        # Muller-Brown potential: V = sum_i A_i * exp(a_i*(x-x0_i)^2 + b_i*(x-x0_i)*(y-y0_i) + c_i*(y-y0_i)^2)
        # Parameters from standard literature values
        mb_potential = (
            "A1*exp(a1*(x-x01)^2 + b1*(x-x01)*(y-y01) + c1*(y-y01)^2) + "
            "A2*exp(a2*(x-x02)^2 + b2*(x-x02)*(y-y02) + c2*(y-y02)^2) + "
            "A3*exp(a3*(x-x03)^2 + b3*(x-x03)*(y-y03) + c3*(y-y03)^2) + "
            "A4*exp(a4*(x-x04)^2 + b4*(x-x04)*(y-y04) + c4*(y-y04)^2); "
            "A1=-200; a1=-1; b1=0; c1=-10; x01=1; y01=0; "
            "A2=-100; a2=-1; b2=0; c2=-10; x02=0; y02=0.5; "
            "A3=-170; a3=-6.5; b3=11; c3=-6.5; x03=-0.5; y03=1.5; "
            "A4=15; a4=0.7; b4=0.6; c4=0.7; x04=-1; y04=1"
        )
        
        force = CustomExternalForce(mb_potential)
        force.addParticle(0, [])
        
        return force
    
    def _load_system(self):
        """Create 1-particle system with Muller-Brown potential."""
        from openmm import System
        
        system = System()
        system.addParticle(1.0 * self.unit.dalton)
        
        # Add base Muller-Brown potential
        mb_force = self._create_muller_brown_force()
        system.addForce(mb_force)
        
        # Create minimal topology
        topology = self.app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("MB", chain)
        topology.addAtom("P", self.app.element.Element.getBySymbol("C"), residue)
        
        # Initial positions
        positions = self.unit.Quantity(
            np.array([[self.initial_position[0], self.initial_position[1], 0.0]]),
            self.unit.nanometers
        )
        
        return topology, system, positions
    
    def run(self, output_path: str, bias: Bias = None):
        """Run Muller-Brown simulation with optional bias."""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Load system
        topology, system, positions = self._load_system()
        
        # Add bias if provided
        if bias is not None:
            if hasattr(bias, 'set_output_path'):
                bias.set_output_path(output_path)
            
            bias_force = bias.get_openmm_force()
            system.addForce(bias_force)
        
        # Create and run simulation
        integrator = self._setup_integrator()
        simulation = self._create_simulation(topology, system, integrator)
        simulation.context.setPositions(positions)
        
        # Add reporters
        simulation.reporters.append(
            self.app.PDBReporter(f"{output_path}/output.pdb", self.report_interval)
        )
        simulation.reporters.append(
            self.app.StateDataReporter(
                f"{output_path}/state.log",
                self.report_interval,
                step=True,
                potentialEnergy=True,
                temperature=True
            )
        )
        
        print(f"Running {self.simulation_steps} steps on Muller-Brown surface...")
        simulation.step(self.simulation_steps)
        print("Simulation complete.")
        
        return simulation
