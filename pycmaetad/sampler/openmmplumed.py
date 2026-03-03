"""OpenMM samplers with bias integration."""

from .base import OpenMMLangevinSampler
from ..bias.base import Bias
from pathlib import Path
import numpy as np


class ColvarReporter:
    """Custom OpenMM reporter for writing CV values in PLUMED COLVAR format.
    
    Writes collective variable (CV) values to a file compatible with PLUMED's
    COLVAR format, enabling analysis with PLUMED tools.
    """
    
    def __init__(self, file, reportInterval, cv_names=None, cv_ranges=None):
        """Initialize COLVAR reporter.
        
        Args:
            file: File path or file object to write to
            reportInterval: Frequency (in steps) for writing output
            cv_names: List of CV names (default: ['x', 'y', 'z'])
            cv_ranges: Dict with CV ranges, e.g. {'x': (-1.5, 1.5), 'y': (-0.5, 2.5)}
        """
        self._reportInterval = reportInterval
        self._cv_names = cv_names or ['x', 'y', 'z']
        self._cv_ranges = cv_ranges or {}
        self._out = open(file, 'w') if isinstance(file, str) else file
        self._hasInitialized = False
    
    def describeNextReport(self, simulation):
        """Get information about the next report."""
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, False, False, False, None)
    
    def report(self, simulation, state):
        """Generate a report at the current simulation step."""
        if not self._hasInitialized:
            # Write header in PLUMED format
            self._out.write(f"#! FIELDS time {' '.join(self._cv_names)}\n")
            
            # Write SET directives for CV ranges (useful for analysis)
            for cv_name in self._cv_names:
                if cv_name in self._cv_ranges:
                    min_val, max_val = self._cv_ranges[cv_name]
                    self._out.write(f"#! SET min_{cv_name} {min_val}\n")
                    self._out.write(f"#! SET max_{cv_name} {max_val}\n")
            
            self._hasInitialized = True
        
        # Get positions and time from state
        from openmm import unit
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        time_ps = state.getTime().value_in_unit(unit.picosecond)
        
        # Write CV values (for MullerBrown: x, y coordinates of single particle)
        values = [positions[0][i] for i in range(min(len(self._cv_names), positions.shape[1]))]
        
        self._out.write(f"{time_ps:12.6f}")
        for val in values:
            self._out.write(f" {val:12.6f}")
        self._out.write("\n")
        self._out.flush()  # Ensure data is written immediately
    
    def __del__(self):
        """Close file on deletion."""
        if hasattr(self, '_out'):
            self._out.close()


class OpenMMPlumedSampler(OpenMMLangevinSampler):
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


class MullerBrownSampler(OpenMMLangevinSampler):
    """OpenMM sampler for analytical Muller-Brown 2D potential."""
    
    def __init__(
        self,
        temperature: float,
        time_step: float,
        friction: float,
        simulation_steps: int,
        report_interval: int = 1000,
        initial_position: tuple[float, float] | None = None,
        cv_range: tuple[tuple[float, float], tuple[float, float]] = ((-1.5, 1.5), (-0.5, 2.5))
    ):
        """Initialize Muller-Brown sampler.
        
        Args:
            temperature: Temperature in Kelvin
            time_step: Integration timestep in ps
            friction: Friction coefficient in 1/ps
            simulation_steps: Number of MD steps per simulation
            report_interval: Frequency for data collection
            initial_position: Starting (x, y) position. If None, randomized per evaluation using seed.
            cv_range: CV bounds ((x_min, x_max), (y_min, y_max)) for randomized starting positions
        """
        super().__init__(temperature, time_step, friction, simulation_steps, report_interval)
        self.initial_position = initial_position
        self.cv_range = cv_range
    
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
    
    def _load_system(self, initial_pos):
        """Create 1-particle system with Muller-Brown potential.
        
        Args:
            initial_pos: (x, y) starting position for this simulation
        """
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
            np.array([[initial_pos[0], initial_pos[1], 0.0]]),
            self.unit.nanometers
        )
        
        return topology, system, positions
    
    def run(self, output_path: str, bias: Bias = None, seed: int = None):
        """Run Muller-Brown simulation with optional bias.
        
        Args:
            output_path: Directory for output files.
            bias: Optional Bias object.
            seed: Random seed for generating starting position (used when initial_position=None).
        """
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Determine starting position
        if self.initial_position is None:
            # Generate random starting position within CV range
            if seed is not None:
                rng = np.random.RandomState(seed)
            else:
                rng = np.random
            
            x = rng.uniform(self.cv_range[0][0], self.cv_range[0][1])
            y = rng.uniform(self.cv_range[1][0], self.cv_range[1][1])
            start_pos = (x, y)
        else:
            # Use fixed starting position
            start_pos = self.initial_position
        
        # Load system with determined starting position
        topology, system, positions = self._load_system(start_pos)
        
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
            ColvarReporter(
                f"{output_path}/COLVAR",
                self.report_interval,
                cv_names=['x', 'y'],
                cv_ranges={'x': (-1.5, 1.5), 'y': (-0.5, 2.5)}
            )
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
    
    def get_colvar_values(self, output_path: str) -> np.ndarray:
        """Extract CV values from COLVAR file.
        
        Args:
            output_path: Directory containing COLVAR file.
            
        Returns:
            Nx2 array of (x, y) positions from the trajectory.
        """
        colvar_file = Path(output_path) / "COLVAR"
        
        if not colvar_file.exists():
            raise FileNotFoundError(f"COLVAR file not found: {colvar_file}")
        
        # Parse COLVAR file (skip comment lines starting with #)
        data = []
        with open(colvar_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:  # time, x, y
                    data.append([float(parts[1]), float(parts[2])])
        
        return np.array(data)
