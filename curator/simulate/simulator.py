from ase.md.md import MolecularDynamics
from abc import ABC, abstractmethod
from curator.data import read_trajectory, Trajectory
import os
import numpy as np
from typing import Optional
from .uncertainty import BaseUncertainty

from .logger import BaseLogger

class BaseSimulator(ABC):
    def __init__(self):
        pass
    
    # read atoms from given path
    @abstractmethod
    def setup_atoms(self):
        pass
    
    # configure atoms with given parameters
    @abstractmethod
    def configure_atoms(self):
        pass

    # run simulation
    @abstractmethod
    def run(self):
        pass

class MDSimulator(BaseSimulator):
    def __init__(
            self, 
            init_traj: str,
            calculator,
            logger: BaseLogger,
            dynamics: MolecularDynamics,
            uncertainty: Optional[BaseUncertainty] = None,
            out_traj: str = 'MD.traj',
            start_index: int = -1,
            rattle: bool = False,
            fix_under: Optional[float] = None,
            print_step: int = 1,
            dump_step: int = 100,
            max_steps: int = 1000,
            initialize_velocities: bool = True,
            temperature: float = 298.15,
            *args,
            **kwargs,
        ):
        # initialize parameters
        self.init_traj = init_traj
        self.out_traj = out_traj
        self.start_index = start_index
        self.rattle = rattle
        self.fix_under = fix_under
        self.print_step = print_step
        self.dump_step = dump_step
        self.max_steps = max_steps
        self.initialize_velocities = initialize_velocities
        self.temperature = temperature

        # initialize objects
        self.calculator = calculator
        self.logger = logger.logger
        self.md_logger = logger
        if uncertainty is not None:
            if hasattr(self.md_logger, 'uncertainty_calc'):
                self.logger.warning("Warning! Uncertainty calculator will be overrided.")
            self.md_logger.attach_uncertainty(uncertainty)
            
        self.dynamics = dynamics
        self.atoms = None

    def setup_atoms(self):
        if not os.path.isfile(self.init_traj):
            raise RuntimeError("Please provide valid initial data path!")
        
        images = read_trajectory(self.init_traj)
        start_index = np.random.choice(len(images)) if self.start_index is None else self.start_index
        self.logger.info(f'MD starts from No.{start_index} configuration in {self.init_traj}')
        self.atoms = images[start_index]

    def configure_atoms(self):
        if self.rattle:
            self.logger.debug(f'Rattle atoms with {self.rattle} eV/Å^2')
            self.atoms.rattle(self.rattle)
        if self.fix_under is not None:
            from ase.constraints import FixAtoms
            self.logger.debug(f'Fix atoms under {self.fix_under} Å')
            cons = FixAtoms(mask=self.atoms.positions[:, 2] < self.fix_under) if self.fix_under else []
            self.atoms.set_constraint(cons)

        self.atoms.wrap()
        self.atoms.calc = self.calculator
        self.atoms.get_potential_energy()

    def run_md_step(self):
        def step_function():
            self.md_logger(self.atoms, self.print_step)
        return step_function

    def run(self):
        # setup and configure atoms
        self.setup_atoms()
        self.configure_atoms()

        # set initial velocities
        if self.initialize_velocities:
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.temperature)

        # set up MD dynamics
        self.dynamics = self.dynamics(self.atoms)
        self.dynamics.attach(self.run_md_step(), interval=self.print_step)
        traj = Trajectory(self.out_traj, 'w', self.atoms)
        self.dynamics.attach(traj.write, interval=self.dump_step)

        # run MD
        self.dynamics.run(self.max_steps)