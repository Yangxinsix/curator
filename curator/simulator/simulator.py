import ase
from ase.md.md import MolecularDynamics
from abc import ABC, abstractmethod
from curator.data import read_trajectory, Trajectory
import os
import numpy as np

from .logger import BaseLogger

class BaseSimulator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def setup_atoms(self):
        pass

    @abstractmethod
    def configure_atoms(self):
        pass

    @abstractmethod
    def run(self):
        pass

class MDSimulator(BaseSimulator):
    def __init__(self, config, calculator, md_logger: BaseLogger, dynamics: MolecularDynamics):
        self.config = config
        self.calculator = calculator
        self.logger = md_logger.logger
        self.md_logger = md_logger
        self.dynamics = dynamics
        self.atoms = None

    def setup_atoms(self):
        if not os.path.isfile(self.config.init_traj):
            raise RuntimeError("Please provide valid initial data path!")
        
        images = read_trajectory(self.config.init_traj)
        start_index = np.random.choice(len(images)) if self.config.start_index is None else self.config.start_index
        self.logger.info(f'MD starts from No.{start_index} configuration in {self.config.init_traj}')
        self.atoms = images[start_index]
        self.atoms.wrap()

    def configure_atoms(self):
        if self.config.rattle:
            self.logger.debug(f'Rattle atoms with {self.config.rattle} eV/A^2')
            self.atoms.rattle(self.config.rattle)
        if self.config.fix_under:
            from ase.constraints import FixAtoms
            self.logger.debug(f'Fix atoms under {self.config.fix_under} A')
            cons = FixAtoms(mask=self.atoms.positions[:, 2] < self.config.fix_under) if self.config.fix_under else []
            self.atoms.set_constraint(cons)
        
        self.atoms.calc = self.calculator
        self.atoms.get_potential_energy()

    def run_md_step(self):
        def step_function():
            self.md_logger(self.atoms, self.config.print_step)
        return step_function

    def run(self):
        # setup and configure atoms
        self.setup_atoms()
        self.configure_atoms()

        # set initial velocities
        if self.config.initialize_velocities:
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.config.temperature)

        # set up MD dynamics
        self.dynamics = self.dynamics(self.atoms)
        self.dynamics.attach(self.run_md_step(), interval=self.config.print_step)
        traj = Trajectory('MD.traj', 'w', self.atoms)
        self.dynamics.attach(traj.write, interval=self.config.dump_step)

        # run MD
        self.dynamics.run(self.config.max_steps)