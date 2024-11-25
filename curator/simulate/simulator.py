from ase.md.md import MolecularDynamics
from abc import ABC, abstractmethod
from curator.data import read_trajectory, Trajectory
import os, shutil, sys
import numpy as np
from typing import Optional, Union, List, Dict
from .uncertainty import BaseUncertainty
import logging
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

class LammpsSimulator(BaseSimulator):
    def __init__(
        self,
        init_traj: str,
        start_index: int = -1,
        out_traj: str = 'MD.traj',
        uncertain_traj: Optional[str] = 'warning_struct.traj',
        lammps_input: str = 'in.lammps',
        masses: bool=True,
        units: str='metal',
        atom_style: str='atomic',
        lammps_output: Union[str, List[str]] = 'dump.lammps',
        uncertain_output: Union[str, List[str], None] = None, 
        specorder: Optional[List[int]] = None,       # type to species mapping
        shell_commands: str = 'lmp -in in.lammps',
        *args,
        **kwargs,
    ):
        super().__init__()
        self.init_traj = init_traj
        self.start_index = start_index
        self.out_traj = out_traj
        self.uncertain_traj = uncertain_traj
        self.masses = masses
        self.units = units
        self.atom_style = atom_style
        self.lammps_input = lammps_input
        self.lammps_output = lammps_output
        self.shell_commands = shell_commands
        self.uncertain_output = uncertain_output
        self.specorder = specorder
        
        self.logger = logging.getLogger(__name__)

    def setup_atoms(self):
        if not os.path.isfile(self.init_traj):
            raise RuntimeError("Please provide valid initial data path!")
        
        images = read_trajectory(self.init_traj)
        start_index = np.random.choice(len(images)) if self.start_index is None else self.start_index
        self.logger.info(f'Simulation starts from No.{start_index} configuration in {self.init_traj}')
        self.atoms = images[start_index]

    def configure_atoms(self):
        from ase.io import write
        if not os.path.isfile(self.lammps_input):
            raise RuntimeError(f"Please provide {self.lammps_input} for running Lammps!")
        else:
            shutil.copy(self.lammps_input, './in.lammps')
        write('in.data', self.atoms, format='lammps-data', masses=self.masses, units=self.units, atom_style=self.atom_style)

    def run(self):
        from ase.io import write
        from lammps import lammps

        self.setup_atoms()
        self.configure_atoms()

        try:
            lmp = lammps()
            lmp.file('in.lammps')
        except:
            self.logger.info('Running LAMMPS from python failed! Try to run from CMD.')

            try:
                import subprocess
                # TODO: redirect output to stdout and log file
                # for handler in self.logger.handlers:
                #     if isinstance(logging.FileHandler, handler):
                #         log_path = handler.baseFilename

                # proc = subprocess.Popen(self.shell_commands, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                # with open(log_path, 'a') as f:
                #     for line in proc.stdout:
                #         sys.stdout.write(line)
                #         f.write(line)
                subprocess.run(self.shell_commands, shell=True)
            except:
                self.logger.info('Running LAMMPS from CMD failed! Check your simulation!')
        finally:
            images = read_trajectory(self.lammps_output, specorder=self.specorder)
            write(self.out_traj, images)
            self.logger.info('Saving LAMMPS simulation trajectory to {}'.format(self.out_traj))
            if self.uncertain_output is not None:
                uncertain_images = read_trajectory(self.uncertain_output, specorder=self.specorder)
                write(self.uncertain_traj, uncertain_images)
                self.logger.info('Saving uncertain simulation trajectory to {}'.format(self.uncertain_traj))

class MDSimulator(BaseSimulator):
    def __init__(
            self, 
            init_traj: str,
            calculator,
            model,
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
        self.calculator = calculator(model=model)           # partial instantiation
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