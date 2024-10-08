from abc import ABC, abstractmethod
from ase import Atoms, units
from ase.io import Trajectory
from typing import Optional, Union, List, Dict, Callable, Any
from .uncertainty import BaseUncertainty
import logging
import sys

class BaseLogger(ABC):
    # Abstract class for logging simulation results
    def __init__(self):
        pass

    def attach_uncertainty(self, uncertainty: Optional[BaseUncertainty]=None):
        self.uncertainty_calc = uncertainty
        if self.uncertainty_calc is not None:
            self.add_error_handler()

    def add_error_handler(self, filename: str='warning.log'):
        error_handler_exists = any(isinstance(handler, logging.FileHandler) and handler.baseFilename == filename 
                                   for handler in self.logger.handlers)
        if not error_handler_exists:
            errorHandler = logging.FileHandler(filename, mode='w')
            errorHandler.setLevel(logging.WARNING)
            errorHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
            self.logger.addHandler(errorHandler)

class MDLogger:
    """
    A logger for molecular dynamics simulations that mimics LAMMPS's output style.
    Allows users to specify which variables to log, similar to LAMMPS's thermo_style.
    """

    default_combinations = {
        'basic_energies': ['step', 'epot', 'ekin', 'etot'],
        'temp_pressure': ['step', 'temp', 'pressure'],
        'full_thermodynamics': ['step', 'epot', 'ekin', 'etot', 'temp', 'stress'],
        'structural_properties': ['step', 'volume', 'density'],
        'energy_per_atom': ['step', 'epot_per_atom', 'ekin_per_atom'],
        'dynamic_properties': ['step', 'temp', 'pressure', 'epot', 'ekin'],
    }

    def __init__(
        self,
        logfile: Optional[str] = None,
        variables: Optional[Union[str, List[str]]] = None,
        header: bool = True,
        per_atom: bool = False,
        mode: str = 'w',
        logger: Optional[logging.Logger] = None,
        loginterval: int = 1,
        custom_functions: Optional[Dict[str, Callable[[Atoms], Any]]] = None,
    ):
        """
        Initialize the logger.

        Args:
            logfile (str): Path to the log file. If None, logs to stdout.
            variables (List[str]): List of variable names to log.
            header (bool): Whether to print the header.
            per_atom (bool): Whether to print per-atom quantities.
            mode (str): File mode for the logfile ('w' for write, 'a' for append).
            logger (logging.Logger): Optional logger object.
            loginterval (int): Interval of steps between logs.
            custom_functions (Dict[str, Callable[[Atoms], Any]]): 
                Custom functions to calculate additional variables.
        """
        self.logfile = logfile
        self.loginterval = loginterval
        self.calls = 0
        self.per_atom = per_atom
        self.header = header
        self.mode = mode

        self.logger = logger if logger is not None else logging.getLogger(__name__)

        # Determine the variables to log
        if isinstance(variables, str) and variables in self.default_combinations:
            self.variables = self.default_combinations[variables]
        elif variables is None:
            self.variables = self.default_combinations['full_thermodynamics']
        else:
            self.variables = variables

        # Dictionary mapping variable names to their calculation functions
        self.variable_funcs = {
            'step': self.get_step,
            'time': self.get_time,
            'epot': self.get_epot,
            'ekin': self.get_ekin,
            'etot': self.get_etot,
            'epot_per_atom': self.get_epot_per_atom,
            'ekin_per_atom': self.get_ekin_per_atom,
            'temp': self.get_temp,
            'stress': self.get_stress,
            'volume': self.get_volume,
            'density': self.get_density,
            # cell 
            # 'cella': self.get_cella,
            # 'cellb': self.get_cellb,
            # 'cellc': self.get_cellc,
            
            # Add more default variables as needed
        }

        # Include custom functions if provided
        if custom_functions:
            self.variable_funcs.update(custom_functions)

        # Log header if required
        if header:
            self.log_header()

    def log_header(self):
        """Log the header with variable names."""
        header_line = ' '.join(f'{var:>15}' for var in self.variables)
        self.logger.info(header_line)

    def __call__(self, atoms: Atoms):
        """Log the variables at the current step."""
        self.calls += 1

        if self.calls % self.loginterval != 0:
            return

        values = []
        for var in self.variables:
            func = self.variable_funcs.get(var)
            if func:
                value = func(atoms)
                if isinstance(value, float):
                    values.append(f'{value:15.5f}')
                else:
                    values.append(f'{value:>15}')
            else:
                values.append(f'{"N/A":>15}')

        log_line = ''.join(values)
        self.logger.info(log_line)

    # Methods to calculate variables
    def get_step(self, atoms):
        """Get the current simulation step."""
        return self.calls

    def get_time(self, atoms):
        """Get the current simulation time."""
        return atoms.get_calculator().get_time() if hasattr(atoms.get_calculator(), 'get_time') else 0.0

    def get_epot(self, atoms):
        """Get the potential energy."""
        return atoms.get_potential_energy()

    def get_ekin(self, atoms):
        """Get the kinetic energy."""
        return atoms.get_kinetic_energy()

    def get_etot(self, atoms):
        """Get the total energy."""
        return self.get_epot(atoms) + self.get_ekin(atoms)

    def get_temp(self, atoms):
        """Get the temperature."""
        #TODO: try to access fixed atoms
        dof = 3 * atoms.get_global_number_of_atoms()
        ekin = self.get_ekin(atoms)
        return 2 * ekin / (dof * units.kB)

    def get_stress(self, atoms):
        """Get the pressure."""
        if hasattr(atoms, 'get_stress'):
            return atoms.get_stress()
        else:
            return 'N/A'

    def get_volume(self, atoms):
        """Get the volume of the simulation cell."""
        return atoms.get_volume()

    def get_density(self, atoms):
        return atoms.get_masses().sum() / atoms.get_volume()

    def get_epot_per_atom(self, atoms):
        """Get potential energy per atom."""
        return self.get_epot(atoms) / atoms.get_global_number_of_atoms()

    def get_ekin_per_atom(self, atoms):
        """Get kinetic energy per atom."""
        return self.get_ekin(atoms) / atoms.get_global_number_of_atoms()

    def get_uncertainty(self, atoms):
        pass

class MDLogger(BaseLogger):
    # Logger for molecular dynamics simulations
    def __init__(
            self,
            logger: Optional[logging.Logger]=None, 
            uncertainty: Optional[BaseUncertainty]=None,
            min_calls: int = 0,          # if uncertaint_calls > min_calls and meets stop criteria, just stop the simulation, else raise errors.
        ):
        """ Class to setup uncertainty method and print physical quantities in a simualtion.
        
        Args:
            logger (logging.Logger): logger object
            uncertainty (BaseUncertainty): uncertainty calculator

        """
        self.calls = 0
        self.min_calls = min_calls

        self.logger = logger if logger is not None else logging.getLogger(__name__)
        if uncertainty is not None:
            self.attach_uncertainty(uncertainty)

    def __call__(self, atoms: Atoms, print_step: int=1):
        """ Method to log physical quantities of a simulation.
        
        Args:
            atoms (ase.Atoms): Atoms object
            print_step (int): print step

        """
        self.log(atoms, print_step)    

    def log(self, atoms: Atoms, print_step: int=1):
        """ Method to log physical quantities of a simulation.
        
        Args:
            atoms (ase.Atoms): Atoms object
            print_step (int): print step

        """
        self.calls += 1
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB) / atoms.get_global_number_of_atoms()

        string = "Steps={:10d} Epot={:12.3f} Ekin={:12.3f} temperature={:8.2f} ".format(
            self.calls * print_step,
            epot,
            ekin,
            temp,
        )

        if self.uncertainty_calc is not None:
            check, uncertainty = self.uncertainty_calc(atoms, return_check=True)
            for key, value in uncertainty.items():
                string += f"{key}={value:12.3f} "

            # check uncertainty
            if check == 0:
                self.logger.info(string)
            elif check > 0:
                self.logger.warning(string)
                if check == 2:
                    if self.calls > self.min_calls:
                        sys.exit(0)
                    else:
                        raise RuntimeError(f"MD steps ({self.calls}) are fewer than minimum step ({self.min_calls})!")
        else:
            self.logger.info(string)