import numpy as np
import ase
from typing import Dict
def ensemble(atoms: ase.Atom, threshold: str='forces_sd') -> Dict:  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy.
    
    Args:
        atoms (ase.Atom): Atoms object
        threshold (str, optional): [description]. Defaults to 'forces_sd'.
    
    Returns:
        uncertainty (dict): uncertainty dictionary
    """
    # Get ensemble results
    ensemble = atoms.calc.results['ensemble']

    # Define uncertanty dictionary    
    uncertainty = {}
    uncertainty['energy_var'] = ensemble['energy_var']
    uncertainty['force_var'] = np.mean(ensemble['forces_var'])
    uncertainty['forces_sd'] = np.mean(np.sqrt(ensemble['forces_var']))
    uncertainty['forces_l2_var'] = np.mean(ensemble['forces_l2_var'])

    # Set threshold
    uncertainty['threshold'] = uncertainty[threshold]
    #a.info['ensemble'] = ensemble
    return uncertainty

def MCDropout(atoms: ase.Atom) -> Dict:
    pass

def Mahalanobis(atoms):
    pass

class GetUncertainty:
    """Class to get uncertainty informations.
    
    """
    def __init__(self, method: str ='ensemble',threshold: str ='forces_sd') -> None:
        """Class to get uncertainty informations.
        Args:
            method (str): uncertainty method. Defaults to 'ensemble'.
            threshold (str): uncertainty threshold. Defaults to 'forces_sd'.
        """
        self.method = method
        self.threshold = threshold
    
    def __call__(self, atoms: ase.Atom)-> Dict: 
        """ Get uncertainty informations.

        Args:
            atoms (ase.Atom): Atoms object
        
        Returns:
            uncertainty (dict): uncertainty dictionary
        """

        # Get uncertainty results based on method
        if self.method == 'ensemble':
            uncertainty = ensemble(atoms,self.threshold)
        
        elif self.method == 'MCDropout':
            uncertainty = MCDropout(atoms,self.threshold)
        
        elif self.method == 'Mahalanobis':
            uncertainty = Mahalanobis(atoms,self.threshold)
        
        else:
            raise RuntimeError('Valid method should be provided!')

        # Set uncertainty dictionary to atoms.info and return uncertainty dictionary
        atoms.info['uncertainty'] = uncertainty
        return uncertainty
