from ase.calculators.calculator import Calculator, all_changes
from curator.data import AseDataReader, TypeMapper
import ase
import numpy as np
import torch


class MLCalculator(Calculator):
    """ ML model calulator used for ASE applications """
    implemented_properties = ["energy", "forces"]
    def __init__(
        self,
        model: torch.nn.Module,
        cutoff: float,
#        species,
        energy_scale: float =1.0,
        forces_scale: float =1.0,
#        stress_scale=1.0,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        """Provide a ML model instance to calculate various atomic properties using ASE.
        
        Args:
            model (torch.nn.Module): ML model 
            cutoff (float): cutoff radius
            energy_scale (float, optional): energy scale. Defaults to 1.0.
            forces_scale (float, optional): forces scale. Defaults to 1.0.
        """
        self.model = model
        self.model_device = next(model.parameters()).device
        self.ase_data_reader = AseDataReader(cutoff)#,transforms=[TypeMapper(species)])
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
#        self.stress_scale = stress_scale

    def calculate(
            self, 
            atoms: ase.Atoms =None, 
            properties: list =["energy"], 
            system_changes: list =all_changes
            ) -> None:
        """
        Calculate atomic properties using ML model.
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)
        if atoms is not None:
            self.atoms = atoms.copy()       

        # Convert atoms to model inputs
        model_inputs = self.ase_data_reader(self.atoms)
        model_inputs = {
            k: v.to(self.model_device) for (k, v) in model_inputs.items()
        }
        
        # Run model
        model_results = self.model(model_inputs)

        results = {}

        # Convert outputs to calculator format
        results["forces"] = (
            model_results["forces"].detach().cpu().numpy() * self.forces_scale
        )
        results["energy"] = (
            model_results["energy"][0].detach().cpu().numpy().item()
            * self.energy_scale
        )
        for k in model_results.keys():
            if k != "forces" and k != "energy":
                results[k] = model_results[k][0].detach().cpu().numpy().item()
    
        self.results = results

class EnsembleCalculator(Calculator):
    """ Ensemble calulator for ML models used for ASE applications """
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        models: list,
        cutoff: float,
        energy_scale: int =1.0,
        forces_scale: int =1.0,
#        stress_scale=1.0,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        """ Provide a list of ML models to calculate various atomic properties using ASE.

        Args:
            models (list): list of ML models 
            cutoff (float): cutoff radius
            energy_scale (float, optional): energy scale. Defaults to 1.0.
            forces_scale (float, optional): forces scale. Defaults to 1.0.
        
        """
        self.models = models
        if hasattr(models[0], 'parameters'):
            self.model_device = next(models[0].parameters()).device
        else:
            self.model_device = models[0]['device']
        self.cutoff = cutoff
        self.ase_data_reader = AseDataReader(cutoff)#,transforms=[TypeMapper(species)])
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
#        self.stress_scale = stress_scale

    def calculate(
            self, 
            atoms=None, 
            properties: list =["energy"], 
            system_changes: list =all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)
        if atoms is not None:
            self.atoms = atoms.copy()       

        # Convert atoms to model inputs
        model_inputs = self.ase_data_reader(self.atoms)
        model_inputs = {
            k: v.to(self.model_device) for (k, v) in model_inputs.items()
        }

        # Run models
        predictions = {'energy': [], 'forces': []}
        for model in self.models:
            model_results = model(model_inputs)
            predictions['energy'].append(model_results["energy"][0].detach().cpu().numpy().item() * self.energy_scale)
            predictions['forces'].append(model_results["forces"].detach().cpu().numpy() * self.forces_scale)

        # Convert outputs to calculator format
        results = {"energy": np.mean(predictions['energy'])}
        results["forces"] = np.mean(np.stack(predictions['forces']), axis=0)

        # Calculate ensemble variance
        ensemble = {
            'energy_var': np.var(predictions['energy']),
            'forces_var': np.var(np.stack(predictions['forces']), axis=0),
            'forces_l2_var': np.var(np.linalg.norm(predictions['forces'], axis=2), axis=0),
        }

        # Save ensemble results
        results['ensemble'] = ensemble

        self.results = results
