from ase.calculators.calculator import Calculator, all_changes
from curator.data import AseDataReader, Transform
import ase
import numpy as np
import torch
from typing import Optional, List

class MLCalculator(Calculator):
    """ ML model calulator used for ASE applications """
    implemented_properties = ["energy", "forces", "stress"]
    def __init__(
        self,
        model: torch.nn.Module,
        cutoff: Optional[float] = None,
        compute_neighbor_list: bool = True,
        transforms: List[Transform] = [],
        energy_scale: float = 1.0,
        forces_scale: float = 1.0,
        stress_scale: float = 1.0,
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
        self.model.eval()
        self.model_device = next(model.parameters()).device
        if cutoff is None:
            for name, module in model.named_modules():
                if "representation" in name:
                    cutoff = module.cutoff
                    break
        
        assert cutoff is not None, "Valid cutoff value should be given or inferred from model!"
        self.ase_data_reader = AseDataReader(cutoff, compute_neighbor_list=compute_neighbor_list, transforms=transforms)
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
        self.stress_scale = stress_scale

    def _convert_to_cpu(self, tensor):
        tensor = tensor.detach().cpu()
        if tensor.ndimension() == 0:
            return tensor.item()
        elif tensor.numel() == 1:
            return tensor.item()
        else:
            return tensor.numpy()

    def _process_results(self, results):
        processed_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                processed_results[key] = self._convert_to_cpu(value)
            elif isinstance(value, dict):
                processed_results[key] = self._process_results(value)
            else:
                processed_results[key] = value
        return processed_results

    def calculate(
            self, 
            atoms: ase.Atoms =None, 
            properties: list = ["energy", "forces", "stress"], 
            system_changes: list = all_changes,
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
        self.results = self._process_results(self.model(model_inputs))

        # calculate stress if get virial
        if "virial" in self.results and "stress" not in self.results:
            self.results["stress"] = - self.results["virial"] / self.atoms.get_volume()

        # Convert outputs to calculator format
        if "energy" in self.results:
            self.results["energy"] *= self.energy_scale
        if "forces" in self.results:
            self.results["forces"] *= self.forces_scale
        if "stress" in self.results:
            self.results["stress"] *= self.stress_scale

class EnsembleCalculator(Calculator):
    """ Ensemble calulator for ML models used for ASE applications """
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        models: list,
        cutoff: Optional[float] = None,
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
        self.ase_data_reader = AseDataReader(cutoff if cutoff is not None else models[0].representation.cutoff)
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
