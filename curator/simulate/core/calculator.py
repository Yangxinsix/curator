from ase.calculators.calculator import Calculator, all_changes
from curator.data import AseDataReader, Transform
from curator.utils import load_models
from curator.layer import find_layer_by_name_recursive
from curator.model import EnsembleModel
import ase
import numpy as np
import torch
from typing import Optional, List, Union
from curator.interface.plumed import Plumed

class MLCalculator(Calculator):
    """ ML model calulator used for ASE applications """
    implemented_properties = ["energy", "forces", "stress"]
    def __init__(
        self,
        model: Union[torch.nn.Module, List[str], str, List[torch.nn.Module]],
        cutoff: Optional[float] = None,
        compute_neighbor_list: bool = True,
        transforms: List[Transform] = [],
        energy_scale: float = 1.0,
        forces_scale: float = 1.0,
        stress_scale: float = 1.0,
        plumed_bias: Optional[Plumed] = None,
        device=None,
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
        
        model_like = load_models(model, device=device)
        self.model = EnsembleModel(model_like) if len(model_like) > 1 else model_like[0]
        self.model.eval()
        self.model_device = next(self.model.parameters()).device
        self.model_dtype = next(self.model.parameters()).dtype
        if cutoff is None:
            cutoff = find_layer_by_name_recursive(self.model, 'cutoff')
        
        assert cutoff is not None, "Valid cutoff value should be given or inferred from model!"
        self.ase_data_reader = AseDataReader(cutoff, compute_neighbor_list=compute_neighbor_list, transforms=transforms)
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
        self.stress_scale = stress_scale
        self.plumed_bias = plumed_bias

    def _convert_to_cpu(self, tensor):
        tensor = tensor.detach().cpu()
        if tensor.ndimension() == 0:
            return tensor.item()
        elif tensor.numel() == 1:
            return tensor.item()
        else:
            return tensor.squeeze().numpy()

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
            k: v.to(self.model_device, dtype=self.model_dtype)
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
            else (v.to(self.model_device) if isinstance(v, torch.Tensor) else v)
            for (k, v) in model_inputs.items()
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

        if self.plumed_bias is not None:
            self._apply_plumed_bias()

    def _apply_plumed_bias(self) -> None:
        if "energy" not in self.results or "forces" not in self.results:
            return
        energy = float(self.results["energy"])
        forces = np.asarray(self.results["forces"])
        try:
            e_new, f_new = self.plumed_bias.apply_atoms(self.atoms, energy, forces)
        except Exception:
            return
        self.results["energy"] = float(np.asarray(e_new).reshape(-1)[0])
        self.results["forces"] = np.asarray(f_new)

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
            self.model_dtype = next(models[0].parameters()).dtype
        else:
            self.model_device = models[0]['device']
            self.model_dtype = None
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
            k: v.to(self.model_device, dtype=self.model_dtype)
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v) and self.model_dtype is not None
            else (v.to(self.model_device) if isinstance(v, torch.Tensor) else v)
            for (k, v) in model_inputs.items()
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
