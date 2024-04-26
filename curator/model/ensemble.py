import torch
from torch import nn
from .base import NeuralNetworkPotential
from typing import List
from curator.data import properties

class EnsembleModel(nn.Module):
    """
    Ensemble model of PaiNN. This is used to get uncertainty informations.
    """
    def __init__(self, models: List[NeuralNetworkPotential]) -> None:
        """Ensemble model of PaiNN. This is used to get uncertainty informations.

        Args:
            models (List[NeuralNetworkPotential]): List of models to ensemble
        """
        super().__init__()
        self.models = nn.ModuleList([model for model in models])

    def forward(self, data: properties.Type) -> properties.Type:
        energy = []
        forces = []
        for model in self.models:
            out = model(data)
            energy.append(out[properties.energy].detach())
            forces.append(out[properties.forces].detach())
        
        energy = torch.stack(energy)
        forces = torch.stack(forces)
        f_scatter = torch.zeros(data[properties.n_atoms].shape[0], device=out[properties.energy].device)
        result_dict ={
            'energy': torch.mean(energy, dim=0),
            'forces': torch.mean(forces, dim=0),
            'e_var': torch.var(energy, dim=0),
            'e_sd': torch.std(energy, dim=0),
            'e_max': torch.max(energy, dim=0),
            'e_min': torch.min(energy, dim=0),
            'f_var': f_scatter.index_add(0, data[properties.image_idx], torch.var(forces, dim=0).mean(dim=1)) / data[properties.n_atoms],
        }
        
        result_dict['f_sd'] = result_dict['f_var'].sqrt()
        if 'energy' in data.keys():
            e_diff = result_dict['energy'] - data['energy']
            f_diff = result_dict['forces'] - data['forces']
            result_dict['forces'] = torch.split(result_dict['forces'], data[properties.n_atoms].tolist(), dim=0)
            result_dict['e_ae'] = torch.abs(e_diff)
            result_dict['e_se'] = torch.square(e_diff)
            result_dict['f_ae'] = f_scatter.index_add(0, data[properties.image_idx], torch.abs(f_diff).mean(dim=1)) / data[properties.n_atoms]
            result_dict['f_se'] = f_scatter.index_add(0, data[properties.image_idx], torch.square(f_diff).mean(dim=1)) / data[properties.n_atoms]

        return result_dict 
