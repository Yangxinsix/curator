import torch
from torch import nn
from .base import NeuralNetworkPotential
from typing import List
from curator.data import properties
from curator.utils import scatter_add, scatter_mean

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
        result_dict ={
            'energy': torch.mean(energy, dim=0),
            'forces': torch.mean(forces, dim=0),
            'e_var': torch.var(energy, dim=0),
            'e_sd': torch.std(energy, dim=0),
            'e_max': torch.max(energy, dim=0),
            'e_min': torch.min(energy, dim=0),
            'f_var': scatter_mean(torch.var(forces, dim=0).mean(dim=1), data[properties.image_idx], dim=0)
        }
        
        result_dict['f_sd'] = result_dict['f_var'].sqrt()
        if 'energy' in data.keys():
            e_diff = result_dict['energy'] - data['energy']
            f_diff = result_dict['forces'] - data['forces']
            result_dict['forces'] = torch.split(result_dict['forces'], data[properties.n_atoms].tolist(), dim=0)
            result_dict['e_ae'] = torch.abs(e_diff)
            result_dict['e_se'] = torch.square(e_diff)
            result_dict['f_ae'] = scatter_mean(torch.abs(f_diff).mean(dim=1), data[properties.image_idx], 0)
            result_dict['f_se'] = scatter_mean(torch.square(f_diff).mean(dim=1), data[properties.image_idx], 0)

        return result_dict 
