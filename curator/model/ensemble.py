import torch
from torch import nn
from .base import NeuralNetworkPotential
from typing import List, Dict, Union
from collections import defaultdict
from curator.data import properties
try:
    from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min
except ImportError:
    from curator.utils import scatter_add, scatter_mean, scatter_max, scatter_min

class EnsembleModel(nn.Module):
    """
    Ensemble model for evaluating uncertainties
    """
    def __init__(self, models: List[NeuralNetworkPotential], per_atom_uncertainty: bool=False) -> None:
        super().__init__()
        self.models = nn.ModuleList([model for model in models])
        self.compute_uncertainty = True if len(models) > 1 else False
        self.per_atom_uncertainty = per_atom_uncertainty
        self.model_outputs = []
        for model in self.models:
            for key in model.model_outputs:
                if key not in self.model_outputs:
                    self.model_outputs.append(key)

    def forward(self, data: properties.Type) -> properties.Type:
        model_outputs_lists: Dict[str, List[torch.Tensor]] = {}
        for model in self.models:
            out = model(data)
            for key in self.model_outputs:
                if key not in model_outputs_lists:
                    model_outputs_lists[key] = [out[key].detach()]
                else:
                    model_outputs_lists[key].append(out[key].detach())
        
        model_outputs_dict: Dict[str, torch.Tensor] = {}
        for k, v in model_outputs_lists.items():
            model_outputs_dict[k] = torch.stack(v)
        del model_outputs_lists

        result_dict: Dict[str, torch.Tensor] = {}
        for k, v in model_outputs_dict.items():
            result_dict[k] = v.mean(dim=0)

        if self.compute_uncertainty:
            if properties.image_idx not in data:
                data[properties.image_idx] = torch.zeros(data[properties.n_atoms].item(), dtype=data[properties.edge_idx].dtype, device=data[properties.edge_idx].device)
            result_dict[properties.e_max] = torch.max(model_outputs_dict[properties.energy]).unsqueeze(-1)
            result_dict[properties.e_min] = torch.min(model_outputs_dict[properties.energy]).unsqueeze(-1)
            result_dict[properties.e_var] = torch.var(model_outputs_dict[properties.energy], dim=0)
            result_dict[properties.e_sd] = torch.std(model_outputs_dict[properties.energy], dim=0)
            if self.per_atom_uncertainty:
                result_dict[properties.f_var] = torch.var(model_outputs_dict[properties.forces], dim=0).mean(dim=1)
            else:
                result_dict[properties.f_var] = scatter_mean(torch.var(model_outputs_dict[properties.forces], dim=0).mean(dim=1), data[properties.image_idx], dim=0)

            result_dict[properties.f_sd] = result_dict[properties.f_var].sqrt()
        
        if properties.energy in data:
            # calculate errors
            if properties.image_idx not in data:
                data[properties.image_idx] = torch.zeros(data[properties.n_atoms].item(), dtype=data[properties.edge_idx].dtype, device=data[properties.edge_idx].device)
            e_diff = result_dict[properties.energy] - data[properties.energy]
            f_diff = result_dict[properties.forces] - data[properties.forces]
            result_dict[properties.e_ae] = torch.abs(e_diff)
            result_dict[properties.e_se] = torch.square(e_diff)
            result_dict[properties.f_ae] = scatter_mean(torch.abs(f_diff).mean(dim=1), data[properties.image_idx], dim=0)
            result_dict[properties.f_se] = scatter_mean(torch.square(f_diff).mean(dim=1), data[properties.image_idx], dim=0)
            # currently torch scatter does not support jit script these two operations
            # result_dict[properties.f_maxe], _ = scatter_max(f_diff.square().sum(dim=1).sqrt(), data[properties.image_idx], dim=0)
            # result_dict[properties.f_mine], _ = scatter_min(f_diff.square().sum(dim=1).sqrt(), data[properties.image_idx], dim=0)

        return result_dict

class DropoutModel(nn.Module):
    pass