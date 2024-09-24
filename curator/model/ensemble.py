import torch
from torch import nn
from .base import NeuralNetworkPotential
from typing import List
from collections import defaultdict
from curator.data import properties
from curator.utils import scatter_add, scatter_mean, scatter_max, scatter_min

class EnsembleModel(nn.Module):
    """
    Ensemble model for evaluating uncertainties
    """
    def __init__(self, models: List[NeuralNetworkPotential]) -> None:
        super().__init__()
        self.models = nn.ModuleList([model for model in models])
        self.compute_uncertainty = True if len(models) > 1 else False

    def forward(self, data: properties.Type) -> properties.Type:
        model_outputs = defaultdict(list)
        for model in self.models:
            out = model(data)
            for key in model.model_outputs:
                model_outputs[key].append(out[key].detach())
        
        for k, v in model_outputs.items():
            model_outputs[k] = torch.stack(v)

        result_dict = {}
        for k, v in model_outputs.items():
            result_dict[k] = v.mean(dim=0)

        if self.compute_uncertainty:
            uncertainty = {
                properties.e_max: torch.max(model_outputs[properties.energy]).unsqueeze(-1),
                properties.e_min: torch.min(model_outputs[properties.energy]).unsqueeze(-1),
                properties.e_var: torch.var(model_outputs[properties.energy], dim=0),
                properties.e_sd: torch.std(model_outputs[properties.energy], dim=0),
                properties.f_var: scatter_mean(torch.var(model_outputs[properties.forces], dim=0).mean(dim=1), data[properties.image_idx], dim=0),
            }
            uncertainty[properties.f_sd] = uncertainty[properties.f_var].sqrt()
            result_dict[properties.uncertainty] = uncertainty
        
        if properties.energy in data:
            # calculate errors
            e_diff = result_dict[properties.energy] - data[properties.energy]
            f_diff = result_dict[properties.forces] - data[properties.forces]
            error = {
                properties.e_ae: torch.abs(e_diff),
                properties.e_se: torch.square(e_diff),
                properties.f_ae: scatter_mean(torch.abs(f_diff).mean(dim=1), data[properties.image_idx], dim=0),
                properties.f_se: scatter_mean(torch.square(f_diff).mean(dim=1), data[properties.image_idx], dim=0),
            }

            # TODO: wait for torch_scatter.
            error[properties.f_maxe] = scatter_max(f_diff.norm(dim=1), data[properties.image_idx], dim=0)
            error[properties.f_mine] = scatter_min(f_diff.norm(dim=1), data[properties.image_idx], dim=0)
            result_dict[properties.error] = error

        return result_dict

class DropoutModel(nn.Module):
    pass