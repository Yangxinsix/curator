import torch
from torch import nn
from typing import Optional, Union, Sequence
from curator.data import properties
try:
    from torch_scatter import scatter_add, scatter_mean
except ImportError:
    from curator.utils import scatter_add, scatter_mean
import math

class AtomwiseReduce(nn.Module):
    def __init__(
        self,
        output_key: str = "energy",
        per_atom_output: bool = False,
        aggregation_mode: str = "sum",     # should be sum or mean
    ) -> None:
        super().__init__()
        self.model_outputs = [output_key]
        if per_atom_output:
            self.model_outputs.append(output_key + '_per_atom')
        self.aggregation_mode = aggregation_mode
        self.output_key = output_key
        self.per_atom_output = per_atom_output
    
    def forward(self, data: properties.Type) -> properties.Type:
        if self.aggregation_mode == "sum":
            y = scatter_add(data[properties.atomic_energy], data[properties.image_idx], dim=0)
        elif self.aggregation_mode == "mean":
            y = scatter_mean(data[properties.atomic_energy], data[properties.image_idx], dim=0)
        else:
            # using scatter_add by default
            y = scatter_add(data[properties.atomic_energy], data[properties.image_idx], dim=0)
        
        data[self.output_key] = y
        if self.per_atom_output:
            data[self.output_key + '_per_atom'] = data[properties.atomic_energy]
        
        return data