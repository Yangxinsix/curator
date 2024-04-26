import torch
from torch import nn
from typing import List, Optional, Dict

from curator.data import (
    properties,
)
from curator.layer import (
    PainnMessage, 
    PainnUpdate, 
)


class PainnModel(nn.Module):
    """PainnModel without edge updating"""
    def __init__(
        self, 
        num_interactions: int, 
        num_features: int,
        cutoff: float,
        num_basis: int = 20,
        cutoff_fn: Optional[nn.Module]=None,
        radial_basis: Optional[nn.Module]=None,
        **kwargs,
    ):
        """PainnModel without edge updating

        Args:
            num_interactions (int): Number of interaction blocks
            num_features (int): Number of features
            cutoff (float): Cutoff radius
            num_basis (int, optional): Number of radial basis. Defaults to 20.
            cutoff_fn (Optional[nn.Module], optional): Cutoff function. Defaults to None.
            radial_basis (Optional[nn.Module], optional): Radial basis. Defaults to None.
        """
        super().__init__()
        
        num_embedding = 119   # number of all elements
        self.cutoff = cutoff
        self.num_interactions = num_interactions
        self.num_features = num_features
        self.num_basis = num_basis

        # Setup atom embeddings
        self.atom_embedding = nn.Embedding(num_embedding, num_features)

        # Setup message-passing layers
        self.message_layers = nn.ModuleList(
            [
                PainnMessage(self.num_features, self.num_basis, self.cutoff, cutoff_fn, radial_basis)
                for _ in range(self.num_interactions)
            ]
        )
        self.update_layers = nn.ModuleList(
            [
                PainnUpdate(self.num_features)
                for _ in range(self.num_interactions)
            ]            
        )
        
        # Setup readout function
        self.readout_mlp = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.SiLU(),
            nn.Linear(self.num_features, 1),
        )

    def forward(
        self, 
        data: properties.Type,
    ) -> properties.Type:
        total_atoms = int(torch.sum(data[properties.n_atoms]))
        data[properties.node_feat] = self.atom_embedding(data[properties.Z])
        data[properties.node_vect] = torch.zeros(
            (total_atoms, 3, self.num_features),
            device=data[properties.edge_diff].device,
            dtype=data[properties.edge_diff].dtype,
        )
        
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            data = message_layer(data)
            data = update_layer(data)
        
        # it can be any atomic properties like atomic charge although it is called atomic energy
        data[properties.atomic_energy] = self.readout_mlp(data[properties.node_feat]).squeeze()
        
        return data