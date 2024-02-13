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
        # normalization: bool=True,
        # scale: float=1.0,
        # shift: float=1.0,
        # atomwise_normalization: bool=True,
        # compute_neighborlist: bool=False,
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
            normalization (bool, optional): Normalization. Defaults to True.
            scale (float, optional): Scale. Defaults to 1.0.
            shift (float, optional): Shift. Defaults to 1.0.
            atomwise_normalization (bool, optional): Atomwise normalization. Defaults to True.
            compute_neighborlist (bool, optional): Compute neighborlist. Defaults to False.
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
        
        num_atoms = data[properties.n_atoms]
        edge = data[properties.edge_idx]
        edge_diff = data[properties.edge_diff]
        edge_dist = data[properties.edge_dist]  
        total_atoms = int(torch.sum(num_atoms))
        
        node_scalar = self.atom_embedding(data[properties.Z])
        node_vector = torch.zeros(
            (total_atoms, 3, self.num_features),
            device=edge_diff.device,
            dtype=edge_diff.dtype,
        )
        
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            node_scalar, node_vector = message_layer(node_scalar, node_vector, edge, edge_diff, edge_dist)
            node_scalar, node_vector = update_layer(node_scalar, node_vector)
        
        data[properties.node_feat] = node_scalar
        node_scalar = self.readout_mlp(node_scalar)
        node_scalar.squeeze_()
        data[properties.atomic_energy] = node_scalar              # it can be any atomic properties like atomic charge although it is called atomic energy
        
        return data