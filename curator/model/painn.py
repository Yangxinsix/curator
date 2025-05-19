import torch
from torch import nn
from typing import List, Optional, Dict, Type, Union
from functools import partial

from curator.data import (
    properties,
)
from curator.layer import (
    PainnMessage, 
    PainnUpdate,
    AtomwiseNN,
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
        readout: Union[AtomwiseNN, Type[AtomwiseNN], partial] = AtomwiseNN,
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
        if isinstance(readout, AtomwiseNN):
            self.readout = readout
        else:
            self.readout = readout(num_features)

    def forward(
        self, 
        data: properties.Type,
    ) -> properties.Type:
        # add mask for local interaction part
        edge_idx, edge_diff, edge_dist = data[properties.edge_idx], data[properties.edge_diff], data[properties.edge_dist]
        mask = edge_dist < self.cutoff
        data[properties.edge_idx], data[properties.edge_diff], data[properties.edge_dist] = edge_idx[mask], edge_diff[mask], edge_dist[mask]

        total_atoms = int(torch.sum(data[properties.n_atoms]))
        node_scalar = self.atom_embedding(data[properties.Z])
        node_vector = torch.zeros(
            (total_atoms, self.num_features * 3),
            device=edge_diff.device,
            dtype=edge_diff.dtype,
        )
        node_feat = torch.cat([node_scalar, node_vector], dim=-1)
        data[properties.node_embedding] = node_scalar        # store node embedding for some modules (charge equilibration)
        
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            node_feat = message_layer(
                node_feat,
                edge_idx,
                edge_dist,
                edge_diff,
            )
            node_feat = update_layer(node_feat)
        
        data[properties.node_feat] = node_feat
        data = self.readout(data)

        # restore neighbor list
        data[properties.edge_idx], data[properties.edge_diff], data[properties.edge_dist] = edge_idx, edge_diff, edge_dist

        return data