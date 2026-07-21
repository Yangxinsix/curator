import torch

from curator.data import properties
from curator.data._neighborlist import TorchNeighborList


def test_neighborlist_keeps_nearest_neighbors_per_center():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    )
    output = TorchNeighborList(cutoff=5.0, max_neighbors=2, return_distance=True)(
        {properties.positions: positions}
    )

    centers = output[properties.edge_idx][:, 0]
    assert torch.bincount(centers, minlength=4).max().item() == 2

    center_zero = centers == 0
    assert torch.allclose(
        torch.sort(output[properties.edge_dist][center_zero]).values,
        torch.tensor([1.0, 2.0]),
    )
