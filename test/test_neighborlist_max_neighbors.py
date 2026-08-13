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


def test_neighborlist_can_symmetrize_limited_edges():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.1, 0.0, 0.0]]
    )
    output = TorchNeighborList(
        cutoff=5.0,
        max_neighbors=1,
        symmetrize_edges=True,
        return_distance=True,
    )({properties.positions: positions})

    edges = output[properties.edge_idx]
    edge_diff = output[properties.edge_diff]
    edge_dist = output[properties.edge_dist]
    half = edges.shape[0] // 2

    assert torch.equal(edges[half:], edges[:half].flip(1))
    assert torch.equal(edge_diff[half:], -edge_diff[:half])
    assert torch.equal(edge_dist[half:], edge_dist[:half])
    assert torch.equal(output[properties.n_pairs], torch.tensor([edges.shape[0]]))


def test_neighborlist_does_not_symmetrize_by_default():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.1, 0.0, 0.0]]
    )
    output = TorchNeighborList(cutoff=5.0, max_neighbors=1)(
        {properties.positions: positions}
    )

    assert output[properties.edge_idx].shape[0] == 3


def test_symmetrizes_periodic_self_edges_and_cell_displacements():
    edge_info = {
        properties.edge_idx: torch.tensor([[0, 0], [0, 0]]),
        properties.edge_diff: torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]),
        properties.cell_displacements: torch.tensor(
            [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]
        ),
    }

    output = TorchNeighborList._symmetrize_edges(
        edge_info,
        torch.eye(3) * 2.0,
    )

    assert torch.equal(output[properties.edge_idx], torch.tensor([[0, 0], [0, 0]]))
    assert torch.equal(
        output[properties.edge_diff],
        torch.tensor([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    assert torch.equal(
        output[properties.cell_displacements],
        torch.tensor([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
