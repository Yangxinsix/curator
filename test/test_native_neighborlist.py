from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

pytest.importorskip("curator.native._neighbors")

from curator.data import BatchNeighborList, NativeNeighborList, properties


def _edge_keys(edge_idx: torch.Tensor, edge_diff: torch.Tensor) -> set[tuple[int, int, tuple[float, float, float]]]:
    return {
        (int(i), int(j), tuple(np.round(diff.detach().cpu().numpy(), 8)))
        for (i, j), diff in zip(edge_idx.detach().cpu().numpy(), edge_diff, strict=True)
    }


def test_native_neighbor_list_returns_directed_edges_without_pbc():
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    data = {properties.positions: positions}

    output = NativeNeighborList(0.85, return_distance=True)(data)

    assert set(map(tuple, output[properties.edge_idx].tolist())) == {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    }
    assert torch.allclose(output[properties.edge_dist], torch.linalg.norm(output[properties.edge_diff], dim=1))


def test_native_neighbor_list_matches_ase_with_cell_displacements():
    cell = np.diag([4.0, 4.0, 4.0])
    positions = np.array(
        [
            [0.15, 2.0, 2.0],
            [3.85, 2.0, 2.0],
            [2.0, 0.2, 2.0],
            [2.0, 3.7, 2.0],
        ],
        dtype=np.float64,
    )
    atoms = Atoms("OOOO", positions=positions, cell=cell, pbc=True)
    ase_i, ase_j, ase_diff = neighbor_list("ijD", atoms, 0.55, self_interaction=False)
    expected = {
        (int(i), int(j), tuple(np.round(diff, 8)))
        for i, j, diff in zip(ase_i, ase_j, ase_diff, strict=True)
    }
    data = {
        properties.positions: torch.tensor(positions, dtype=torch.float64),
        properties.cell: torch.tensor(cell, dtype=torch.float64),
    }

    output = NativeNeighborList(
        0.55,
        return_distance=True,
        return_cell_displacements=True,
        num_threads=2,
    )(data)

    reconstructed = (
        data[properties.positions][output[properties.edge_idx][:, 1]]
        - data[properties.positions][output[properties.edge_idx][:, 0]]
        + output[properties.cell_displacements]
    )
    assert torch.allclose(output[properties.edge_diff], reconstructed)
    assert _edge_keys(output[properties.edge_idx], output[properties.edge_diff]) == expected


def test_batch_neighbor_list_accepts_native_option():
    data = {
        properties.positions: torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.4, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        properties.n_atoms: torch.tensor([2, 2], dtype=torch.long),
    }

    output = BatchNeighborList(0.85, neighbor_list="Native")(data)

    assert set(map(tuple, output[properties.edge_idx].tolist())) == {
        (0, 1),
        (1, 0),
        (2, 3),
        (3, 2),
    }
