"""Finite-system cutoff edges from occupied cells, in Curator's original edge order."""
from itertools import product
from math import floor, isfinite

import numpy as np
import torch

from curator.data import properties
from curator.data._neighborlist import TorchNeighborList
from curator.layer._pairwise_distance import PairwiseDistance


_OFFSETS = tuple(product((-1, 0, 1), repeat=3))


def _candidate_edges(positions, cutoff):
    """Return flat candidate pairs ordered first by source and then by target."""
    cells = {}
    atom_cells = []
    for atom, (x, y, z) in enumerate(positions):
        key = (floor(x / cutoff), floor(y / cutoff), floor(z / cutoff))
        cells.setdefault(key, []).append(atom)
        atom_cells.append(key)

    # Reuse the same 27-bucket lookup for every atom in each occupied cell.
    neighbors = {}
    for x, y, z in cells:
        atoms = []
        for dx, dy, dz in _OFFSETS:
            atoms.extend(cells.get((x + dx, y + dy, z + dz), ()))
        neighbors[x, y, z] = np.sort(np.asarray(atoms, dtype=np.int64), kind="stable")

    targets = [neighbors[key] for key in atom_cells]
    lengths = np.fromiter(map(len, targets), dtype=np.intp, count=len(targets))
    sources = np.repeat(np.arange(len(targets), dtype=np.int64), lengths)
    return np.column_stack((sources, np.concatenate(targets)))


class FiniteCellNeighborList(TorchNeighborList):
    """Use sparse occupied cells for nonperiodic coordinates; retain the periodic implementation."""

    def _simple_neighbor_list(self, pos):
        if len(pos) < 2:
            edge = torch.empty((0, 2), dtype=torch.long, device=pos.device)
        else:
            with torch.no_grad():
                # Only topology leaves the device. Test strict cutoff distances in the
                # input tensor's precision and evaluate final vectors from the live tensor.
                xyz = pos.detach().to(device="cpu", dtype=torch.float64).tolist()
                pairs = _candidate_edges(xyz, self.cutoff)
                candidates = torch.from_numpy(pairs).to(pos.device)
                source, target = candidates.unbind(1)
                distance = torch.linalg.vector_norm(pos[target] - pos[source], dim=1)
                edge = candidates[(source != target) & (distance < self.cutoff)]
        return {properties.edge_idx: edge,
                properties.edge_diff: pos[edge[:, 1]] - pos[edge[:, 0]]}


def install_linear_neighbors(model):
    """Install on the assembled MACE model; preserve parameters, buffers, and periodic settings."""
    lists = [block.batch_nl.neighbor_list for block in model.input_modules
             if isinstance(block, PairwiseDistance) and block.batch_nl is not None]
    if not lists or any(type(item) not in (TorchNeighborList, FiniteCellNeighborList) for item in lists):
        raise ValueError("Expected Curator's Torch neighbor-list input for this MACE model")
    if any(not isfinite(item.cutoff) or item.cutoff <= 0 for item in lists):
        raise ValueError("Cutoff must be finite and positive")
    for item in lists:
        item.__class__ = FiniteCellNeighborList
    return model
