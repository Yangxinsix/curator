"keys to access properties of structures"

from typing import Final, Dict, Set
import torch

Type = Dict[str, torch.Tensor]

# basic properties
atomic_numbers: Final[str] = "_atomic_number"
Z: Final[str] = atomic_numbers
positions: Final[str] = "_positions"
R: Final[str] = positions
cell: Final[str] = "_cell"
n_atoms : Final[str] = "_n_atoms"
n_types : Final[str] = "_n_types"              # read from config file, useful for onehot embedding in nequip
atomic_types: Final[str] = "_atomic_types"     # map chemical symbols to numbers
symbols: Final[str] = "_symbols"
image_idx: Final[str] = "_image_index"             # image index of atoms in a batch

# neighbor list related properties
edge_idx: Final[str] = "_edge_index"             # index of i (center atoms), j (neighboring atoms)
edge_diff: Final[str] = "_edge_difference"           # R_j - R_i
edge_dist: Final[str] = "_edge_distance"           # distance between R_i and R_j
n_pairs: Final[str] = "_n_pairs"               # number of pairs
cell_displacements: Final[str] = "_cell_displacements"     # cell displacements used to reconstruct neighbor list for gradients calculation

# chemical properties
energy: Final[str] = "energy"
forces: Final[str] = "forces"
strain: Final[str] = "strain"
stress: Final[str] = "stress"
virial: Final[str] = "virial"
atomic_energy: Final[str] = "atomic_energy"
energy_uncertainty: Final[str] = "energy_uncertainty"
forces_uncertainty: Final[str] = "forces_uncertainty"

# node and edge feature keys (for nequip)
edge_diff_embedding: Final[str] = "_edge_diff_embedding"
edge_dist_embedding: Final[str] = "_edge_dist_embedding"
node_attr: Final[str] = "_node_attribute"
node_feat: Final[str] = "_node_feature"
node_vect: Final[str] = "_node_vector"
sc: Final[str] = "_skip_connection"



_DEFAULT_INDEX_FIELDS: Set[str] = {
    image_idx,
    edge_idx,
    atomic_types,
}

_DEFAULT_NODE_FIELDS: Set[str] = {
    positions,
    node_feat,
    node_attr,
    symbols,
    atomic_numbers,
    atomic_types,
    atomic_energy,
    forces,
    image_idx,
}

_DEFAULT_EDGE_FIELDS: Set[str] = {
    edge_diff,
    edge_dist,
    edge_diff_embedding,
    edge_dist_embedding,
    cell_displacements,
}

_DEFAULT_GRAPH_FIELDS: Set[str] = {
    energy,
    stress,
    strain,
    virial,
    cell,
    n_atoms,
    n_pairs,
    energy_uncertainty,
    forces_uncertainty,
}

_NODE_FIELDS: Set[str] = set(_DEFAULT_NODE_FIELDS)
_EDGE_FIELDS: Set[str] = set(_DEFAULT_EDGE_FIELDS)
_GRAPH_FIELDS: Set[str] = set(_DEFAULT_GRAPH_FIELDS)
_INDEX_FIELDS: Set[str] = set(_DEFAULT_INDEX_FIELDS)