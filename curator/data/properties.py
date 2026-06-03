"keys to access properties of structures"

from typing import Final, Dict, Set, List, Optional, Union, Literal
import torch
from dataclasses import dataclass

Type = Dict[str, torch.Tensor]

# basic properties
atomic_numbers: Final[str] = "atomic_numbers"
Z: Final[str] = atomic_numbers
positions: Final[str] = "positions"
pbc: Final[str] = "pbc"                       # periodic boundary conditions
R: Final[str] = positions
cell: Final[str] = "cell"
n_atoms : Final[str] = "n_atoms"
n_types : Final[str] = "_n_types"              # read from config file, useful for onehot embedding in nequip
atomic_types: Final[str] = "_atomic_types"     # map chemical symbols to numbers
symbols: Final[str] = "symbols"
image_idx: Final[str] = "_image_index"             # image index of atoms in a batch
domain: Final[str] = "_domain"                     # domain id per-structure (batch dimension)
domain_atom: Final[str] = "_domain_atom"           # domain id per-atom (node dimension)

# neighbor list related properties
edge_idx: Final[str] = "_edge_index"             # index of i (center atoms), j (neighboring atoms)
edge_diff: Final[str] = "_edge_difference"           # R_j - R_i
edge_dist: Final[str] = "_edge_distance"           # distance between R_i and R_j
n_pairs: Final[str] = "_n_pairs"               # number of pairs
cell_displacements: Final[str] = "_cell_displacements"     # cell displacements used to reconstruct neighbor list for gradients calculation

# chemical properties
energy: Final[str] = "energy"
forces: Final[str] = "forces"
energy_hessian: Final[str] = "energy_hessian"
energy_hessian_sampled: Final[str] = "energy_hessian_sampled"
energy_hessian_sample_indices: Final[str] = "energy_hessian_sample_indices"
energy_hessian_projected: Final[str] = "energy_hessian_projected"
energy_hessian_probe_vectors: Final[str] = "energy_hessian_probe_vectors"
edge_forces: Final[str] = "edge_forces"
strain: Final[str] = "strain"
stress: Final[str] = "stress"
virial: Final[str] = "virial"
total_charge: Final[str] = "total_charge"
atomic_charge: Final[str] = "atomic_charge"
dipole: Final[str] = "dipole"
total_magmom: Final[str] = "total_magmom"
atomic_energy: Final[str] = "atomic_energy"
atomic_energy_sd: Final[str] = "atomic_energy_sd"
fermi_level: Final[str] = "fermi_level"
ewald_energy: Final[str] = "ewald_energy"
ewald_forces: Final[str] = "ewald_forces"
residual_forces: Final[str] = "residual_forces"     # residual forces from chi and hardness, this is meaningless

# uncertainties
e_var: Final[str] = "energy_var"      # energy variance
e_sd: Final[str] = "energy_sd"        # energy standard deviation
e_max: Final[str] = "energy_max"      # energy maximum
e_min: Final[str] = "energy_min"      # energy minimum
e_ae: Final[str] = "energy_ae"        # energy absolute error
e_se: Final[str] = "energy_se"        # energy standard error

f_var: Final[str] = "force_var"      # forces variance
f_sd: Final[str] = "force_sd"        # forces standard deviation
force_sd_per_atom: Final[str] = "force_sd_per_atom"
f_max: Final[str] = "force_max"      # forces maximum
f_min: Final[str] = "force_min"      # forces minimum
f_ae: Final[str] = "force_ae"        # forces absolute error
f_se: Final[str] = "force_se"        # forces standard error
f_maxe: Final[str] = "force_maxe"    # forces maximum error
f_mine: Final[str] = "force_mine"    # forces minimum error

uncertainty: Final[str] = "uncertainties"
error: Final[str] = "errors"
energy_uncertainty: Final[str] = "energy_uncertainty"
forces_uncertainty: Final[str] = "forces_uncertainty"

# node and edge feature keys (for nequip and mace)
edge_diff_embedding: Final[str] = "_edge_diff_embedding"        # this will not change during forward once generated
edge_dist_embedding: Final[str] = "_edge_dist_embedding"        # this will not change during forward once generated
node_embedding: Final[str] = "_node_embedding"                  # this is typically generated from node embedding block in the model, and will not change once generated
node_attr: Final[str] = "_node_attribute"                       # this will not change during forward once generated
node_feat: Final[str] = "_node_feature"
node_vect: Final[str] = "_node_vector"
sc: Final[str] = "_skip_connection"

# features and gradients (for active learning and uncertainty quantification)
feature: Final[str] = "feature"
gradient: Final[str] = "gradient"
maha_dist: Final[str] = "maha_dist"         # mahalanobis distance
maha_dist_per_atom: Final[str] = "maha_dist_per_atom"  # per-atom mahalanobis distance

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
    atomic_energy_sd,
    forces,
    force_sd_per_atom,
    maha_dist_per_atom,
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
    e_var,
    e_sd,
    e_max,
    e_min,
    e_ae,
    e_se,
    f_var,
    f_sd,
    f_max,
    f_min,
    f_ae,
    f_se,
    energy_uncertainty,
    forces_uncertainty,
}

_NODE_FIELDS: Set[str] = set(_DEFAULT_NODE_FIELDS)
_EDGE_FIELDS: Set[str] = set(_DEFAULT_EDGE_FIELDS)
_GRAPH_FIELDS: Set[str] = set(_DEFAULT_GRAPH_FIELDS)
_INDEX_FIELDS: Set[str] = set(_DEFAULT_INDEX_FIELDS)

# activation functions
activation_fn = {
    "silu": torch.nn.SiLU(),
    'identity': torch.nn.Identity(),
    "tanh": torch.tanh,
    "abs": torch.abs,
    "None": None,
}

# output modules

# --------------------------------------------------------------------------- #
# Head configuration (shared across models/readouts/rescale)
# --------------------------------------------------------------------------- #
@dataclass
class HeadConfig:
    key: str  # property key in curator.data.properties
    dim: int = 1
    irreps_out: Optional[str] = None  # optional e3nn irreps for the head output, e.g. "1x1o"
    is_atomwise: bool = False  # whether the readout is per-atom before reduction
    reduction: Optional[Literal["sum", "mean", "none"]] = "sum"  # "sum", "mean", "none", or None
    atomwise_key: Optional[str] = None  # optional key for per-atom output, e.g. atomic_energy
    write_atomwise: bool = False  # whether to write per-atom output to data
    # scale_by / shift_by accept:
    # - False/None: disable scaling/shifting (use 1.0/0.0)
    # - True/"default": compute from dataset stats
    # - "rms" (scale_by only): compute RMS for scale
    # - float: use the provided value
    scale_by: Union[float, Dict[str, float], str, bool, None] = None
    shift_by: Union[float, Dict[str, float], str, bool, None] = None
    atomwise_shift: bool = False  # whether the shift applies on per-atom values
    atomwise_normalization: bool = True  # if shifting structure-level value, multiply by n_atoms
    domains: Optional[List[Union[str, int]]] = None  # optional domain whitelist for this head
    per_species_scale: Union[Dict[int, float], Dict[str, float], str, None] = None  # dict, "auto", or None
    per_species_shift: Union[Dict[int, float], Dict[str, float], str, None] = None  # dict, "auto", or None


HEAD_PRESETS: Dict[str, HeadConfig] = {
    "energy": HeadConfig(
        key=energy,
        is_atomwise=True,
        reduction="sum",
        atomwise_key=atomic_energy,
        write_atomwise=False,
        dim=1,
        irreps_out="1x0e",
        scale_by=True,
        shift_by=True,
    ),
    "forces": HeadConfig(
        key=forces,
        is_atomwise=True,
        reduction=None,
        dim=3,
        irreps_out="1x1o",
        write_atomwise=True,
        scale_by="rms",
        shift_by=False,
    ),
    "atomic_energy": HeadConfig(
        key=atomic_energy,
        is_atomwise=True,
        reduction=None,
        dim=1,
        irreps_out="1x0e",
        write_atomwise=True,
    ),
    "total_charge": HeadConfig(
        key=total_charge,
        is_atomwise=False,
        reduction=None,
        dim=1,
        irreps_out="1x0e",
    ),
    "atomic_charge": HeadConfig(
        key=atomic_charge,
        is_atomwise=True,
        reduction=None,
        dim=1,
        irreps_out="1x0e",
        write_atomwise=True,
    ),
}


def normalize_head_flag(value: Union[float, Dict[str, float], str, bool, None]):
    if value is None or value is False:
        return None
    if value is True:
        return "default"
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ("default", "true"):
            return "default"
        if lowered in ("none", "false"):
            return None
        if lowered == "rms":
            return "rms"
    return value


def resolve_heads(heads: List[Union[str, HeadConfig, Dict]]) -> List[HeadConfig]:
    """
    Convert a list of string/dict/HeadConfig into a list of HeadConfig.

    Strings are looked up in HEAD_PRESETS; dicts are passed to HeadConfig(**dict).
    """
    resolved: List[HeadConfig] = []
    for h in heads:
        if isinstance(h, HeadConfig):
            resolved.append(h)
        elif isinstance(h, str):
            if h not in HEAD_PRESETS:
                raise KeyError(f"Unknown head preset '{h}'. Available: {list(HEAD_PRESETS.keys())}")
            resolved.append(HeadConfig(**HEAD_PRESETS[h].__dict__.copy()))
        elif isinstance(h, dict):
            key = h.get("key")
            if key in HEAD_PRESETS:
                base = HEAD_PRESETS[key].__dict__.copy()
                base.update(h)
                resolved.append(HeadConfig(**base))
            else:
                resolved.append(HeadConfig(**h))
        else:
            raise TypeError(f"Unsupported head spec type: {type(h)}")
    return resolved


class HeadConfigFactory:
    """
    Thin factory to let Hydra instantiate head configs from strings/dicts.
    """

    def __new__(cls, heads: List[Union[str, Dict, HeadConfig]]):
        return resolve_heads(heads)
