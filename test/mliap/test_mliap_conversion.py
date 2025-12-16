import copy
from pathlib import Path

import torch
from ase.io import read

from curator.data import AseDataReader, properties
from curator.layer import find_layer_by_name_recursive
from curator.simulate.lammps_mliap_interface import LAMMPS_MLIAP
from curator.utils import load_models


def _aggregate_forces(edge_forces: torch.Tensor, edge_idx: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """Reconstruct atomic forces from per-edge forces following GradientOutput logic."""
    i_forces = torch.zeros((n_atoms, 3), device=edge_forces.device, dtype=edge_forces.dtype)
    j_forces = torch.zeros_like(i_forces)
    i_forces.index_add_(0, edge_idx[:, 0], edge_forces)
    j_forces.index_add_(0, edge_idx[:, 1], -edge_forces)
    return i_forces + j_forces


def test_mliap_conversion_matches_energy_and_forces():
    device = torch.device("cpu")
    repo_root = Path(__file__).resolve().parents[2]

    atoms = read(repo_root / "test" / "LiFePO4.traj", index=0)
    original_model = load_models(repo_root / "test" / "best_model.ckpt", device=device)[0].eval()

    cutoff = float(find_layer_by_name_recursive(original_model, "cutoff"))
    reader = AseDataReader(cutoff, compute_neighbor_list=True)
    inputs = reader(atoms)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    original_inputs = {k: v.clone() for k, v in inputs.items()}
    original_output = original_model(original_inputs)
    original_energy = original_output[properties.energy]
    original_forces = original_output[properties.forces]

    converted_model = copy.deepcopy(original_model)
    LAMMPS_MLIAP._convert_model(converted_model)

    converted_inputs = {k: v.clone() for k, v in inputs.items()}
    converted_output = converted_model(converted_inputs)
    atomic_energy = converted_output[properties.atomic_energy]
    edge_forces = converted_output[properties.edge_forces]

    n_atoms = inputs[properties.n_atoms].item()
    converted_energy = atomic_energy[:n_atoms].sum().reshape_as(original_energy)
    converted_forces = _aggregate_forces(edge_forces, inputs[properties.edge_idx], n_atoms)

    print(f'Original energy: {original_energy}')
    print(f'Converted energy: {converted_energy}')
    print(f'Original forces: {original_forces}')
    print(f'Converted forces: {converted_forces}')
    torch.testing.assert_close(converted_energy, original_energy)
    torch.testing.assert_close(converted_forces, original_forces, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    test_mliap_conversion_matches_energy_and_forces()
