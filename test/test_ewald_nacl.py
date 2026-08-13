import torch
from ase.build import bulk
from ase.neighborlist import neighbor_list

from curator.data import properties
from curator.layer import EwaldSummation


def test_ewald_matches_nacl_madelung_energy():
    """NaCl lattice energy for formal charges must match the Madelung sum."""
    lattice_constant = 5.64
    madelung_constant = 1.7475645946331822
    atoms = bulk("NaCl", "rocksalt", a=lattice_constant, cubic=True)
    ewald = EwaldSummation(alpha=0.4, acc_factor=12.0)

    center, neighbor, cell_shift = neighbor_list(
        "ijS",
        atoms,
        ewald.cutoff,
        self_interaction=False,
    )
    positions = torch.tensor(
        atoms.positions,
        dtype=torch.float64,
        requires_grad=True,
    )
    cell = torch.tensor(atoms.cell.array, dtype=torch.float64).unsqueeze(0)
    edge_idx = torch.tensor(
        list(zip(center, neighbor)),
        dtype=torch.long,
    )
    shifts = torch.tensor(cell_shift, dtype=torch.float64) @ cell[0]
    edge_diff = positions[edge_idx[:, 1]] - positions[edge_idx[:, 0]] + shifts
    charges = torch.tensor(
        [1.0 if symbol == "Na" else -1.0 for symbol in atoms.get_chemical_symbols()],
        dtype=torch.float64,
    )
    data = {
        properties.positions: positions,
        properties.cell: cell,
        properties.n_atoms: torch.tensor([len(atoms)], dtype=torch.long),
        properties.image_idx: torch.zeros(len(atoms), dtype=torch.long),
        properties.edge_idx: edge_idx,
        properties.edge_dist: torch.linalg.vector_norm(edge_diff, dim=1),
        properties.atomic_charge: charges,
    }

    energy = ewald(data)

    # The conventional rocksalt cell contains four NaCl pairs and its
    # nearest-neighbor distance is a / 2.
    expected = (
        -4.0
        * madelung_constant
        * EwaldSummation.CONV_FACT
        / (lattice_constant / 2.0)
    )
    torch.testing.assert_close(
        energy,
        torch.tensor([expected], dtype=energy.dtype),
        rtol=1.0e-10,
        atol=1.0e-10,
    )

    forces = -torch.autograd.grad(energy.sum(), positions)[0]
    torch.testing.assert_close(
        forces,
        torch.zeros_like(forces),
        rtol=0.0,
        atol=1.0e-10,
    )
