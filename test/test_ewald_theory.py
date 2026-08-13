import math

import pytest
import torch
from torch import nn
from ase import Atoms
from ase.neighborlist import neighbor_list

from curator.data import properties
from curator.layer import AtomwiseNN, ChargeEquilibration, EwaldSummation


DTYPE = torch.float64
FRACTIONAL_POSITIONS = [[0.0, 0.0, 0.0], [0.23, 0.37, 0.41]]
CHARGES = [1.0, -1.0]
ORTHORHOMBIC_CELL = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]
TRICLINIC_CELL = [[4.0, 0.0, 0.0], [1.7, 5.0, 0.0], [0.8, 1.2, 6.0]]


class _ConstantReadout(AtomwiseNN):
    def __init__(self, value):
        nn.Module.__init__(self)
        self.value = value

    def _compute(self, features):
        return features.new_full((features.shape[0], 1), self.value)


def _periodic_data(
    cell,
    fractional_positions,
    charges,
    ewald,
    *,
    neighbor_cutoff=None,
    translation=None,
    position_grad=False,
    charge_grad=False,
):
    atoms = Atoms(
        numbers=[1] * len(charges),
        scaled_positions=fractional_positions,
        cell=cell,
        pbc=True,
    )
    if translation is not None:
        atoms.translate(translation)

    center, neighbor, cell_shift = neighbor_list(
        "ijS",
        atoms,
        ewald.cutoff if neighbor_cutoff is None else neighbor_cutoff,
        self_interaction=False,
    )
    positions = torch.tensor(
        atoms.positions,
        dtype=DTYPE,
        requires_grad=position_grad,
    )
    cell_tensor = torch.tensor(atoms.cell.array, dtype=DTYPE).unsqueeze(0)
    edge_idx = torch.tensor(list(zip(center, neighbor)), dtype=torch.long)
    cell_displacements = torch.tensor(cell_shift, dtype=DTYPE) @ cell_tensor[0]
    edge_diff = (
        positions[edge_idx[:, 1]]
        - positions[edge_idx[:, 0]]
        + cell_displacements
    )
    atomic_charge = torch.tensor(
        charges,
        dtype=DTYPE,
        requires_grad=charge_grad,
    )
    return {
        properties.positions: positions,
        properties.cell: cell_tensor,
        properties.n_atoms: torch.tensor([len(charges)], dtype=torch.long),
        properties.image_idx: torch.zeros(len(charges), dtype=torch.long),
        properties.edge_idx: edge_idx,
        properties.edge_diff: edge_diff,
        properties.edge_dist: torch.linalg.vector_norm(edge_diff, dim=1),
        properties.cell_displacements: cell_displacements,
        properties.atomic_charge: atomic_charge,
    }


def _replace_positions(data, positions):
    edge_idx = data[properties.edge_idx]
    edge_diff = (
        positions[edge_idx[:, 1]]
        - positions[edge_idx[:, 0]]
        + data[properties.cell_displacements]
    )
    updated = data.copy()
    updated[properties.positions] = positions
    updated[properties.edge_diff] = edge_diff
    updated[properties.edge_dist] = torch.linalg.vector_norm(edge_diff, dim=1)
    return updated


def _batch_data(*structures):
    edge_blocks = []
    image_blocks = []
    offset = 0
    for image, data in enumerate(structures):
        edge_blocks.append(data[properties.edge_idx] + offset)
        count = int(data[properties.n_atoms].item())
        image_blocks.append(torch.full((count,), image, dtype=torch.long))
        offset += count

    positions = torch.cat([data[properties.positions] for data in structures])
    edge_idx = torch.cat(edge_blocks)
    cell_displacements = torch.cat(
        [data[properties.cell_displacements] for data in structures]
    )
    edge_diff = (
        positions[edge_idx[:, 1]]
        - positions[edge_idx[:, 0]]
        + cell_displacements
    )
    return {
        properties.positions: positions,
        properties.cell: torch.cat([data[properties.cell] for data in structures]),
        properties.n_atoms: torch.cat(
            [data[properties.n_atoms] for data in structures]
        ),
        properties.image_idx: torch.cat(image_blocks),
        properties.edge_idx: edge_idx,
        properties.edge_diff: edge_diff,
        properties.edge_dist: torch.linalg.vector_norm(edge_diff, dim=1),
        properties.cell_displacements: cell_displacements,
        properties.atomic_charge: torch.cat(
            [data[properties.atomic_charge] for data in structures]
        ),
    }


def _strained_data(data, strain):
    # This is the same row-vector convention as curator.layer.Strain:
    # A' = A (I + strain^T), r' = r (I + strain^T).
    deformation = torch.eye(3, dtype=DTYPE) + strain.T
    cell = data[properties.cell][0] @ deformation
    positions = data[properties.positions].detach() @ deformation
    cell_displacements = data[properties.cell_displacements] @ deformation
    edge_idx = data[properties.edge_idx]
    edge_diff = (
        positions[edge_idx[:, 1]]
        - positions[edge_idx[:, 0]]
        + cell_displacements
    )
    updated = data.copy()
    updated[properties.cell] = cell.unsqueeze(0)
    updated[properties.positions] = positions
    updated[properties.cell_displacements] = cell_displacements
    updated[properties.edge_diff] = edge_diff
    updated[properties.edge_dist] = torch.linalg.vector_norm(edge_diff, dim=1)
    updated[properties.strain] = strain
    return updated


def _rounded_vector_set(vectors):
    return {tuple(row.tolist()) for row in torch.round(vectors * 1.0e10) / 1.0e10}


def test_reciprocal_vectors_are_complete_for_skew_triclinic_cells():
    """k=nB must use row reciprocal vectors and a cancellation-safe n bound."""
    cell = torch.tensor(
        [
            [2.8578794060260995, 0.0, 0.0],
            [2.2821256884660124, 3.570715607842654, 0.0],
            [-2.3171831684383233, 3.533758200379892, 3.414732544637375],
        ],
        dtype=DTYPE,
    )
    k_cutoff = 4.2
    actual, _ = EwaldSummation.get_reciprocal_k_vectors(cell, k_cutoff)

    # Independent oversized integer cube: this deliberately does not use the
    # production bound and is the direct definition of the reciprocal set.
    reciprocal_rows = 2.0 * math.pi * torch.linalg.inv(cell).T
    values = torch.arange(-8, 9, dtype=DTYPE)
    integer_vectors = torch.cartesian_prod(values, values, values)
    expected = integer_vectors @ reciprocal_rows
    norms = torch.linalg.vector_norm(expected, dim=1)
    expected = expected[(norms <= k_cutoff) & (norms > 0.0)]

    assert _rounded_vector_set(actual) == _rounded_vector_set(expected)


@pytest.mark.parametrize(
    ("cell", "reference_energy"),
    [
        (ORTHORHOMBIC_CELL, -5.610376737245),
        (TRICLINIC_CELL, -5.581606408703),
    ],
)
def test_cell_shapes_and_ewald_parameter_independence(cell, reference_energy):
    """The full Ewald sum is independent of alpha as both sums converge."""
    energies = []
    for alpha in (0.25, 0.4, 0.6):
        ewald = EwaldSummation(alpha=alpha, acc_factor=12.0)
        data = _periodic_data(cell, FRACTIONAL_POSITIONS, CHARGES, ewald)
        energies.append(ewald(data).item())

    torch.testing.assert_close(
        torch.tensor(energies, dtype=DTYPE),
        torch.full((3,), reference_energy, dtype=DTYPE),
        rtol=0.0,
        atol=5.0e-10,
    )


def test_real_and_reciprocal_cutoffs_converge_to_theoretical_ewald_sum():
    reference_energy = -5.581606408703
    neighbor_cutoff = 15.0
    template = EwaldSummation(alpha=0.4, cutoff=neighbor_cutoff, k_cutoff=5.5)
    data = _periodic_data(
        TRICLINIC_CELL,
        FRACTIONAL_POSITIONS,
        CHARGES,
        template,
        neighbor_cutoff=neighbor_cutoff,
    )

    real_errors = []
    for cutoff in (3.5, 5.0, 7.0, 9.0, 13.2):
        energy = EwaldSummation(alpha=0.4, cutoff=cutoff, k_cutoff=5.5)(data)
        real_errors.append(abs(energy.item() - reference_energy))
    reciprocal_errors = []
    for k_cutoff in (1.5, 2.0, 2.5, 3.0, 4.2):
        energy = EwaldSummation(
            alpha=0.4,
            cutoff=neighbor_cutoff,
            k_cutoff=k_cutoff,
        )(data)
        reciprocal_errors.append(abs(energy.item() - reference_energy))

    assert all(a > b for a, b in zip(real_errors, real_errors[1:]))
    assert all(a > b for a, b in zip(reciprocal_errors, reciprocal_errors[1:]))
    assert real_errors[-1] < 2.0e-9
    assert reciprocal_errors[-1] < 2.0e-9


def test_translation_invariance_and_force_finite_difference():
    ewald = EwaldSummation(alpha=0.4, acc_factor=12.0)
    data = _periodic_data(
        TRICLINIC_CELL,
        FRACTIONAL_POSITIONS,
        CHARGES,
        ewald,
        position_grad=True,
    )
    energy = ewald(data)
    force = -torch.autograd.grad(energy.sum(), data[properties.positions])[0]

    translated = _periodic_data(
        TRICLINIC_CELL,
        FRACTIONAL_POSITIONS,
        CHARGES,
        ewald,
        translation=[0.371, -1.217, 2.413],
        position_grad=True,
    )
    translated_energy = ewald(translated)
    translated_force = -torch.autograd.grad(
        translated_energy.sum(), translated[properties.positions]
    )[0]
    torch.testing.assert_close(translated_energy, energy, rtol=0.0, atol=2.0e-12)
    torch.testing.assert_close(translated_force, force, rtol=0.0, atol=2.0e-11)
    torch.testing.assert_close(
        force.sum(dim=0), torch.zeros(3, dtype=DTYPE), rtol=0.0, atol=2.0e-11
    )

    step = 1.0e-5
    plus = data[properties.positions].detach().clone()
    minus = data[properties.positions].detach().clone()
    plus[1, 0] += step
    minus[1, 0] -= step
    finite_difference_force = -(
        ewald(_replace_positions(data, plus))
        - ewald(_replace_positions(data, minus))
    ) / (2.0 * step)
    torch.testing.assert_close(
        force[1, 0],
        finite_difference_force.squeeze(0),
        rtol=2.0e-7,
        atol=2.0e-8,
    )


def test_batch_matches_individual_structures():
    ewald = EwaldSummation(alpha=0.4, acc_factor=12.0)
    first = _periodic_data(
        ORTHORHOMBIC_CELL,
        FRACTIONAL_POSITIONS,
        CHARGES,
        ewald,
        position_grad=True,
    )
    second = _periodic_data(
        TRICLINIC_CELL,
        [[0.11, 0.07, 0.19], [0.61, 0.53, 0.47]],
        [0.6, -0.6],
        ewald,
        position_grad=True,
    )

    first_energy = ewald(first)
    second_energy = ewald(second)
    first_force = -torch.autograd.grad(
        first_energy.sum(), first[properties.positions]
    )[0]
    second_force = -torch.autograd.grad(
        second_energy.sum(), second[properties.positions]
    )[0]
    batch = _batch_data(first, second)
    actual_energy = ewald(batch)
    actual_force = -torch.autograd.grad(
        actual_energy.sum(), batch[properties.positions]
    )[0]

    torch.testing.assert_close(
        actual_energy,
        torch.cat([first_energy, second_energy]),
        rtol=0.0,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        actual_force,
        torch.cat([first_force, second_force]),
        rtol=0.0,
        atol=2.0e-11,
    )


def test_charged_simple_cubic_jellium_matches_madelung_theory():
    """One ion plus uniform background has the sc jellium Madelung energy."""
    length = 10.0
    charge = 1.0
    expected = (
        -2.837297479
        * EwaldSummation.CONV_FACT
        * charge**2
        / (2.0 * length)
    )
    energies = []
    for alpha in (0.2, 0.3, 0.4):
        ewald = EwaldSummation(alpha=alpha, acc_factor=12.0)
        data = _periodic_data(
            torch.eye(3, dtype=DTYPE) * length,
            [[0.173, 0.291, 0.417]],
            [charge],
            ewald,
            position_grad=True,
            charge_grad=True,
        )
        energy = ewald(data)
        energies.append(energy.item())
        force, chemical_potential = torch.autograd.grad(
            energy.sum(),
            (data[properties.positions], data[properties.atomic_charge]),
        )
        torch.testing.assert_close(force, torch.zeros_like(force), atol=1.0e-11, rtol=0.0)
        torch.testing.assert_close(
            chemical_potential.squeeze(),
            2.0 * energy.squeeze() / charge,
            atol=2.0e-10,
            rtol=0.0,
        )

    torch.testing.assert_close(
        torch.tensor(energies, dtype=DTYPE),
        torch.full((3,), expected, dtype=DTYPE),
        atol=8.0e-9,
        rtol=0.0,
    )


@pytest.mark.parametrize("charges", ([1.0, -1.0], [0.7, -0.2]))
def test_ewald_kernel_reproduces_energy_and_charge_gradient(charges):
    ewald = EwaldSummation(alpha=0.4, acc_factor=12.0)
    data = _periodic_data(
        TRICLINIC_CELL,
        FRACTIONAL_POSITIONS,
        charges,
        ewald,
        neighbor_cutoff=15.0,
        charge_grad=True,
    )
    energy = ewald(data)
    kernel = ewald.get_ewald_kernel(
        data[properties.cell],
        data[properties.n_atoms],
        data[properties.positions],
        data[properties.edge_dist],
        data[properties.edge_idx],
    )[0]
    charge = data[properties.atomic_charge]
    kernel_energy = 0.5 * charge @ kernel @ charge

    torch.testing.assert_close(kernel_energy, energy.squeeze(), atol=2.0e-11, rtol=0.0)
    charge_gradient = torch.autograd.grad(energy.sum(), charge)[0]
    torch.testing.assert_close(charge_gradient, kernel @ charge, atol=2.0e-11, rtol=0.0)


def test_cell_derivative_matches_scaling_theory_and_finite_difference():
    length = 10.0
    ewald = EwaldSummation(alpha=0.4, acc_factor=12.0)
    data = _periodic_data(
        torch.eye(3, dtype=DTYPE) * length,
        [[0.173, 0.291, 0.417]],
        [1.0],
        ewald,
    )
    strain = torch.zeros((3, 3), dtype=DTYPE, requires_grad=True)
    energy = ewald(_strained_data(data, strain)).squeeze()
    derivative = torch.autograd.grad(energy, strain)[0]
    volume = length**3

    # E(lambda A) = E(A)/lambda for Coulomb energy.  Cubic symmetry gives
    # dE/d eps_xx = dE/d eps_yy = dE/d eps_zz = -E/3.
    expected_diagonal = torch.full((3,), -energy.detach() / 3.0, dtype=DTYPE)
    torch.testing.assert_close(
        torch.diagonal(derivative), expected_diagonal, atol=2.0e-10, rtol=0.0
    )
    pressure = -torch.trace(derivative / volume) / 3.0
    torch.testing.assert_close(pressure, energy.detach() / (3.0 * volume), atol=2.0e-13, rtol=0.0)

    triclinic = _periodic_data(
        TRICLINIC_CELL, FRACTIONAL_POSITIONS, CHARGES, ewald
    )
    triclinic_strain = torch.zeros((3, 3), dtype=DTYPE, requires_grad=True)
    triclinic_energy = ewald(_strained_data(triclinic, triclinic_strain)).squeeze()
    triclinic_derivative = torch.autograd.grad(
        triclinic_energy, triclinic_strain
    )[0]
    step = 1.0e-5
    plus = torch.zeros((3, 3), dtype=DTYPE)
    minus = torch.zeros((3, 3), dtype=DTYPE)
    plus[0, 1] = step
    minus[0, 1] = -step
    finite_difference = (
        ewald(_strained_data(triclinic, plus))
        - ewald(_strained_data(triclinic, minus))
    ) / (2.0 * step)
    torch.testing.assert_close(
        triclinic_derivative[0, 1],
        finite_difference.squeeze(),
        atol=2.0e-8,
        rtol=2.0e-7,
    )


def test_charge_equilibration_exports_theoretical_ewald_stress_and_virial():
    ewald = EwaldSummation(alpha=0.4, acc_factor=12.0)
    base = _periodic_data(
        TRICLINIC_CELL,
        FRACTIONAL_POSITIONS,
        CHARGES,
        ewald,
        charge_grad=True,
    )
    strain = torch.zeros((3, 3), dtype=DTYPE, requires_grad=True)
    data = _strained_data(base, strain)
    data[properties.node_embedding] = torch.zeros((2, 2), dtype=DTYPE)
    data[properties.total_charge] = torch.zeros(1, dtype=DTYPE)

    reference_energy = ewald(data).sum()
    strain_gradient = torch.autograd.grad(
        reference_energy, strain, retain_graph=True
    )[0]
    volume = torch.abs(torch.linalg.det(data[properties.cell][0]))
    expected_virial = (-strain_gradient).reshape(9)[[0, 4, 8, 5, 2, 1]]
    expected_stress = (strain_gradient / volume).reshape(9)[[0, 4, 8, 5, 2, 1]]

    qeq = ChargeEquilibration(
        num_features=2,
        electronegativity_mlp=_ConstantReadout(0.0),
        hardness_mlp=_ConstantReadout(1.0),
        ewald=ewald,
        min_hardness=0.0,
    ).eval()
    output = qeq(data, training=False)

    torch.testing.assert_close(
        output[properties.ewald_virial].squeeze(0),
        expected_virial,
        atol=2.0e-11,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output[properties.ewald_stress].squeeze(0),
        expected_stress,
        atol=2.0e-13,
        rtol=0.0,
    )
