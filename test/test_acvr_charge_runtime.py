import pytest
import torch

from curator.data import properties
from curator.layer import EwaldSummation
from curator.train import VariationalJointLoss
from curator.variational import ACVRChargeRuntime, VariableLayout, compile_boundary, solve_dense
from curator.variational.variables import (
    ChargeFunctional,
    charge_variable_spec,
    fixed_electrode_potentials_recipe,
    fixed_total_charge_recipe,
)
from test_ewald_theory import (
    FRACTIONAL_POSITIONS,
    TRICLINIC_CELL,
    _periodic_data,
    _replace_positions,
)


DTYPE = torch.float64


class ToyInteraction:
    def __init__(self):
        self.calls = 0

    def kernels(self, data):
        self.calls += 1
        positions = data[properties.positions]
        counts = data[properties.n_atoms]
        kernels = []
        offset = 0
        for count in counts.tolist():
            local = positions[offset : offset + count, 0]
            if count != 2:
                raise ValueError("toy interaction expects two atoms")
            coupling = -0.15 + 0.05 * (local[0] - local[1])
            kernels.append(
                torch.stack(
                    (
                        torch.stack((0.4 + 0.1 * local[0].square(), coupling)),
                        torch.stack((coupling, 0.5 + 0.08 * local[1].square())),
                    )
                )
            )
            offset += count
        return kernels


def _boundaries():
    variable = charge_variable_spec()
    layout = VariableLayout(variable=variable, n_entities=2, dtype=DTYPE)
    fixed_q = compile_boundary(fixed_total_charge_recipe(0.2, variable=variable), layout)
    fixed_v = compile_boundary(
        fixed_electrode_potentials_recipe(
            {"left": [0], "right": [1]},
            {"left": -0.2, "right": 0.3},
            variable=variable,
        ),
        layout,
    )
    return fixed_q, fixed_v


def _inputs(first_x=0.2, force=False, strain_value=None):
    if strain_value is None:
        positions = torch.tensor(
            [[first_x, 0.0, 0.0], [1.1, 0.0, 0.0]],
            dtype=DTYPE,
            requires_grad=force,
        )
        cell = torch.eye(3, dtype=DTYPE).reshape(1, 3, 3) * 8.0
        strain = None
    else:
        strain = torch.zeros((1, 3, 3), dtype=DTYPE, requires_grad=force)
        strain = strain + torch.tensor(
            [[[strain_value, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=DTYPE,
        )
        base_cell = torch.eye(3, dtype=DTYPE).reshape(1, 3, 3) * 8.0
        base_positions = torch.tensor(
            [[first_x, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=DTYPE
        )
        cell = base_cell + torch.matmul(base_cell, strain.transpose(1, 2))
        positions = base_positions + torch.matmul(
            base_positions[:, None, :], strain[torch.zeros(2, dtype=torch.long)].transpose(1, 2)
        ).squeeze(1)
    data = {
        properties.positions: positions,
        properties.cell: cell,
        properties.n_atoms: torch.tensor([2], dtype=torch.long),
        properties.image_idx: torch.tensor([0, 0], dtype=torch.long),
    }
    if strain is not None:
        data[properties.strain] = strain
    chi = torch.stack((0.3 + 0.2 * positions[0, 0].square(), -0.1 * positions[1, 0]))
    hardness = torch.stack((5.0 + 0.1 * positions[0, 0], 6.0 + 0.2 * positions[1, 0].square()))
    return data, chi, hardness


def _run(
    first_x=0.2,
    force_mode=None,
    strain_value=None,
    solver="implicit",
    proposals=None,
    create_graph=False,
):
    data, chi, hardness = _inputs(
        first_x=first_x,
        force=force_mode is not None,
        strain_value=strain_value,
    )
    interaction = ToyInteraction()
    runtime = ACVRChargeRuntime(
        interaction,
        solver=solver,
        force_mode=force_mode,
        residual_tolerance=1.0e-10,
        feasibility_tolerance=1.0e-10,
    )
    result = runtime(
        data,
        chi,
        hardness,
        _boundaries(),
        condition_geometry_index=torch.tensor([0, 0]),
        proposals=proposals,
        create_graph=create_graph,
    )
    return result, interaction, data, chi, hardness


def test_runtime_reuses_geometry_and_matches_both_exact_boundary_problems():
    result, interaction, data, chi, hardness = _run()
    kernel = ToyInteraction().kernels(data)[0]
    expected = [
        solve_dense(ChargeFunctional(chi, hardness, kernel).problem(boundary))
        for boundary in _boundaries()
    ]

    assert interaction.calls == 1
    assert result.condition_state_ptr.tolist() == [0, 2, 4]
    assert result.multiplier_ptr.tolist() == [0, 1, 1]
    torch.testing.assert_close(result.state[:2], expected[0].state)
    torch.testing.assert_close(result.state[2:], expected[1].state)
    torch.testing.assert_close(result.thermodynamic_energy, torch.stack([item.energy for item in expected]))
    torch.testing.assert_close(result.base_energy, result.onsite_energy + result.interaction_energy)
    torch.testing.assert_close(
        result.boundary_energy,
        result.thermodynamic_energy - result.base_energy,
    )
    torch.testing.assert_close(
        result.chemical_potential[2:],
        torch.tensor([-0.2, 0.3], dtype=DTYPE),
        atol=2.0e-14,
        rtol=0.0,
    )
    assert result.reliable.tolist() == [True, True]
    assert result.status == ("implicit_exact", "implicit_exact")
    assert result.normalized_kkt_residual.abs().max() < 1.0e-12


@pytest.mark.parametrize("condition", [0, 1])
def test_complete_envelope_force_matches_relaxed_finite_difference(condition):
    result, _, _, _, _ = _run(force_mode="envelope")
    delta = 1.0e-5
    minus = _run(first_x=0.2 - delta)[0].thermodynamic_energy[condition]
    plus = _run(first_x=0.2 + delta)[0].thermodynamic_energy[condition]
    finite_difference_force = -(plus - minus) / (2.0 * delta)
    atom_offset = condition * 2

    torch.testing.assert_close(
        result.condition_forces[atom_offset, 0],
        finite_difference_force,
        atol=2.0e-10,
        rtol=2.0e-8,
    )
    assert not result.conservative_for_reported_energy[condition]


def test_exact_full_and_envelope_forces_agree_but_only_full_is_strictly_reported():
    envelope = _run(force_mode="envelope")[0]
    full = _run(force_mode="full_differentiable")[0]
    torch.testing.assert_close(envelope.condition_forces, full.condition_forces, atol=2.0e-13, rtol=0.0)
    assert envelope.conservative_for_reported_energy.tolist() == [False, False]
    assert full.conservative_for_reported_energy.tolist() == [True, True]


def test_strain_virial_contains_onsite_and_interaction_relaxation():
    result = _run(force_mode="envelope", strain_value=0.0)[0]
    delta = 1.0e-5
    minus = _run(strain_value=-delta)[0].thermodynamic_energy[0]
    plus = _run(strain_value=delta)[0].thermodynamic_energy[0]
    expected_virial_xx = -(plus - minus) / (2.0 * delta)

    torch.testing.assert_close(
        result.condition_virial[0, 0],
        expected_virial_xx,
        atol=2.0e-10,
        rtol=2.0e-8,
    )
    volume = torch.abs(torch.linalg.det(_inputs(strain_value=0.0)[0][properties.cell]))[0]
    torch.testing.assert_close(
        result.condition_stress[0, 0],
        -result.condition_virial[0, 0] / volume,
    )


def test_amortized_runtime_corrects_proposals_and_exports_loss_contract():
    proposals = torch.tensor([0.8, -0.6, 0.5, -0.3], dtype=DTYPE)
    amortized = _run(solver="amortized", proposals=proposals)[0]
    exact = _run()[0]

    torch.testing.assert_close(amortized.state, exact.state, atol=2.0e-14, rtol=0.0)
    assert amortized.reliable.tolist() == [True, True]
    assert amortized.status == ("corrected", "corrected")
    prediction = amortized.as_variational_prediction(properties.condition_energy)
    assert prediction["value_space"] == "physical"
    assert properties.condition_energy in prediction["labels"]
    assert prediction["labels"][properties.variable_value]["shape"] == ((2, 1), (2, 1))
    assert prediction["certificates"]["kkt"]["source"] == "compiled_variational_operator"
    without_energy = amortized.as_variational_prediction()
    assert properties.condition_energy not in without_energy["labels"]
    assert properties.condition_free_energy not in without_energy["labels"]
    with pytest.raises(ValueError, match="energy_label"):
        amortized.as_variational_prediction("ambiguous_energy")


def test_runtime_rejects_implicit_condition_duplication_and_missing_proposals():
    data, chi, hardness = _inputs()
    runtime = ACVRChargeRuntime(ToyInteraction(), force_mode=None)
    with pytest.raises(ValueError, match="condition_geometry_index is required"):
        runtime(data, chi, hardness, _boundaries())
    runtime = ACVRChargeRuntime(ToyInteraction(), solver="amortized", force_mode=None)
    with pytest.raises(ValueError, match="requires proposals"):
        runtime(
            data,
            chi,
            hardness,
            _boundaries(),
            condition_geometry_index=torch.tensor([0, 0]),
        )


def test_real_ewald_runtime_relaxed_force_matches_finite_difference():
    ewald = EwaldSummation(alpha=0.4, acc_factor=12.0)
    data = _periodic_data(
        TRICLINIC_CELL,
        FRACTIONAL_POSITIONS,
        [0.0, 0.0],
        ewald,
        neighbor_cutoff=15.0,
        position_grad=True,
    )
    positions = data[properties.positions]
    chi = torch.stack((0.2 + 0.01 * positions[0, 0].square(), -0.1 * positions[1, 0]))
    hardness = torch.tensor([8.0, 7.0], dtype=DTYPE)
    boundary = _boundaries()[0]
    runtime = ACVRChargeRuntime(
        ewald,
        force_mode="envelope",
        residual_tolerance=1.0e-10,
    )
    result = runtime(data, chi, hardness, (boundary,), create_graph=False)

    delta = 1.0e-5
    energies = []
    for sign in (-1.0, 1.0):
        shifted_positions = positions.detach().clone()
        shifted_positions[0, 0] += sign * delta
        shifted = _replace_positions(data, shifted_positions)
        shifted_chi = torch.stack(
            (
                0.2 + 0.01 * shifted_positions[0, 0].square(),
                -0.1 * shifted_positions[1, 0],
            )
        )
        shifted_result = ACVRChargeRuntime(ewald, force_mode=None)(
            shifted, shifted_chi, hardness, (boundary,)
        )
        energies.append(shifted_result.thermodynamic_energy[0])
    finite_difference_force = -(energies[1] - energies[0]) / (2.0 * delta)

    torch.testing.assert_close(
        result.condition_forces[0, 0],
        finite_difference_force,
        atol=2.0e-7,
        rtol=2.0e-6,
    )
    torch.testing.assert_close(result.state.sum(), torch.tensor(0.2, dtype=DTYPE))


def test_runtime_prediction_trains_two_boundaries_without_duplicating_geometry():
    result, _, data, _, _ = _run(force_mode="envelope", create_graph=True)
    prediction = {"variational_prediction": result.as_variational_prediction(
        properties.condition_free_energy
    )}
    predicted_labels = prediction["variational_prediction"]["labels"]
    target_labels = {}
    for name, entry in predicted_labels.items():
        target = entry["values"].detach().clone()
        if name == properties.condition_forces:
            target = target + 0.1
        target_labels[name] = {
            "values": target.reshape(-1),
            "ptr": entry["ptr"],
            "shape": entry["shape"],
            "present": torch.tensor([True, True]),
            "unit": entry["unit"],
            "metadata": (
                {"energy_kind": "compiled thermodynamic potential"},
                {"energy_kind": "compiled thermodynamic potential"},
            ),
        }
    batch = {
        "weight": torch.tensor([1.0], dtype=DTYPE),
        "variational": {
            "condition_ptr": torch.tensor([0, 2]),
            "condition_geometry_index": torch.tensor([0, 0]),
            "condition_weight": torch.tensor([1.0, 1.0], dtype=DTYPE),
            "boundary_spec": ({"compile_status": "compiled"},) * 2,
            "labels": target_labels,
        },
    }
    loss = VariationalJointLoss(
        supervised_weights={
            properties.condition_free_energy: 1.0,
            properties.variable_value: 1.0,
            properties.condition_forces: 1.0,
        },
        supervised_scales={
            properties.condition_free_energy: 1.0,
            properties.variable_value: 1.0,
            properties.condition_forces: 1.0,
        },
        required_labels=(properties.condition_free_energy, properties.condition_forces),
        required_metadata={
            properties.condition_free_energy: {
                "energy_kind": "compiled thermodynamic potential"
            }
        },
        certificate_weights={"kkt": 1.0},
    )(prediction, batch)["loss"]
    loss.backward()

    assert loss > 0
    assert data[properties.positions].grad is not None
    assert torch.isfinite(data[properties.positions].grad).all()
