import torch

from curator.data import properties
from curator.layer import ChargeEquilibration, EwaldSummation
from curator.variational import (
    EwaldChargeOperator,
    VariableLayout,
    compile_boundary,
    evaluate_kkt,
    solve_dense,
    solve_projected_cg,
)
from curator.variational.variables import (
    ChargeFunctional,
    ChargeQuadraticOperator,
    charge_variable_spec,
    fixed_electrode_potentials_recipe,
    fixed_total_charge_boundary,
    project_fixed_total_charge,
)

from test_ewald_theory import FRACTIONAL_POSITIONS, TRICLINIC_CELL, _periodic_data


DTYPE = torch.float64


def test_generic_projection_matches_legacy_ragged_fixed_q_projection():
    raw = torch.tensor([0.2, -0.1, 0.4, 0.3, -0.2], dtype=DTYPE)
    hardness = torch.tensor([1.0, 2.0, 4.0, 0.5, 3.0], dtype=DTYPE)
    image_idx = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    total = torch.tensor([1.0, -0.5], dtype=DTYPE)

    generic = project_fixed_total_charge(raw, hardness, image_idx, total)
    legacy = ChargeEquilibration._conserve_total_charge(
        raw, hardness, image_idx, total
    )

    torch.testing.assert_close(generic, legacy, atol=2.0e-15, rtol=0.0)


def test_charge_functional_exact_solve_and_multiplier_sign():
    variable = charge_variable_spec()
    layout = VariableLayout(variable=variable, n_entities=3)
    boundary = compile_boundary(fixed_total_charge_boundary(0.7, variable), layout)
    chi = torch.tensor([0.3, -0.2, 0.5], dtype=DTYPE)
    hardness = torch.tensor([1.0, 2.0, 4.0], dtype=DTYPE)
    interaction = torch.tensor(
        [[0.4, -0.1, 0.05], [-0.1, 0.3, -0.02], [0.05, -0.02, 0.2]],
        dtype=DTYPE,
    )
    functional = ChargeFunctional(chi, hardness, interaction)

    result = solve_dense(functional.problem(boundary))
    chemical_potential = functional.chemical_potential(result.state)

    torch.testing.assert_close(result.state.sum(), torch.tensor(0.7, dtype=DTYPE))
    torch.testing.assert_close(
        chemical_potential,
        torch.full_like(chemical_potential, -result.multipliers[0]),
        atol=2.0e-14,
        rtol=0.0,
    )
    residual = evaluate_kkt(
        functional.problem(boundary), result.state, result.multipliers
    )
    assert residual.maximum_norm < 1.0e-12


def test_real_ewald_operator_matches_energy_kernel_and_charge_gradient():
    ewald = EwaldSummation(alpha=0.4, acc_factor=12.0)
    data = _periodic_data(
        TRICLINIC_CELL,
        FRACTIONAL_POSITIONS,
        [0.7, -0.2],
        ewald,
        neighbor_cutoff=15.0,
        charge_grad=True,
    )
    operator = EwaldChargeOperator(ewald)
    charge = data[properties.atomic_charge]
    kernel = operator.kernels(data)[0]
    energy = operator.energy(data, charge).squeeze()

    torch.testing.assert_close(
        0.5 * charge @ kernel @ charge, energy, atol=2.0e-11, rtol=0.0
    )
    torch.testing.assert_close(
        operator.matvec(data, charge),
        torch.autograd.grad(energy, charge)[0],
        atol=3.0e-11,
        rtol=0.0,
    )


def test_real_ewald_qeq_problem_matches_direct_energy_and_is_feasible():
    ewald = EwaldSummation(alpha=0.4, acc_factor=12.0)
    data = _periodic_data(
        TRICLINIC_CELL,
        FRACTIONAL_POSITIONS,
        [0.1, 0.2],
        ewald,
        neighbor_cutoff=15.0,
    )
    operator = EwaldChargeOperator(ewald)
    kernel = operator.kernels(data)[0]
    variable = charge_variable_spec()
    compiled = compile_boundary(
        fixed_total_charge_boundary(0.3, variable),
        VariableLayout(variable=variable, n_entities=2),
    )
    chi = torch.tensor([0.4, 0.1], dtype=DTYPE)
    hardness = torch.tensor([8.0, 7.0], dtype=DTYPE)
    functional = ChargeFunctional(chi, hardness, kernel)

    result = solve_dense(functional.problem(compiled))
    direct = (
        chi @ result.state
        + 0.5 * hardness @ result.state.square()
        + operator.energy(data, result.state).squeeze()
    )

    torch.testing.assert_close(result.energy, direct, atol=3.0e-11, rtol=0.0)
    torch.testing.assert_close(result.state.sum(), torch.tensor(0.3, dtype=DTYPE))
    assert result.residuals.maximum_norm < 1.0e-10


def test_matrix_free_charge_operator_matches_dense_ewald_fixed_q_solution():
    ewald = EwaldSummation(alpha=0.4, acc_factor=12.0)
    data = _periodic_data(
        TRICLINIC_CELL,
        FRACTIONAL_POSITIONS,
        [0.1, 0.2],
        ewald,
        neighbor_cutoff=15.0,
    )
    ewald_operator = EwaldChargeOperator(ewald)
    interaction = ewald_operator.bind(data)
    chi = torch.tensor([0.4, 0.1], dtype=DTYPE)
    hardness = torch.tensor([8.0, 7.0], dtype=DTYPE)
    variable = charge_variable_spec()
    compiled = compile_boundary(
        fixed_total_charge_boundary(0.3, variable),
        VariableLayout(variable=variable, n_entities=2),
    )
    dense = solve_dense(
        ChargeFunctional(chi, hardness, interaction.kernel()).problem(compiled)
    )

    iterative = solve_projected_cg(
        ChargeQuadraticOperator(hardness, interaction),
        chi,
        compiled.constraint_matrix,
        compiled.constraint_target,
        rtol=1.0e-12,
        atol=1.0e-12,
        reduced_stability_certified=True,
    )

    assert iterative.reliable
    torch.testing.assert_close(iterative.state, dense.state, atol=3.0e-11, rtol=0.0)
    torch.testing.assert_close(iterative.energy, dense.energy, atol=3.0e-11, rtol=0.0)
    torch.testing.assert_close(
        iterative.multipliers, dense.multipliers, atol=3.0e-11, rtol=0.0
    )


def test_real_ewald_fixed_potential_recipe_sets_charge_chemical_potentials():
    ewald = EwaldSummation(alpha=0.4, acc_factor=12.0)
    data = _periodic_data(
        TRICLINIC_CELL,
        FRACTIONAL_POSITIONS,
        [0.0, 0.0],
        ewald,
        neighbor_cutoff=15.0,
    )
    interaction = EwaldChargeOperator(ewald).bind(data).kernel()
    variable = charge_variable_spec()
    boundary = fixed_electrode_potentials_recipe(
        {"left": [0], "right": [1]},
        {"left": -0.2, "right": 0.3},
        variable=variable,
    )
    compiled = compile_boundary(
        boundary,
        VariableLayout(variable=variable, n_entities=2),
    )
    functional = ChargeFunctional(
        torch.tensor([0.4, 0.1], dtype=DTYPE),
        torch.tensor([20.0, 19.0], dtype=DTYPE),
        interaction,
    )

    result = solve_dense(functional.problem(compiled))

    torch.testing.assert_close(
        functional.chemical_potential(result.state),
        torch.tensor([-0.2, 0.3], dtype=DTYPE),
        atol=3.0e-11,
        rtol=0.0,
    )
