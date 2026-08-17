import pytest
import torch

from curator.variational import GaugeSpec, VariableLayout, compile_boundary, solve_dense
from curator.variational.variables import ChargeFunctional, charge_variable_spec
from curator.variational.variables.charge_boundary import (
    fixed_electrode_potentials_recipe,
    fixed_fragment_charges_recipe,
    fixed_total_charge_recipe,
    floating_conductor_recipe,
    mixed_electrode_electrolyte_recipe,
)


DTYPE = torch.float64


def _compile(boundary, n_atoms):
    return compile_boundary(
        boundary,
        VariableLayout(variable=boundary.variable, n_entities=n_atoms),
    )


def _functional(n_atoms, interaction=None):
    return ChargeFunctional(
        electronegativity=torch.linspace(-0.2, 0.3, n_atoms, dtype=DTYPE),
        hardness=torch.linspace(2.0, 3.0, n_atoms, dtype=DTYPE),
        interaction_kernel=interaction,
    )


def test_fixed_total_charge_recipe_compiles_and_solves_fixed_q():
    boundary = fixed_total_charge_recipe(0.6)
    compiled = _compile(boundary, 4)

    torch.testing.assert_close(
        compiled.constraint_matrix,
        torch.ones((1, 4), dtype=DTYPE),
    )
    torch.testing.assert_close(
        compiled.constraint_target,
        torch.tensor([0.6], dtype=DTYPE),
    )
    assert compiled.control_matrix.shape == (0, 4)

    functional = _functional(4)
    result = solve_dense(functional.problem(compiled))
    chemical_potential = functional.chemical_potential(result.state)
    torch.testing.assert_close(result.state.sum(), torch.tensor(0.6, dtype=DTYPE))
    torch.testing.assert_close(
        chemical_potential,
        torch.full_like(chemical_potential, -result.multipliers[0]),
        atol=2.0e-14,
        rtol=0.0,
    )


def test_multiple_fragment_charge_targets_are_independent_equalities():
    boundary = fixed_fragment_charges_recipe(
        {"solvent": [0, 1], "ion": [2, 3, 4]},
        {"solvent": -0.25, "ion": 0.75},
    )
    compiled = _compile(boundary, 5)
    expected = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]],
        dtype=DTYPE,
    )
    torch.testing.assert_close(compiled.constraint_matrix, expected)

    functional = _functional(5)
    result = solve_dense(functional.problem(compiled))
    torch.testing.assert_close(
        expected @ result.state,
        torch.tensor([-0.25, 0.75], dtype=DTYPE),
    )
    chemical_potential = functional.chemical_potential(result.state)
    torch.testing.assert_close(
        chemical_potential[:2],
        torch.full((2,), -result.multipliers[0], dtype=DTYPE),
        atol=2.0e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        chemical_potential[2:],
        torch.full((3,), -result.multipliers[1], dtype=DTYPE),
        atol=2.0e-14,
        rtol=0.0,
    )


def test_two_fixed_electrode_potentials_are_legendre_controls_not_equalities():
    potentials = {"left": -0.4, "right": 0.7}
    boundary = fixed_electrode_potentials_recipe(
        {"left": [0, 1], "right": [2, 3]},
        potentials,
    )
    compiled = _compile(boundary, 4)
    expected_control = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
        dtype=DTYPE,
    )
    assert compiled.constraint_matrix.shape == (0, 4)
    torch.testing.assert_close(compiled.control_matrix, expected_control)
    assert all(row.category == "control" for row in compiled.rows)

    functional = _functional(4)
    problem = functional.problem(compiled)
    torch.testing.assert_close(problem.hessian, functional.hessian)
    torch.testing.assert_close(
        problem.linear,
        functional.electronegativity
        - expected_control.transpose(0, 1)
        @ torch.tensor([-0.4, 0.7], dtype=DTYPE),
    )
    result = solve_dense(problem)
    chemical_potential = functional.chemical_potential(result.state)
    torch.testing.assert_close(
        chemical_potential[:2],
        torch.full((2,), potentials["left"], dtype=DTYPE),
        atol=2.0e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        chemical_potential[2:],
        torch.full((2,), potentials["right"], dtype=DTYPE),
        atol=2.0e-14,
        rtol=0.0,
    )


def test_floating_conductor_has_fixed_charge_and_kkt_determined_potential():
    boundary = floating_conductor_recipe("particle", [0, 1, 2], -0.3)
    compiled = _compile(boundary, 3)
    assert compiled.control_matrix.shape == (0, 3)

    functional = _functional(3)
    result = solve_dense(functional.problem(compiled))
    chemical_potential = functional.chemical_potential(result.state)
    torch.testing.assert_close(result.state[:3].sum(), torch.tensor(-0.3, dtype=DTYPE))
    torch.testing.assert_close(
        chemical_potential[:3],
        torch.full((3,), -result.multipliers[0], dtype=DTYPE),
        atol=2.0e-14,
        rtol=0.0,
    )


def test_mixed_electrodes_and_variable_electrolyte_support_optional_fixed_sum():
    vector = torch.tensor(
        [0.7, -0.2, 0.4, -0.5, 0.9, -0.6, 0.3], dtype=DTYPE
    )
    interaction = 0.12 * torch.outer(vector, vector)
    functional = _functional(7, interaction=interaction)
    groups = {"left": [0, 1], "right": [2, 3]}
    voltage = {"left": -0.5, "right": 0.6}

    free_boundary = mixed_electrode_electrolyte_recipe(
        groups,
        voltage,
        "electrolyte",
        [4, 5, 6],
        electrolyte_potential=0.0,
    )
    free_compiled = _compile(free_boundary, 7)
    assert free_compiled.constraint_matrix.shape == (0, 7)
    assert free_compiled.control_matrix.shape == (3, 7)

    fixed_boundary = mixed_electrode_electrolyte_recipe(
        groups,
        voltage,
        "electrolyte",
        [4, 5, 6],
        electrolyte_total_charge=0.2,
    )
    fixed_compiled = _compile(fixed_boundary, 7)
    result = solve_dense(functional.problem(fixed_compiled))
    chemical_potential = functional.chemical_potential(result.state)

    torch.testing.assert_close(result.state[4:].sum(), torch.tensor(0.2, dtype=DTYPE))
    torch.testing.assert_close(
        chemical_potential[:2],
        torch.full((2,), voltage["left"], dtype=DTYPE),
        atol=3.0e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        chemical_potential[2:4],
        torch.full((2,), voltage["right"], dtype=DTYPE),
        atol=3.0e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        chemical_potential[4:],
        torch.full((3,), -result.multipliers[0], dtype=DTYPE),
        atol=3.0e-14,
        rtol=0.0,
    )

    zero_voltage = mixed_electrode_electrolyte_recipe(
        groups,
        {"left": 0.0, "right": 0.0},
        "electrolyte",
        [4, 5, 6],
        electrolyte_total_charge=0.2,
    )
    zero_result = solve_dense(functional.problem(_compile(zero_voltage, 7)))
    assert not torch.allclose(result.state[4:], zero_result.state[4:])


def test_neutral_two_electrode_recipe_accepts_explicit_zero_mean_gauge():
    boundary = fixed_electrode_potentials_recipe(
        {"left": [0, 1], "right": [2, 3]},
        {"left": -0.5, "right": 0.5},
        neutrality=True,
        gauge=GaugeSpec(mode="zero_mean"),
    )
    compiled = _compile(boundary, 4)

    assert compiled.control_gauge_basis.shape == (2, 1)
    torch.testing.assert_close(
        compiled.control_gauge_basis.transpose(0, 1) @ compiled.control_value,
        torch.zeros(1, dtype=DTYPE),
        atol=2.0e-15,
        rtol=0.0,
    )
    result = solve_dense(_functional(4).problem(compiled))
    chemical_potential = _functional(4).chemical_potential(result.state)
    torch.testing.assert_close(
        chemical_potential[2:].mean() - chemical_potential[:2].mean(),
        torch.tensor(1.0, dtype=DTYPE),
        atol=2.0e-14,
        rtol=0.0,
    )


def test_open_electrolyte_requires_an_explicit_reservoir_potential():
    with pytest.raises(ValueError, match="electron-open electrolyte"):
        mixed_electrode_electrolyte_recipe(
            {"left": [0], "right": [1]},
            {"left": -0.2, "right": 0.2},
            "electrolyte",
            [2, 3],
        )
    with pytest.raises(ValueError, match="explicit opt-in"):
        mixed_electrode_electrolyte_recipe(
            {"left": [0], "right": [1]},
            {"left": -0.2, "right": 0.2},
            "electrolyte",
            [2, 3],
            neutrality=True,
            gauge=GaugeSpec(mode="zero_mean"),
        )

    boundary = mixed_electrode_electrolyte_recipe(
        {"left": [0], "right": [1]},
        {"left": -0.2, "right": 0.2},
        "electrolyte",
        [2, 3],
        neutrality=True,
        allow_electrode_electrolyte_charge_transfer=True,
        gauge=GaugeSpec(mode="zero_mean"),
    )
    compiled = _compile(boundary, 4)
    assert compiled.constraint_matrix.shape == (1, 4)


def test_overlapping_charge_groups_and_nonfinite_controls_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        fixed_electrode_potentials_recipe(
            {"left": [0, 1], "right": [1, 2]},
            {"left": -0.2, "right": 0.2},
        )
    with pytest.raises(ValueError, match="finite"):
        fixed_electrode_potentials_recipe(
            {"left": [0], "right": [1]},
            {"left": float("nan"), "right": 0.2},
        )


def test_physical_wrappers_reject_missing_targets_and_unassigned_atoms():
    with pytest.raises(ValueError, match="every electrode"):
        fixed_electrode_potentials_recipe(
            {"left": [0], "right": [1]}, {"left": -0.2}
        )
    with pytest.raises(ValueError, match="every fragment"):
        fixed_fragment_charges_recipe(
            {"solvent": [0], "ion": [1]}, {"solvent": 0.0}
        )

    incomplete = fixed_electrode_potentials_recipe(
        {"left": [0], "right": [1]}, {"left": -0.2, "right": 0.2}
    )
    with pytest.raises(ValueError, match="do not cover"):
        _compile(incomplete, 3)


def test_fixed_q_and_fixed_v_are_legendre_dual_at_the_same_state():
    functional = _functional(
        4,
        interaction=torch.tensor(
            [
                [0.4, -0.1, 0.0, 0.0],
                [-0.1, 0.3, -0.05, 0.0],
                [0.0, -0.05, 0.35, -0.08],
                [0.0, 0.0, -0.08, 0.25],
            ],
            dtype=DTYPE,
        ),
    )

    def solve_at_potential(value):
        boundary = fixed_electrode_potentials_recipe(
            {"electrode": [0, 1, 2, 3]},
            {"electrode": value},
        )
        return solve_dense(functional.problem(_compile(boundary, 4)))

    potential = 0.35
    fixed_v = solve_at_potential(potential)
    fixed_q = solve_dense(
        functional.problem(
            _compile(fixed_total_charge_recipe(float(fixed_v.state.sum())), 4)
        )
    )

    torch.testing.assert_close(fixed_q.state, fixed_v.state, atol=3.0e-14, rtol=0.0)
    torch.testing.assert_close(
        -fixed_q.multipliers[0],
        torch.tensor(potential, dtype=DTYPE),
        atol=3.0e-14,
        rtol=0.0,
    )

    delta = 1.0e-5
    finite_difference = (
        solve_at_potential(potential + delta).state.sum()
        - solve_at_potential(potential - delta).state.sum()
    ) / (2.0 * delta)
    ones = torch.ones(4, dtype=DTYPE)
    analytic_capacitance = ones @ torch.linalg.solve(functional.hessian, ones)
    torch.testing.assert_close(
        finite_difference,
        analytic_capacitance,
        atol=2.0e-10,
        rtol=2.0e-10,
    )
