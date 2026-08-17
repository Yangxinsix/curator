import pytest
import torch

from curator.variational import (
    BoundarySpec,
    ConstraintSpec,
    ControlSpec,
    EnvironmentSpec,
    GaugeSpec,
    GroupSpec,
    LinearTermSpec,
    VariableLayout,
    VariableSpec,
    compile_boundary,
    compile_boundary_batch,
    solve_dense,
)


DTYPE = torch.float64


def _charge_variable():
    return VariableSpec(
        name="q",
        kind="atomic_charge",
        components=1,
        unit="e",
        conjugate_unit="V",
    )


def test_group_and_fragment_constraints_compile_to_physical_rows():
    variable = _charge_variable()
    layout = VariableLayout(
        variable=variable,
        n_entities=5,
        fields={
            "left_mask": torch.tensor([True, True, False, False, False]),
            "molecule_id": torch.tensor([0, 0, 1, 1, 1]),
        },
    )
    boundary = BoundarySpec(
        variable=variable,
        groups=(
            GroupSpec.from_mask("left", "left_mask"),
            GroupSpec.from_molecules("electrolyte", [1]),
        ),
        equalities=(
            ConstraintSpec.group_sum("left_charge", "left", 0.4, unit="e"),
            ConstraintSpec.group_sum(
                "electrolyte_charge", "electrolyte", -0.4, unit="e"
            ),
        ),
    )

    compiled = compile_boundary(boundary, layout)

    torch.testing.assert_close(
        compiled.constraint_matrix,
        torch.tensor(
            [[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]],
            dtype=DTYPE,
        ),
    )
    torch.testing.assert_close(
        compiled.constraint_target, torch.tensor([0.4, -0.4], dtype=DTYPE)
    )
    assert "left_charge" in compiled.render_report()
    assert "electrolyte_charge" in compiled.render_report()


def test_fixed_intensive_and_robin_are_compiled_with_the_frozen_signs():
    variable = _charge_variable()
    layout = VariableLayout(variable=variable, n_entities=4)
    boundary = BoundarySpec(
        variable=variable,
        groups=(
            GroupSpec.from_indices("left", [0, 1]),
            GroupSpec.from_indices("right", [2, 3]),
        ),
        controls=(
            ControlSpec.group_potential("left_V", "left", 0.7, unit="V"),
        ),
        environment=(
            EnvironmentSpec(
                name="right_reservoir",
                terms=(LinearTermSpec("right"),),
                target=-0.3,
                stiffness=1.5,
                field=-0.2,
                observable_unit="e",
            ),
        ),
    )
    compiled = compile_boundary(boundary, layout)
    base_hessian = torch.diag(torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=DTYPE))
    base_linear = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=DTYPE)

    problem = compiled.to_problem(base_hessian, base_linear, constant=0.6)

    left = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=DTYPE)
    right = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=DTYPE)
    expected_hessian = base_hessian + 1.5 * torch.outer(right, right)
    expected_linear = base_linear - 0.7 * left - right * (1.5 * -0.3 - 0.2)
    expected_constant = torch.tensor(0.6 + 0.5 * 1.5 * 0.3**2, dtype=DTYPE)
    torch.testing.assert_close(problem.hessian, expected_hessian)
    torch.testing.assert_close(problem.linear, expected_linear)
    torch.testing.assert_close(problem.constant, expected_constant)

    result = solve_dense(problem)
    torch.testing.assert_close(
        result.state,
        torch.linalg.solve(expected_hessian, -expected_linear),
        atol=2.0e-12,
        rtol=0.0,
    )


def test_vector_layout_uses_entity_major_flattening_without_solver_changes():
    variable = VariableSpec(
        name="mu",
        kind="induced_dipole",
        components=3,
        unit="e*angstrom",
        conjugate_unit="V/angstrom",
        component_names=("x", "y", "z"),
    )
    layout = VariableLayout(variable=variable, n_entities=2)
    boundary = BoundarySpec(
        variable=variable,
        groups=(GroupSpec.all(),),
        equalities=(
            ConstraintSpec.group_sum(
                "total_dipole", "all", [0.2, -0.1, 0.4], unit="e*angstrom"
            ),
        ),
    )

    compiled = compile_boundary(boundary, layout)

    torch.testing.assert_close(
        compiled.constraint_matrix,
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ],
            dtype=DTYPE,
        ),
    )
    problem = compiled.to_problem(torch.eye(6, dtype=DTYPE), torch.zeros(6, dtype=DTYPE))
    result = solve_dense(problem)
    torch.testing.assert_close(
        result.state.reshape(2, 3).sum(dim=0),
        torch.tensor([0.2, -0.1, 0.4], dtype=DTYPE),
    )


def test_ragged_batch_preserves_per_structure_state_and_constraint_counts():
    variable = _charge_variable()
    layouts = [
        VariableLayout(variable=variable, n_entities=2),
        VariableLayout(variable=variable, n_entities=5),
    ]
    boundaries = [
        BoundarySpec(
            variable=variable,
            groups=(GroupSpec.all(),),
            equalities=(ConstraintSpec.group_sum("Q", "all", 0.0, unit="e"),),
        ),
        BoundarySpec(variable=variable),
    ]

    batch = compile_boundary_batch(boundaries, layouts)

    torch.testing.assert_close(batch.state_offsets, torch.tensor([0, 2, 7]))
    torch.testing.assert_close(batch.constraint_offsets, torch.tensor([0, 1, 1]))
    assert batch.structures[0].constraint_matrix.shape == (1, 2)
    assert batch.structures[1].constraint_matrix.shape == (0, 5)


def test_selectors_cover_element_region_and_weighted_sum():
    variable = _charge_variable()
    layout = VariableLayout(
        variable=variable,
        n_entities=4,
        fields={
            "atomic_numbers": torch.tensor([1, 8, 1, 6]),
            "positions": torch.tensor(
                [[-1.0, 0.0], [0.2, 0.1], [0.8, 0.4], [2.0, 0.0]],
                dtype=DTYPE,
            ),
            "weights": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DTYPE),
        },
    )
    boundary = BoundarySpec(
        variable=variable,
        groups=(
            GroupSpec.from_elements("hydrogen", [1]),
            GroupSpec.from_region("middle", [0.0, -0.5], [1.0, 0.5]),
        ),
        controls=(
            ControlSpec(
                name="weighted_H",
                terms=(LinearTermSpec("hydrogen", weight_field="weights"),),
                value=0.1,
                unit="V",
            ),
            ControlSpec.group_potential("middle_V", "middle", -0.2, unit="V"),
        ),
    )

    compiled = compile_boundary(boundary, layout)

    torch.testing.assert_close(
        compiled.control_matrix,
        torch.tensor([[1.0, 0.0, 3.0, 0.0], [0.0, 1.0, 1.0, 0.0]], dtype=DTYPE),
    )


def test_constraint_constructors_cover_weighted_sum_and_fixed_component():
    variable = VariableSpec(
        name="mu",
        kind="induced_dipole",
        components=3,
        unit="e*angstrom",
        conjugate_unit="V/angstrom",
    )
    layout = VariableLayout(variable=variable, n_entities=3)
    boundary = BoundarySpec(
        variable=variable,
        groups=(
            GroupSpec.all(),
            GroupSpec.from_indices("site", [1]),
        ),
        equalities=(
            ConstraintSpec.weighted_sum(
                "weighted_x",
                "all",
                0.5,
                weights=[1.0, 2.0, -1.0],
                components=[0],
                unit="e*angstrom",
            ),
            ConstraintSpec.fixed_component(
                "site_z", "site", component=2, target=-0.2, unit="e*angstrom"
            ),
        ),
    )

    compiled = compile_boundary(boundary, layout)

    torch.testing.assert_close(
        compiled.constraint_matrix,
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 2.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=DTYPE,
        ),
    )
    torch.testing.assert_close(
        compiled.constraint_target, torch.tensor([0.5, -0.2], dtype=DTYPE)
    )

    with pytest.raises(ValueError, match="literal weights"):
        compile_boundary(
            BoundarySpec(
                variable=variable,
                groups=(GroupSpec.from_indices("many", [0, 1]),),
                equalities=(
                    ConstraintSpec.fixed_component(
                        "not_a_site", "many", component=0, target=0.0
                    ),
                ),
            ),
            layout,
        )


def test_empty_groups_units_and_conflicting_constraints_fail_before_solve():
    variable = _charge_variable()
    layout = VariableLayout(
        variable=variable,
        n_entities=2,
        fields={"none": torch.tensor([False, False])},
    )
    with pytest.raises(ValueError, match="selects no entities"):
        compile_boundary(
            BoundarySpec(
                variable=variable,
                groups=(GroupSpec.from_mask("empty", "none"),),
                equalities=(ConstraintSpec.group_sum("Q", "empty", 0.0),),
            ),
            layout,
        )

    with pytest.raises(ValueError, match="variable observable"):
        compile_boundary(
            BoundarySpec(
                variable=variable,
                groups=(GroupSpec.all(),),
                equalities=(ConstraintSpec.group_sum("Q", "all", 0.0, unit="C"),),
            ),
            layout,
        )

    with pytest.raises(ValueError, match="inconsistent"):
        compile_boundary(
            BoundarySpec(
                variable=variable,
                groups=(GroupSpec.all(),),
                equalities=(
                    ConstraintSpec.group_sum("Q0", "all", 0.0),
                    ConstraintSpec.group_sum("Q1", "all", 1.0),
                ),
            ),
            layout,
        )


def test_thermodynamic_control_gauge_requires_an_explicit_policy():
    variable = _charge_variable()
    layout = VariableLayout(variable=variable, n_entities=4)
    groups = (
        GroupSpec.from_indices("left", [0, 1]),
        GroupSpec.from_indices("right", [2, 3]),
        GroupSpec.all(),
    )
    equalities = (ConstraintSpec.group_sum("Q", "all", 0.0, unit="e"),)
    controls = (
        ControlSpec.group_potential("left_V", "left", 1.0, unit="V"),
        ControlSpec.group_potential("right_V", "right", -1.0, unit="V"),
    )

    with pytest.raises(ValueError, match="thermodynamic gauge"):
        compile_boundary(
            BoundarySpec(
                variable=variable,
                groups=groups,
                equalities=equalities,
                controls=controls,
            ),
            layout,
        )

    compiled = compile_boundary(
        BoundarySpec(
            variable=variable,
            groups=groups,
            equalities=equalities,
            controls=controls,
            gauge=GaugeSpec(mode="zero_mean"),
        ),
        layout,
    )
    assert compiled.control_gauge_basis.shape == (2, 1)

    shifted = compile_boundary(
        BoundarySpec(
            variable=variable,
            groups=groups,
            equalities=equalities,
            controls=(
                ControlSpec.group_potential("left_V", "left", 3.5, unit="V"),
                ControlSpec.group_potential("right_V", "right", 1.5, unit="V"),
            ),
            gauge=GaugeSpec(mode="allow"),
        ),
        layout,
    )
    hessian = torch.eye(4, dtype=DTYPE)
    linear = torch.zeros(4, dtype=DTYPE)
    base_result = solve_dense(compiled.to_problem(hessian, linear))
    shifted_result = solve_dense(shifted.to_problem(hessian, linear))
    torch.testing.assert_close(base_result.state, shifted_result.state)
    assert not torch.allclose(base_result.multipliers, shifted_result.multipliers)
