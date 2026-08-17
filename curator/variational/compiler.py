"""Compile declarative thermodynamic boundaries into dense linear algebra."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import torch

from .functional import QuadraticProblem
from .operators import (
    BoundaryQuadraticOperator,
    OperatorQuadraticProblem,
    VariationalOperator,
)
from .spec import (
    BoundarySpec,
    EnvironmentSpec,
    GroupSpec,
    LinearTermSpec,
    NumericValue,
    VariableLayout,
)
from .validation import (
    ThermodynamicGaugeError,
    thermodynamic_gauge_basis,
    validate_equalities,
)


@dataclass(frozen=True)
class BoundaryRow:
    category: str
    name: str
    component: int
    expression: str
    value: float
    unit: str


@dataclass(frozen=True)
class CompiledBoundary:
    """Dense boundary operators for one variable layout."""

    constraint_matrix: torch.Tensor
    constraint_target: torch.Tensor
    control_matrix: torch.Tensor
    control_value: torch.Tensor
    environment_matrix: torch.Tensor
    environment_target: torch.Tensor
    environment_stiffness: torch.Tensor
    environment_field: torch.Tensor
    control_gauge_basis: torch.Tensor
    rows: Tuple[BoundaryRow, ...]

    @property
    def n_state(self) -> int:
        return self.constraint_matrix.shape[1]

    def to_problem(
        self,
        hessian: torch.Tensor,
        linear: torch.Tensor,
        constant: Union[torch.Tensor, float] = 0.0,
    ) -> QuadraticProblem:
        """Apply Legendre and Robin terms to a base quadratic functional."""

        if hessian.shape != (self.n_state, self.n_state):
            raise ValueError("base hessian does not match the compiled layout")
        if linear.shape != (self.n_state,):
            raise ValueError("base linear term does not match the compiled layout")
        if not isinstance(constant, torch.Tensor):
            constant = hessian.new_tensor(constant)
        effective_hessian = hessian
        effective_linear = linear
        effective_constant = constant.reshape(())
        if self.control_matrix.shape[0]:
            effective_linear = (
                effective_linear
                - self.control_matrix.transpose(0, 1) @ self.control_value
            )
        if self.environment_matrix.shape[0]:
            env_matrix = self.environment_matrix
            stiffness = self.environment_stiffness
            target = self.environment_target
            effective_hessian = (
                effective_hessian
                + env_matrix.transpose(0, 1) @ stiffness @ env_matrix
            )
            effective_linear = effective_linear - env_matrix.transpose(0, 1) @ (
                stiffness @ target + self.environment_field
            )
            effective_constant = (
                effective_constant + 0.5 * target @ stiffness @ target
            )
        return QuadraticProblem(
            hessian=effective_hessian,
            linear=effective_linear,
            constraint_matrix=self.constraint_matrix,
            constraint_target=self.constraint_target,
            constant=effective_constant,
        )

    def to_operator_problem(
        self,
        operator: VariationalOperator,
        linear: torch.Tensor,
        constant: Union[torch.Tensor, float] = 0.0,
    ) -> OperatorQuadraticProblem:
        """Apply every compiled boundary term without materializing a Hessian."""

        if linear.shape != (self.n_state,):
            raise ValueError("base linear term does not match the compiled layout")
        if not isinstance(constant, torch.Tensor):
            constant = linear.new_tensor(constant)
        effective_operator = operator
        effective_linear = linear
        effective_constant = constant.reshape(())
        if self.control_matrix.shape[0]:
            effective_linear = (
                effective_linear
                - self.control_matrix.transpose(0, 1) @ self.control_value
            )
        if self.environment_matrix.shape[0]:
            effective_operator = BoundaryQuadraticOperator(
                operator,
                self.environment_matrix,
                self.environment_stiffness,
            )
            effective_linear = effective_linear - self.environment_matrix.transpose(
                0, 1
            ) @ (
                self.environment_stiffness @ self.environment_target
                + self.environment_field
            )
            effective_constant = effective_constant + 0.5 * (
                self.environment_target
                @ self.environment_stiffness
                @ self.environment_target
            )
        return OperatorQuadraticProblem(
            operator=effective_operator,
            linear=effective_linear,
            constraint_matrix=self.constraint_matrix,
            constraint_target=self.constraint_target,
            constant=effective_constant,
        )

    def render_report(self) -> str:
        if not self.rows:
            return "No thermodynamic boundary rows were compiled."
        lines = ["category | name | component | expression | value | unit"]
        lines.append("--- | --- | ---: | --- | ---: | ---")
        for row in self.rows:
            lines.append(
                "%s | %s | %d | %s | %.12g | %s"
                % (
                    row.category,
                    row.name,
                    row.component,
                    row.expression,
                    row.value,
                    row.unit,
                )
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class RaggedCompiledBoundary:
    """A batch represented as independent compiled structures."""

    structures: Tuple[CompiledBoundary, ...]
    state_offsets: torch.Tensor
    constraint_offsets: torch.Tensor


def _resolve_group(layout: VariableLayout, group: GroupSpec) -> torch.Tensor:
    device = layout.torch_device
    if group.selection == "all":
        indices = torch.arange(layout.n_entities, device=device)
    elif group.selection == "indices":
        indices = torch.tensor(group.indices, dtype=torch.long, device=device)
        if bool(((indices < 0) | (indices >= layout.n_entities)).any().item()):
            raise ValueError("group %s contains an out-of-range index" % group.name)
        if indices.unique().numel() != indices.numel():
            raise ValueError("group %s contains duplicate indices" % group.name)
    else:
        if group.field not in layout.fields:
            raise ValueError(
                "group %s requires missing layout field %s"
                % (group.name, group.field)
            )
        values = layout.fields[group.field]
        if group.selection == "mask":
            if values.ndim != 1:
                raise ValueError("mask field for group %s must be rank 1" % group.name)
            mask = values.to(dtype=torch.bool)
        elif group.selection in ("element", "molecule"):
            if values.ndim != 1:
                raise ValueError("categorical group fields must be rank 1")
            mask = torch.zeros(layout.n_entities, dtype=torch.bool, device=values.device)
            for selected in group.values:
                mask = mask | (values == selected)
        elif group.selection == "region":
            if values.ndim != 2 or values.shape[1] != len(group.lower):
                raise ValueError("region field has incompatible coordinates")
            lower = values.new_tensor(group.lower)
            upper = values.new_tensor(group.upper)
            if bool((upper <= lower).any().item()):
                raise ValueError("region upper bounds must exceed lower bounds")
            mask = ((values >= lower) & (values < upper)).all(dim=1)
        else:
            raise AssertionError("unreachable group selection")
        indices = torch.nonzero(mask, as_tuple=False).reshape(-1).to(device=device)
    if indices.numel() == 0 and not group.allow_empty:
        raise ValueError("group %s selects no entities" % group.name)
    return indices


def _component_indices(components: Tuple[int, ...], n_components: int) -> Tuple[int, ...]:
    selected = tuple(range(n_components)) if not components else components
    if not selected:
        raise ValueError("at least one component must be selected")
    if len(set(selected)) != len(selected):
        raise ValueError("components cannot contain duplicates")
    if any(component < 0 or component >= n_components for component in selected):
        raise ValueError("component index is out of range")
    return selected


def _values(value: NumericValue, count: int, tensor: torch.Tensor) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=tensor.dtype, device=tensor.device).reshape(-1)
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("boundary values must be finite")
    if result.numel() == 1:
        return result.expand(count)
    if result.numel() != count:
        raise ValueError("boundary value must be scalar or match selected components")
    return result


def _term_weights(
    term: LinearTermSpec,
    group_indices: torch.Tensor,
    layout: VariableLayout,
) -> torch.Tensor:
    if term.weight_field:
        if term.weight_field not in layout.fields:
            raise ValueError("missing weight field %s" % term.weight_field)
        all_weights = layout.fields[term.weight_field]
        if all_weights.ndim != 1:
            raise ValueError("weight fields must be rank 1")
        weights = all_weights[group_indices]
    elif term.weights:
        weights = torch.tensor(
            term.weights,
            dtype=layout.dtype,
            device=layout.torch_device,
        )
        if weights.numel() == layout.n_entities:
            weights = weights[group_indices]
        elif weights.numel() != group_indices.numel():
            raise ValueError(
                "literal weights must match the group or the complete entity layout"
            )
    else:
        weights = torch.ones(
            group_indices.numel(), dtype=layout.dtype, device=layout.torch_device
        )
    return weights * term.coefficient


def _observable_rows(
    terms: Tuple[LinearTermSpec, ...],
    components: Tuple[int, ...],
    layout: VariableLayout,
    groups: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, List[str]]:
    if not terms:
        raise ValueError("linear observable requires at least one term")
    component_indices = _component_indices(components, layout.variable.components)
    matrix = torch.zeros(
        (len(component_indices), layout.n_state),
        dtype=layout.dtype,
        device=layout.torch_device,
    )
    expressions = []
    for row, component in enumerate(component_indices):
        pieces = []
        for term in terms:
            if term.group not in groups:
                raise ValueError("boundary references unknown group %s" % term.group)
            indices = groups[term.group]
            weights = _term_weights(term, indices, layout)
            flat_indices = indices * layout.variable.components + component
            matrix[row].index_add_(0, flat_indices, weights)
            pieces.append("%+.12g*sum(%s[%d])" % (term.coefficient, term.group, component))
        expressions.append(" ".join(pieces).lstrip("+"))
    return matrix, expressions


def _stiffness_matrix(
    environment: EnvironmentSpec,
    count: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    stiffness = torch.as_tensor(
        environment.stiffness,
        dtype=reference.dtype,
        device=reference.device,
    )
    if stiffness.ndim == 0:
        stiffness = stiffness * torch.eye(
            count, dtype=reference.dtype, device=reference.device
        )
    elif stiffness.ndim == 1:
        if stiffness.numel() != count:
            raise ValueError("diagonal environment stiffness has the wrong size")
        stiffness = torch.diag(stiffness)
    elif stiffness.shape != (count, count):
        raise ValueError("environment stiffness matrix has the wrong shape")
    if not bool(torch.isfinite(stiffness).all().item()):
        raise ValueError("environment stiffness must be finite")
    if not torch.allclose(stiffness, stiffness.transpose(0, 1)):
        raise ValueError("environment stiffness must be symmetric")
    if bool((torch.linalg.eigvalsh(stiffness) < -1.0e-12).any().item()):
        raise ValueError("environment stiffness must be positive semidefinite")
    return stiffness


def compile_boundary(
    boundary: BoundarySpec,
    layout: VariableLayout,
) -> CompiledBoundary:
    """Compile a boundary for one structure using only public specifications."""

    if boundary.variable != layout.variable:
        raise ValueError("boundary variable does not match the supplied layout")
    group_specs = boundary.groups or (GroupSpec.all(),)
    groups = {group.name: _resolve_group(layout, group) for group in group_specs}
    if boundary.partition_groups:
        coverage = torch.zeros(
            layout.n_entities, dtype=torch.int64, device=layout.torch_device
        )
        for name in boundary.partition_groups:
            indices = groups[name]
            coverage.index_add_(0, indices, torch.ones_like(indices))
        if bool((coverage == 0).any().item()):
            missing = torch.nonzero(coverage == 0, as_tuple=False).reshape(-1)
            raise ValueError(
                "boundary groups do not cover entity indices %s"
                % missing.detach().cpu().tolist()
            )
        if bool((coverage > 1).any().item()):
            overlapping = torch.nonzero(coverage > 1, as_tuple=False).reshape(-1)
            raise ValueError(
                "boundary groups overlap at entity indices %s"
                % overlapping.detach().cpu().tolist()
            )
    reference = torch.empty((), dtype=layout.dtype, device=layout.torch_device)
    constraint_blocks = []
    constraint_targets = []
    control_blocks = []
    control_values = []
    environment_blocks = []
    environment_targets = []
    environment_fields = []
    stiffness_blocks = []
    report_rows = []

    for constraint in boundary.equalities:
        rows, expressions = _observable_rows(
            constraint.terms, constraint.components, layout, groups
        )
        target = _values(constraint.target, rows.shape[0], reference)
        if constraint.unit and constraint.unit != layout.variable.unit:
            raise ValueError(
                "constraint %s uses %s but variable observable uses %s"
                % (constraint.name, constraint.unit, layout.variable.unit)
            )
        constraint_blocks.append(rows)
        constraint_targets.append(target)
        components = _component_indices(
            constraint.components, layout.variable.components
        )
        for component, expression, value in zip(components, expressions, target):
            report_rows.append(
                BoundaryRow(
                    "equality",
                    constraint.name,
                    component,
                    expression,
                    float(value.detach().cpu()),
                    constraint.unit or layout.variable.unit,
                )
            )

    for control in boundary.controls:
        rows, expressions = _observable_rows(
            control.terms, control.components, layout, groups
        )
        value = _values(control.value, rows.shape[0], reference)
        if control.unit and control.unit != layout.variable.conjugate_unit:
            raise ValueError(
                "control %s uses %s but variable conjugate uses %s"
                % (control.name, control.unit, layout.variable.conjugate_unit)
            )
        control_blocks.append(rows)
        control_values.append(value)
        components = _component_indices(control.components, layout.variable.components)
        for component, expression, scalar in zip(components, expressions, value):
            report_rows.append(
                BoundaryRow(
                    "control",
                    control.name,
                    component,
                    "-(%s)" % expression,
                    float(scalar.detach().cpu()),
                    control.unit or layout.variable.conjugate_unit,
                )
            )

    for environment in boundary.environment:
        rows, expressions = _observable_rows(
            environment.terms, environment.components, layout, groups
        )
        target = _values(environment.target, rows.shape[0], reference)
        fixed_field = _values(environment.field, rows.shape[0], reference)
        stiffness = _stiffness_matrix(environment, rows.shape[0], reference)
        if (
            environment.observable_unit
            and environment.observable_unit != layout.variable.unit
        ):
            raise ValueError("environment observable unit does not match variable")
        environment_blocks.append(rows)
        environment_targets.append(target)
        environment_fields.append(fixed_field)
        stiffness_blocks.append(stiffness)
        components = _component_indices(
            environment.components, layout.variable.components
        )
        for component, expression, scalar in zip(components, expressions, target):
            report_rows.append(
                BoundaryRow(
                    "environment",
                    environment.name,
                    component,
                    expression,
                    float(scalar.detach().cpu()),
                    environment.observable_unit or layout.variable.unit,
                )
            )

    def stack(blocks: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(blocks, dim=0) if blocks else reference.new_zeros((0, layout.n_state))

    constraints = stack(constraint_blocks)
    targets = (
        torch.cat(constraint_targets)
        if constraint_targets
        else reference.new_zeros((0,))
    )
    validate_equalities(constraints, targets)
    controls = stack(control_blocks)
    controls_value = (
        torch.cat(control_values) if control_values else reference.new_zeros((0,))
    )
    gauge_basis = thermodynamic_gauge_basis(constraints, controls)
    if gauge_basis.shape[1]:
        gauge = boundary.gauge
        if gauge is None or gauge.mode == "error":
            raise ThermodynamicGaugeError(
                "intensive controls contain %d thermodynamic gauge direction(s); "
                "declare GaugeSpec(mode='reference', 'zero_mean', or 'allow')"
                % gauge_basis.shape[1]
            )
        gauge_component = gauge_basis.transpose(0, 1) @ controls_value
        if gauge.mode == "zero_mean" and bool(
            (gauge_component.abs() > gauge.tolerance).any().item()
        ):
            raise ThermodynamicGaugeError(
                "control values do not satisfy the declared zero-mean gauge"
            )
        if gauge.mode == "reference":
            if gauge_basis.shape[1] != 1:
                raise ThermodynamicGaugeError(
                    "a single reference_index cannot resolve a multidimensional gauge"
                )
            index = gauge.reference_index
            if index is None or index < 0 or index >= controls_value.numel():
                raise ThermodynamicGaugeError("reference_index is out of range")
            if bool((gauge_basis[index].abs() <= gauge.tolerance).all().item()):
                raise ThermodynamicGaugeError(
                    "reference_index does not constrain the thermodynamic gauge"
                )
            if abs(float(controls_value[index].detach().cpu())) > gauge.tolerance:
                raise ThermodynamicGaugeError(
                    "the referenced control value must be zero"
                )
    environments = stack(environment_blocks)
    env_target = (
        torch.cat(environment_targets)
        if environment_targets
        else reference.new_zeros((0,))
    )
    env_field = (
        torch.cat(environment_fields)
        if environment_fields
        else reference.new_zeros((0,))
    )
    if stiffness_blocks:
        stiffness = torch.block_diag(*stiffness_blocks)
    else:
        stiffness = reference.new_zeros((0, 0))
    return CompiledBoundary(
        constraint_matrix=constraints,
        constraint_target=targets,
        control_matrix=controls,
        control_value=controls_value,
        environment_matrix=environments,
        environment_target=env_target,
        environment_stiffness=stiffness,
        environment_field=env_field,
        control_gauge_basis=gauge_basis,
        rows=tuple(report_rows),
    )


def compile_boundary_batch(
    boundaries: Union[BoundarySpec, Sequence[BoundarySpec]],
    layouts: Sequence[VariableLayout],
) -> RaggedCompiledBoundary:
    """Compile a ragged batch without padding physical constraint matrices."""

    if isinstance(boundaries, BoundarySpec):
        boundary_list = [boundaries] * len(layouts)
    else:
        boundary_list = list(boundaries)
    if len(boundary_list) != len(layouts):
        raise ValueError("one boundary is required per layout")
    compiled = tuple(
        compile_boundary(boundary, layout)
        for boundary, layout in zip(boundary_list, layouts)
    )
    state_counts = torch.tensor([item.n_state for item in compiled], dtype=torch.long)
    constraint_counts = torch.tensor(
        [item.constraint_matrix.shape[0] for item in compiled], dtype=torch.long
    )
    state_offsets = torch.cat((torch.zeros(1, dtype=torch.long), state_counts.cumsum(0)))
    constraint_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), constraint_counts.cumsum(0))
    )
    return RaggedCompiledBoundary(
        structures=compiled,
        state_offsets=state_offsets,
        constraint_offsets=constraint_offsets,
    )
