"""Declarative thermodynamic boundary recipes for atomic charges.

The recipes in this module only describe constraints on charge observables and
their conjugate intensive controls.  In particular, they do *not* choose a
spatial electrostatic Green function, an Ewald ``k=0`` convention, a slab
correction, or a vacuum reference.  Those choices belong to the electrostatic
operator/backend and are independent of :class:`~curator.variational.GaugeSpec`,
which only resolves redundant thermodynamic control directions.

With Curator's frozen sign convention, a potential control ``V`` on a group
contributes ``-V * sum(q_group)``.  Consequently, an unconstrained electrode
at a stationary point satisfies ``dF/dq_i = V`` for every atom in that group.
A floating conductor is different: its total charge is constrained and its
common potential is the negative KKT multiplier, not an input control.
"""

import math
from typing import Mapping, Optional, Sequence

from ..spec import (
    BoundarySpec,
    ConstraintSpec,
    ControlSpec,
    GaugeSpec,
    GroupSpec,
    VariableSpec,
)
from .charge import charge_variable_spec


_SYSTEM_GROUP = "system"


def _variable_or_default(variable: Optional[VariableSpec]) -> VariableSpec:
    result = charge_variable_spec() if variable is None else variable
    if result.kind != "atomic_charge" or result.components != 1:
        raise ValueError(
            "charge boundary recipes require a scalar atomic_charge variable"
        )
    return result


def _scalar_mapping(
    values: Optional[Mapping[str, float]],
    name: str,
) -> Mapping[str, float]:
    if values is None:
        return {}
    result = {}
    for group, value in values.items():
        if not group:
            raise ValueError("%s group names cannot be empty" % name)
        try:
            scalar = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError("%s values must be scalar numbers" % name) from error
        if not math.isfinite(scalar):
            raise ValueError("%s values must be finite" % name)
        result[group] = scalar
    return result


def _validate_disjoint_groups(groups: Mapping[str, Sequence[int]]) -> None:
    owners = {}
    for group, indices in groups.items():
        for index in indices:
            if index in owners:
                raise ValueError(
                    "charge groups %s and %s overlap at atom index %s"
                    % (owners[index], group, index)
                )
            owners[index] = group


def charge_boundary_recipe(
    *,
    groups: Optional[Mapping[str, Sequence[int]]] = None,
    fixed_charges: Optional[Mapping[str, float]] = None,
    fixed_potentials: Optional[Mapping[str, float]] = None,
    total_charge: Optional[float] = None,
    neutrality: bool = False,
    gauge: Optional[GaugeSpec] = None,
    require_complete_groups: bool = False,
    variable: Optional[VariableSpec] = None,
) -> BoundarySpec:
    """Build a charge boundary from named groups and physical targets.

    Parameters
    ----------
    groups
        Mapping from each public group name to explicit atom indices.
    fixed_charges
        Hard extensive targets ``sum(q_group) = Q_group`` in elementary-charge
        units.  This is the correct representation of a floating conductor.
    fixed_potentials
        Intensive controls in volts.  Each entry compiles to a
        :class:`ControlSpec`, never to an equality.
    total_charge
        Optional hard total-system charge target in elementary-charge units.
    neutrality
        Convenience spelling of ``total_charge=0``.  It cannot be combined
        with an explicit ``total_charge``.
    gauge
        Explicit policy for redundant thermodynamic potential directions.
    require_complete_groups
        Require the named groups to partition every variable-charge atom when
        the boundary is compiled against a concrete layout.

    Notes
    -----
    A group cannot simultaneously have a prescribed charge and prescribed
    potential in this recipe.  Under a fixed group charge, a uniform potential
    term is constant on the feasible set; the physically useful quantity is
    instead the KKT multiplier returned by the solver.
    """

    if neutrality and total_charge is not None:
        raise ValueError("use neutrality or total_charge, not both")
    variable = _variable_or_default(variable)
    groups = {} if groups is None else dict(groups)
    fixed_charges = _scalar_mapping(fixed_charges, "fixed_charges")
    fixed_potentials = _scalar_mapping(fixed_potentials, "fixed_potentials")

    if _SYSTEM_GROUP in groups:
        raise ValueError("group name %r is reserved" % _SYSTEM_GROUP)
    _validate_disjoint_groups(groups)
    unknown = (set(fixed_charges) | set(fixed_potentials)) - set(groups)
    if unknown:
        raise ValueError(
            "charge boundary references undefined group(s): %s"
            % ", ".join(sorted(unknown))
        )
    both = set(fixed_charges) & set(fixed_potentials)
    if both:
        raise ValueError(
            "a group cannot have both fixed charge and fixed potential: %s"
            % ", ".join(sorted(both))
        )

    group_specs = []
    for name, indices in groups.items():
        group_specs.append(GroupSpec.from_indices(name, tuple(indices)))

    system_target = 0.0 if neutrality else total_charge
    if system_target is not None:
        try:
            system_target = float(system_target)
        except (TypeError, ValueError) as error:
            raise TypeError("total_charge must be a scalar number") from error
        if not math.isfinite(system_target):
            raise ValueError("total_charge must be finite")
        group_specs.append(GroupSpec.all(_SYSTEM_GROUP))

    equalities = [
        ConstraintSpec.group_sum(
            name="%s_charge" % name,
            group=name,
            target=float(target),
            unit="e",
        )
        for name, target in fixed_charges.items()
    ]
    if system_target is not None:
        equalities.append(
            ConstraintSpec.group_sum(
                name="total_charge",
                group=_SYSTEM_GROUP,
                target=float(system_target),
                unit="e",
            )
        )

    controls = tuple(
        ControlSpec.group_potential(
            name="%s_potential" % name,
            group=name,
            value=float(value),
            unit="V",
        )
        for name, value in fixed_potentials.items()
    )
    return BoundarySpec(
        variable=variable,
        groups=tuple(group_specs),
        equalities=tuple(equalities),
        controls=controls,
        gauge=gauge,
        partition_groups=tuple(groups) if require_complete_groups else (),
    )


def fixed_total_charge_recipe(
    total_charge: float,
    *,
    variable: Optional[VariableSpec] = None,
) -> BoundarySpec:
    """Fix the total charge of all variable-charge atoms."""

    return charge_boundary_recipe(
        total_charge=total_charge,
        variable=variable,
    )


def fixed_fragment_charges_recipe(
    fragment_indices: Mapping[str, Sequence[int]],
    charge_targets: Mapping[str, float],
    *,
    variable: Optional[VariableSpec] = None,
) -> BoundarySpec:
    """Fix one total charge for each named molecular or spatial fragment."""

    if not fragment_indices:
        raise ValueError("at least one fragment is required")
    if set(fragment_indices) != set(charge_targets):
        raise ValueError("every fragment must have exactly one charge target")
    return charge_boundary_recipe(
        groups=fragment_indices,
        fixed_charges=charge_targets,
        require_complete_groups=True,
        variable=variable,
    )


def fixed_electrode_potentials_recipe(
    electrode_indices: Mapping[str, Sequence[int]],
    potential_values: Mapping[str, float],
    *,
    total_charge: Optional[float] = None,
    neutrality: bool = False,
    gauge: Optional[GaugeSpec] = None,
    variable: Optional[VariableSpec] = None,
) -> BoundarySpec:
    """Apply fixed potentials to named electrode groups.

    Supplying ``neutrality=True`` makes a common shift of potentials a
    thermodynamic gauge when the electrode groups span the whole charge state.
    Such a case must declare a compatible ``GaugeSpec`` explicitly.
    """

    if not electrode_indices:
        raise ValueError("at least one electrode is required")
    if set(electrode_indices) != set(potential_values):
        raise ValueError("every electrode must have exactly one potential value")
    return charge_boundary_recipe(
        groups=electrode_indices,
        fixed_potentials=potential_values,
        total_charge=total_charge,
        neutrality=neutrality,
        gauge=gauge,
        require_complete_groups=True,
        variable=variable,
    )


def floating_conductor_recipe(
    group_name: str,
    atom_indices: Sequence[int],
    charge_target: float,
    *,
    variable: Optional[VariableSpec] = None,
) -> BoundarySpec:
    """Fix a conductor's charge and leave its common potential to the KKT solve."""

    return charge_boundary_recipe(
        groups={group_name: atom_indices},
        fixed_charges={group_name: charge_target},
        require_complete_groups=True,
        variable=variable,
    )


def mixed_electrode_electrolyte_recipe(
    electrode_indices: Mapping[str, Sequence[int]],
    potential_values: Mapping[str, float],
    electrolyte_group: str,
    electrolyte_indices: Sequence[int],
    *,
    electrolyte_total_charge: Optional[float] = None,
    electrolyte_potential: Optional[float] = None,
    total_charge: Optional[float] = None,
    neutrality: bool = False,
    allow_electrode_electrolyte_charge_transfer: bool = False,
    gauge: Optional[GaugeSpec] = None,
    variable: Optional[VariableSpec] = None,
) -> BoundarySpec:
    """Combine fixed-potential electrodes with variable electrolyte charges.

    If ``electrolyte_total_charge`` is supplied, only the electrolyte group sum
    is fixed; individual charges still equilibrate and respond.  If it is
    omitted and there is no overall charge constraint, an explicit
    ``electrolyte_potential`` is required: this is an electron-open electrolyte
    coupled to a reservoir.  With an overall charge/neutrality constraint, the
    electrolyte can instead exchange charge with the electrodes while the
    complete variable-charge system remains closed.  That cross-interface
    electron transfer must be enabled explicitly with
    ``allow_electrode_electrolyte_charge_transfer=True``.
    """

    if not electrode_indices:
        raise ValueError("at least one electrode is required")
    if set(electrode_indices) != set(potential_values):
        raise ValueError("every electrode must have exactly one potential value")
    if electrolyte_group in electrode_indices:
        raise ValueError("electrolyte group name must differ from electrode names")
    if electrolyte_total_charge is not None and electrolyte_potential is not None:
        raise ValueError(
            "electrolyte cannot have both fixed total charge and fixed potential"
        )
    if (
        electrolyte_total_charge is None
        and electrolyte_potential is None
        and total_charge is None
        and not neutrality
    ):
        raise ValueError(
            "an electron-open electrolyte requires electrolyte_potential when "
            "no overall charge constraint is present"
        )
    if (
        electrolyte_total_charge is None
        and electrolyte_potential is None
        and (total_charge is not None or neutrality)
        and not allow_electrode_electrolyte_charge_transfer
    ):
        raise ValueError(
            "electrode-electrolyte charge transfer requires explicit opt-in"
        )
    groups = dict(electrode_indices)
    groups[electrolyte_group] = tuple(electrolyte_indices)
    fixed_charges = (
        {}
        if electrolyte_total_charge is None
        else {electrolyte_group: electrolyte_total_charge}
    )
    fixed_potentials = dict(potential_values)
    if electrolyte_potential is not None:
        fixed_potentials[electrolyte_group] = electrolyte_potential
    return charge_boundary_recipe(
        groups=groups,
        fixed_charges=fixed_charges,
        fixed_potentials=fixed_potentials,
        total_charge=total_charge,
        neutrality=neutrality,
        gauge=gauge,
        require_complete_groups=True,
        variable=variable,
    )


__all__ = [
    "charge_boundary_recipe",
    "fixed_electrode_potentials_recipe",
    "fixed_fragment_charges_recipe",
    "fixed_total_charge_recipe",
    "floating_conductor_recipe",
    "mixed_electrode_electrolyte_recipe",
]
