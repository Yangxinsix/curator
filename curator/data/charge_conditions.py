"""ASE configuration adapter for Curator's existing charge boundary recipes."""

from collections.abc import Mapping
from dataclasses import dataclass
import math

from curator.variational import BoundarySpec, GaugeSpec, VariableLayout, compile_boundary
from curator.variational.variables.charge import charge_variable_spec
from curator.variational.variables.charge_boundary import (
    fixed_electrode_potentials_recipe,
    fixed_total_charge_recipe,
)


@dataclass(frozen=True)
class ChargeBoundary:
    """Choose fixed charge or fixed potential; values use e and V.

    A ``*_key`` reads ASE ``Atoms.info`` instead of a constant. Potential
    groups are explicit atom-index lists, optionally supplied per frame.
    With no groups, the potential applies to all atoms. A potential is the
    conjugate control in ``G = U - sum(V_i q_i)``.
    """

    type: str
    total_charge: float | None = None
    total_charge_key: str | None = None
    potential: float | Mapping | None = None
    potential_key: str | None = None
    groups: Mapping | None = None
    groups_key: str | None = None
    gauge: Mapping | None = None

    def __post_init__(self):
        if self.type not in {"none", "fixed_charge", "fixed_potential"}:
            raise ValueError("boundary.type must be none, fixed_charge or fixed_potential")
        if self.type == "none":
            if any(value is not None for name, value in vars(self).items() if name != "type"):
                raise ValueError("boundary.type=none cannot include charge, potential, groups or gauge")
            return
        for name in ("total_charge", "potential", "groups"):
            if getattr(self, name) is not None and getattr(self, name + "_key") is not None:
                raise ValueError(f"choose boundary.{name} or {name}_key, not both")
        if self.type == "fixed_charge":
            if self.total_charge is None and self.total_charge_key is None:
                raise ValueError("fixed_charge requires an explicit total_charge or total_charge_key")
            if any(value is not None for value in
                   (self.potential, self.potential_key, self.groups, self.groups_key, self.gauge)):
                raise ValueError("fixed_charge uses only a total-charge boundary")
        elif self.potential is None and self.potential_key is None:
            raise ValueError("fixed_potential requires potential or potential_key")

    def _value(self, frame, name):
        key = getattr(self, name + "_key")
        if key is None:
            return getattr(self, name)
        if key not in frame.info or frame.info[key] is None:
            raise ValueError(f"missing boundary input {name}: ASE info[{key!r}]")
        return frame.info[key]

    def build(self, frame):
        if self.type == "none":
            return BoundarySpec(charge_variable_spec()), None
        charge = self._value(frame, "total_charge")
        if charge is not None:
            charge = float(charge)
            if not math.isfinite(charge):
                raise ValueError("total charge must be finite")
        if self.type == "fixed_charge":
            boundary = fixed_total_charge_recipe(charge)
        else:
            groups = self._value(frame, "groups")
            groups = {"all": tuple(range(len(frame)))} if groups is None else dict(groups)
            potential = self._value(frame, "potential")
            values = dict(potential) if isinstance(potential, Mapping) else {
                name: float(potential) for name in groups}
            # Allow a redundant common potential without changing its energy
            # reference. The model removes that mode from network inputs.
            gauge = GaugeSpec(**dict(self.gauge)) if self.gauge is not None else GaugeSpec("allow")
            boundary = fixed_electrode_potentials_recipe(
                groups, values, total_charge=charge, gauge=gauge)
        compile_boundary(boundary, VariableLayout(boundary.variable, len(frame)))
        return boundary, charge
