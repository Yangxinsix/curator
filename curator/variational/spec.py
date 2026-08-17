"""Serializable specifications for internal variables and their boundaries."""

from dataclasses import dataclass, field
import math
import numbers
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch


Number = Union[int, float]
NumericValue = Union[Number, Sequence[Number]]


def _tuple(value: Optional[Sequence[Any]]) -> Tuple[Any, ...]:
    return () if value is None else tuple(value)


@dataclass(frozen=True)
class VariableSpec:
    """Layout and units of one flattened per-entity internal variable block."""

    name: str
    kind: str
    components: int = 1
    unit: str = "dimensionless"
    conjugate_unit: str = "energy"
    component_names: Tuple[str, ...] = ()
    semantics: str = "variational_internal"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("variable name cannot be empty")
        if self.components <= 0:
            raise ValueError("variable components must be positive")
        if self.component_names and len(self.component_names) != self.components:
            raise ValueError("component_names must match components")
        if self.semantics not in ("variational_internal", "predicted_auxiliary"):
            raise ValueError(
                "semantics must be variational_internal or predicted_auxiliary"
            )


@dataclass(frozen=True)
class VariableLayout:
    """One structure's variable layout and fields used by group selectors."""

    variable: VariableSpec
    n_entities: int
    fields: Mapping[str, torch.Tensor] = field(default_factory=dict)
    dtype: torch.dtype = torch.float64
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        if self.n_entities <= 0:
            raise ValueError("n_entities must be positive")
        for name, value in self.fields.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError("layout field %s must be a tensor" % name)
            if value.shape[0] != self.n_entities:
                raise ValueError(
                    "layout field %s must have n_entities leading entries" % name
                )

    @property
    def n_state(self) -> int:
        return self.n_entities * self.variable.components

    @property
    def torch_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        for value in self.fields.values():
            return value.device
        return torch.device("cpu")


@dataclass(frozen=True)
class GroupSpec:
    """A named entity selection.

    Supported selection modes are ``all``, ``indices``, ``mask``, ``element``,
    ``molecule``, and axis-aligned ``region``.
    """

    name: str
    selection: str = "all"
    indices: Tuple[int, ...] = ()
    field: Optional[str] = None
    values: Tuple[int, ...] = ()
    lower: Tuple[float, ...] = ()
    upper: Tuple[float, ...] = ()
    allow_empty: bool = False

    def __post_init__(self) -> None:
        allowed = {"all", "indices", "mask", "element", "molecule", "region"}
        if not self.name:
            raise ValueError("group name cannot be empty")
        if self.selection not in allowed:
            raise ValueError("unsupported group selection: %s" % self.selection)
        if self.selection == "indices" and not self.indices:
            raise ValueError("indices selection requires explicit indices")
        if self.selection == "indices" and any(
            isinstance(index, bool) or not isinstance(index, numbers.Integral)
            for index in self.indices
        ):
            raise ValueError("indices selection requires integer indices")
        if self.selection in ("mask", "element", "molecule", "region") and not self.field:
            raise ValueError("%s selection requires a field" % self.selection)
        if self.selection in ("element", "molecule") and not self.values:
            raise ValueError("%s selection requires values" % self.selection)
        if self.selection == "region":
            if not self.lower or len(self.lower) != len(self.upper):
                raise ValueError("region selection requires matching lower/upper bounds")

    @classmethod
    def all(cls, name: str = "all") -> "GroupSpec":
        return cls(name=name, selection="all")

    @classmethod
    def from_indices(cls, name: str, indices: Sequence[int]) -> "GroupSpec":
        return cls(name=name, selection="indices", indices=tuple(indices))

    @classmethod
    def from_mask(cls, name: str, field: str) -> "GroupSpec":
        return cls(name=name, selection="mask", field=field)

    @classmethod
    def from_elements(
        cls,
        name: str,
        values: Sequence[int],
        field: str = "atomic_numbers",
    ) -> "GroupSpec":
        return cls(
            name=name,
            selection="element",
            field=field,
            values=tuple(values),
        )

    @classmethod
    def from_molecules(
        cls,
        name: str,
        values: Sequence[int],
        field: str = "molecule_id",
    ) -> "GroupSpec":
        return cls(
            name=name,
            selection="molecule",
            field=field,
            values=tuple(values),
        )

    @classmethod
    def from_region(
        cls,
        name: str,
        lower: Sequence[float],
        upper: Sequence[float],
        field: str = "positions",
    ) -> "GroupSpec":
        return cls(
            name=name,
            selection="region",
            field=field,
            lower=tuple(lower),
            upper=tuple(upper),
        )


@dataclass(frozen=True)
class LinearTermSpec:
    """One group contribution to a linear observable."""

    group: str
    coefficient: float = 1.0
    weights: Tuple[float, ...] = ()
    weight_field: Optional[str] = None

    def __post_init__(self) -> None:
        if self.weights and self.weight_field:
            raise ValueError("use either literal weights or weight_field, not both")


@dataclass(frozen=True)
class ConstraintSpec:
    """A hard extensive boundary ``observable(x) = target``."""

    name: str
    terms: Tuple[LinearTermSpec, ...]
    target: NumericValue
    components: Tuple[int, ...] = ()
    unit: Optional[str] = None
    kind: str = "linear_equality"

    @classmethod
    def group_sum(
        cls,
        name: str,
        group: str,
        target: NumericValue,
        components: Sequence[int] = (),
        unit: Optional[str] = None,
    ) -> "ConstraintSpec":
        return cls(
            name=name,
            terms=(LinearTermSpec(group=group),),
            target=target,
            components=tuple(components),
            unit=unit,
            kind="group_sum",
        )

    @classmethod
    def difference(
        cls,
        name: str,
        positive_group: str,
        negative_group: str,
        target: NumericValue,
        components: Sequence[int] = (),
        unit: Optional[str] = None,
    ) -> "ConstraintSpec":
        return cls(
            name=name,
            terms=(
                LinearTermSpec(group=positive_group),
                LinearTermSpec(group=negative_group, coefficient=-1.0),
            ),
            target=target,
            components=tuple(components),
            unit=unit,
            kind="difference",
        )

    @classmethod
    def weighted_sum(
        cls,
        name: str,
        group: str,
        target: NumericValue,
        weights: Sequence[float] = (),
        weight_field: Optional[str] = None,
        components: Sequence[int] = (),
        unit: Optional[str] = None,
    ) -> "ConstraintSpec":
        return cls(
            name=name,
            terms=(
                LinearTermSpec(
                    group=group,
                    weights=tuple(weights),
                    weight_field=weight_field,
                ),
            ),
            target=target,
            components=tuple(components),
            unit=unit,
            kind="weighted_sum",
        )

    @classmethod
    def fixed_component(
        cls,
        name: str,
        group: str,
        component: int,
        target: NumericValue,
        unit: Optional[str] = None,
    ) -> "ConstraintSpec":
        """Fix one component of a one-entity group.

        The compiler intentionally validates the group cardinality through
        explicit weights: selecting multiple entities would otherwise mean a
        component *sum*, not a fixed per-entity component.
        """

        return cls(
            name=name,
            terms=(LinearTermSpec(group=group, weights=(1.0,)),),
            target=target,
            components=(component,),
            unit=unit,
            kind="fixed_component",
        )


@dataclass(frozen=True)
class ControlSpec:
    """A fixed intensive boundary contributing ``-value.T @ observable``."""

    name: str
    terms: Tuple[LinearTermSpec, ...]
    value: NumericValue
    components: Tuple[int, ...] = ()
    unit: Optional[str] = None
    kind: str = "conjugate_control"

    @classmethod
    def group_potential(
        cls,
        name: str,
        group: str,
        value: NumericValue,
        components: Sequence[int] = (),
        unit: Optional[str] = None,
    ) -> "ControlSpec":
        return cls(
            name=name,
            terms=(LinearTermSpec(group=group),),
            value=value,
            components=tuple(components),
            unit=unit,
        )


@dataclass(frozen=True)
class EnvironmentSpec:
    """A Robin term around a reference extensive observable.

    Its energy is ``0.5 * (y-target).T @ stiffness @ (y-target)`` plus the
    optional fixed field ``-field.T @ y``.
    """

    name: str
    terms: Tuple[LinearTermSpec, ...]
    target: NumericValue
    stiffness: Union[Number, Sequence[Number], Sequence[Sequence[Number]]]
    field: NumericValue = 0.0
    components: Tuple[int, ...] = ()
    observable_unit: Optional[str] = None
    stiffness_unit: Optional[str] = None
    kind: str = "robin"


@dataclass(frozen=True)
class GaugeSpec:
    """Policy for intensive-control directions that do not change the state."""

    mode: str
    reference_index: Optional[int] = None
    tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.mode not in ("allow", "reference", "zero_mean", "error"):
            raise ValueError("unsupported thermodynamic gauge mode")
        if self.mode == "reference" and self.reference_index is None:
            raise ValueError("reference gauge requires reference_index")
        if not math.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError("gauge tolerance must be finite and non-negative")


@dataclass(frozen=True)
class BoundarySpec:
    """Complete thermodynamic boundary for one variable block."""

    variable: VariableSpec
    groups: Tuple[GroupSpec, ...] = ()
    equalities: Tuple[ConstraintSpec, ...] = ()
    controls: Tuple[ControlSpec, ...] = ()
    environment: Tuple[EnvironmentSpec, ...] = ()
    gauge: Optional[GaugeSpec] = None
    partition_groups: Tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        names = [group.name for group in self.groups]
        if len(set(names)) != len(names):
            raise ValueError("group names must be unique")
        if len(set(self.partition_groups)) != len(self.partition_groups):
            raise ValueError("partition_groups cannot contain duplicates")
        unknown_partition = set(self.partition_groups) - set(names)
        if unknown_partition:
            raise ValueError(
                "partition_groups reference unknown groups: %s"
                % sorted(unknown_partition)
            )
        if self.schema_version != 1:
            raise ValueError("unsupported boundary schema version")
