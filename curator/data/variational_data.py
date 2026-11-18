"""Typed, ragged data contract for variational electronic-state labels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


def canonical_boundary_hash(spec: Dict[str, Any]) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VariationalResponse:
    input_ref: str
    output_ref: str
    value: torch.Tensor
    unit: str
    method: str
    gauge: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.value, torch.Tensor) or self.value.ndim != 2:
            raise ValueError("response value must be a rank-2 tensor")
        if not self.input_ref or not self.output_ref or not self.unit or not self.method:
            raise ValueError("response references, unit, and method are required")
        if not torch.isfinite(self.value).all():
            raise ValueError("response value must be finite")


@dataclass(frozen=True)
class VariationalCondition:
    condition_uid: str
    electron_count: Optional[torch.Tensor]
    electron_count_convention: Optional[str]
    net_charge: Optional[torch.Tensor]
    boundary_spec: Dict[str, Any]
    boundary_hash: str
    gauge: Dict[str, Any]
    labels: Dict[str, torch.Tensor]
    label_units: Dict[str, str]
    label_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    responses: Tuple[VariationalResponse, ...] = ()
    weight: float = 1.0
    provenance: Dict[str, Any] = field(default_factory=dict)
    # Numeric controls remain tensors outside the serialized boundary template.
    # The output head applies these overrides before constructing its AD graph.
    inputs: Dict[str, torch.Tensor] = field(default_factory=dict)
    label_eligible: Dict[str, bool] = field(default_factory=dict)

    def validate(self, n_atoms: int) -> None:
        if not self.condition_uid:
            raise ValueError("condition_uid is required")
        if self.boundary_hash != canonical_boundary_hash(self.boundary_spec):
            raise ValueError("boundary_hash does not match boundary_spec")
        if (self.electron_count is None) != (self.electron_count_convention is None):
            raise ValueError("electron_count and its convention must be present together")
        for name, value in (("electron_count", self.electron_count), ("net_charge", self.net_charge)):
            if value is not None and (not isinstance(value, torch.Tensor) or value.numel() != 1):
                raise ValueError("%s must be a scalar tensor" % name)
            if value is not None and not torch.isfinite(value).all():
                raise ValueError("%s must be finite" % name)
        if not torch.isfinite(torch.tensor(float(self.weight))) or self.weight < 0:
            raise ValueError("condition weight must be finite and nonnegative")
        for name, value in self.inputs.items():
            if name not in {"local_external_field", "external_field", "control_value", "constraint_target"}:
                raise ValueError("unsupported condition input: %s" % name)
            if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("condition input %s must be a finite floating-point tensor" % name)
            if name == "local_external_field":
                if value.ndim != 2 or value.shape[0] != n_atoms:
                    raise ValueError("condition local_external_field must have shape [n_atoms, components]")
            elif value.ndim != 1:
                raise ValueError("condition input %s must be one-dimensional" % name)
        if any(not isinstance(key, str) or type(value) is not bool for key, value in self.label_eligible.items()):
            raise TypeError("label_eligible must map label names to booleans")
        missing_units = sorted(set(self.labels) - set(self.label_units))
        if missing_units:
            raise ValueError("labels are missing units: %s" % missing_units)
        for key, value in self.labels.items():
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError("label %s must be a finite tensor" % key)
            if key == "condition_forces" and tuple(value.shape) != (n_atoms, 3):
                raise ValueError("condition_forces must have shape [n_atoms, 3]")
            if key == "dipole" and self.net_charge is not None and float(self.net_charge) != 0:
                if "origin" not in self.label_metadata.get(key, {}):
                    raise ValueError("charged-system dipole labels require an origin")
            if key == "polarization" and "branch" not in self.label_metadata.get(key, {}):
                raise ValueError("periodic polarization labels require a branch/reference")
            if key in {"group_potential", "electrode_potential"}:
                potential_gauge = self.gauge.get("electrostatic_potential_gauge", {})
                if not potential_gauge.get("kind"):
                    raise ValueError("potential labels require an electrostatic potential gauge")
            if key in {"constraint_multiplier", "primal_residual"}:
                if self.label_metadata.get(key, {}).get("boundary_hash") != self.boundary_hash:
                    raise ValueError("%s must carry the matching boundary_hash" % key)
        for response in self.responses:
            response.validate()


@dataclass(frozen=True)
class VariationalData:
    schema_version: str
    geometry_uid: str
    geometry_hash: str
    variable_spec: Dict[str, Any]
    group_member_ptr: torch.Tensor
    group_atom_index: torch.Tensor
    group_names: Tuple[str, ...]
    group_roles: Tuple[str, ...]
    conditions: Tuple[VariationalCondition, ...]
    provenance: Dict[str, Any] = field(default_factory=dict)

    def validate(self, n_atoms: int) -> None:
        if not self.schema_version or not self.geometry_uid or not self.geometry_hash:
            raise ValueError("schema_version, geometry_uid, and geometry_hash are required")
        if self.group_member_ptr.dtype != torch.long or self.group_atom_index.dtype != torch.long:
            raise TypeError("group CSR tensors must use torch.long")
        if self.group_member_ptr.ndim != 1 or self.group_atom_index.ndim != 1:
            raise ValueError("group CSR tensors must be one-dimensional")
        n_groups = len(self.group_names)
        if len(self.group_roles) != n_groups or self.group_member_ptr.numel() != n_groups + 1:
            raise ValueError("group names, roles, and CSR pointers are inconsistent")
        if int(self.group_member_ptr[0]) != 0 or int(self.group_member_ptr[-1]) != self.group_atom_index.numel():
            raise ValueError("invalid group CSR endpoints")
        if torch.any(self.group_member_ptr[1:] < self.group_member_ptr[:-1]):
            raise ValueError("group CSR pointers must be nondecreasing")
        if self.group_atom_index.numel() and (
            int(self.group_atom_index.min()) < 0 or int(self.group_atom_index.max()) >= n_atoms
        ):
            raise ValueError("group atom index is out of range")
        if not self.conditions:
            raise ValueError("at least one variational condition is required")
        if len({item.condition_uid for item in self.conditions}) != len(self.conditions):
            raise ValueError("condition_uid values must be unique within a geometry")
        for condition in self.conditions:
            condition.validate(n_atoms)
            for name in ("local_external_field", "external_field"):
                value = condition.inputs.get(name)
                if value is not None and value.shape[-1] != int(self.variable_spec.get("components", 1)):
                    raise ValueError("condition %s component count differs from variable_spec" % name)
            for key in ("group_charge", "group_potential"):
                value = condition.labels.get(key)
                if value is not None and value.numel() != n_groups:
                    raise ValueError("%s must have one value per declared group" % key)


def geometry_hash(
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    for value in (atomic_numbers, positions, cell, pbc):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _optional_scalars(
    conditions: Sequence[VariationalCondition], name: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    present = torch.tensor([getattr(item, name) is not None for item in conditions], dtype=torch.bool)
    values = torch.zeros(len(conditions), dtype=torch.float64)
    for index, condition in enumerate(conditions):
        value = getattr(condition, name)
        if value is not None:
            values[index] = value.detach().to(dtype=torch.float64).reshape(())
    return values, present


def move_variational_data(
    payload: VariationalData,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    non_blocking: bool = False,
) -> VariationalData:
    def move(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        target_dtype = dtype if dtype is not None and torch.is_floating_point(value) else None
        return value.to(device=device, dtype=target_dtype, non_blocking=non_blocking)

    conditions = []
    for condition in payload.conditions:
        responses = tuple(
            VariationalResponse(
                input_ref=response.input_ref,
                output_ref=response.output_ref,
                value=move(response.value),
                unit=response.unit,
                method=response.method,
                gauge=response.gauge,
                provenance=response.provenance,
            )
            for response in condition.responses
        )
        conditions.append(
            VariationalCondition(
                condition_uid=condition.condition_uid,
                electron_count=move(condition.electron_count),
                electron_count_convention=condition.electron_count_convention,
                net_charge=move(condition.net_charge),
                boundary_spec=condition.boundary_spec,
                boundary_hash=condition.boundary_hash,
                gauge=condition.gauge,
                labels={key: move(value) for key, value in condition.labels.items()},
                label_units=condition.label_units,
                label_metadata=condition.label_metadata,
                responses=responses,
                weight=condition.weight,
                provenance=condition.provenance,
                inputs={key: move(value) for key, value in condition.inputs.items()},
                label_eligible=condition.label_eligible,
            )
        )
    return VariationalData(
        schema_version=payload.schema_version,
        geometry_uid=payload.geometry_uid,
        geometry_hash=payload.geometry_hash,
        variable_spec=payload.variable_spec,
        group_member_ptr=move(payload.group_member_ptr),
        group_atom_index=move(payload.group_atom_index),
        group_names=payload.group_names,
        group_roles=payload.group_roles,
        conditions=tuple(conditions),
        provenance=payload.provenance,
    )


def collate_variational_data(
    payloads: Sequence[VariationalData], n_atoms: Sequence[int]
) -> Dict[str, Any]:
    """Collate geometry-level payloads without duplicating coordinates per condition."""

    if len(payloads) != len(n_atoms) or not payloads:
        raise ValueError("payloads and n_atoms must have the same nonzero length")
    for payload, count in zip(payloads, n_atoms):
        payload.validate(int(count))

    condition_ptr = [0]
    geometry_group_ptr = [0]
    group_member_ptr = [0]
    group_atom_index = []
    conditions = []
    condition_atom_counts = []
    condition_state_counts = []
    condition_group_counts = []
    atom_offset = 0
    for payload, count in zip(payloads, n_atoms):
        conditions.extend(payload.conditions)
        components = int(payload.variable_spec.get("components", 1))
        if components <= 0:
            raise ValueError("variable components must be positive")
        condition_atom_counts.extend([int(count)] * len(payload.conditions))
        condition_state_counts.extend([int(count) * components] * len(payload.conditions))
        condition_group_counts.extend([len(payload.group_names)] * len(payload.conditions))
        condition_ptr.append(len(conditions))
        geometry_group_ptr.append(geometry_group_ptr[-1] + len(payload.group_names))
        for start, stop in zip(payload.group_member_ptr[:-1], payload.group_member_ptr[1:]):
            members = payload.group_atom_index[int(start) : int(stop)] + atom_offset
            group_atom_index.extend(members.tolist())
            group_member_ptr.append(len(group_atom_index))
        atom_offset += int(count)

    labels = {}
    label_keys = sorted({key for condition in conditions for key in (*condition.labels, *condition.label_eligible)})
    for key in label_keys:
        values = []
        ptr = [0]
        shapes = []
        present = []
        units = set()
        metadata = []
        for condition in conditions:
            value = condition.labels.get(key)
            present.append(value is not None)
            if value is not None:
                flat = value.reshape(-1)
                values.append(flat)
                ptr.append(ptr[-1] + flat.numel())
                shapes.append(tuple(value.shape))
                units.add(condition.label_units[key])
                metadata.append(condition.label_metadata.get(key, {}))
            else:
                ptr.append(ptr[-1])
                shapes.append(None)
                metadata.append({})
        if len(units) > 1:
            raise ValueError("label %s has incompatible units within one batch" % key)
        reference = next((value for value in values), torch.tensor([], dtype=torch.get_default_dtype()))
        labels[key] = {
            "values": torch.cat(values) if values else reference,
            "ptr": torch.tensor(ptr, dtype=torch.long),
            "shape": tuple(shapes),
            "present": torch.tensor(present, dtype=torch.bool),
            "eligible": torch.tensor([item.label_eligible.get(key, True) for item in conditions], dtype=torch.bool),
            "unit": next(iter(units), None),
            "metadata": tuple(metadata),
        }

    condition_inputs = {}
    for name in sorted({key for condition in conditions for key in condition.inputs}):
        values, ptr, shapes, present = [], [0], [], []
        for condition in conditions:
            value = condition.inputs.get(name)
            present.append(value is not None)
            shapes.append(None if value is None else tuple(value.shape))
            if value is not None:
                values.append(value.reshape(-1))
            ptr.append(ptr[-1] + (0 if value is None else value.numel()))
        condition_inputs[name] = {
            "values": torch.cat(values), "ptr": torch.tensor(ptr, dtype=torch.long),
            "shape": tuple(shapes), "present": torch.tensor(present, dtype=torch.bool),
        }

    responses = [response for condition in conditions for response in condition.responses]
    condition_response_ptr = [0]
    for condition in conditions:
        condition_response_ptr.append(condition_response_ptr[-1] + len(condition.responses))
    response_value_ptr = [0]
    for response in responses:
        response_value_ptr.append(response_value_ptr[-1] + response.value.numel())

    electron_count, electron_count_present = _optional_scalars(conditions, "electron_count")
    net_charge, net_charge_present = _optional_scalars(conditions, "net_charge")
    condition_atom_ptr = torch.nn.functional.pad(
        torch.cumsum(torch.tensor(condition_atom_counts, dtype=torch.long), dim=0), (1, 0)
    )
    condition_state_ptr = torch.nn.functional.pad(
        torch.cumsum(torch.tensor(condition_state_counts, dtype=torch.long), dim=0), (1, 0)
    )
    condition_group_ptr = torch.nn.functional.pad(
        torch.cumsum(torch.tensor(condition_group_counts, dtype=torch.long), dim=0), (1, 0)
    )
    return {
        "schema_version": tuple(payload.schema_version for payload in payloads),
        "geometry_uid": tuple(payload.geometry_uid for payload in payloads),
        "geometry_hash": tuple(payload.geometry_hash for payload in payloads),
        "variable_spec": tuple(payload.variable_spec for payload in payloads),
        "condition_ptr": torch.tensor(condition_ptr, dtype=torch.long),
        "condition_geometry_index": torch.repeat_interleave(
            torch.arange(len(payloads)), torch.diff(torch.tensor(condition_ptr))
        ),
        "condition_atom_ptr": condition_atom_ptr,
        "condition_state_ptr": condition_state_ptr,
        "condition_group_ptr": condition_group_ptr,
        "condition_uid": tuple(item.condition_uid for item in conditions),
        "condition_weight": torch.tensor([item.weight for item in conditions]),
        "condition_inputs": condition_inputs,
        "electron_count": electron_count,
        "electron_count_present": electron_count_present,
        "electron_count_convention": tuple(item.electron_count_convention for item in conditions),
        "net_charge": net_charge,
        "net_charge_present": net_charge_present,
        "boundary_spec": tuple(item.boundary_spec for item in conditions),
        "boundary_hash": tuple(item.boundary_hash for item in conditions),
        "gauge": tuple(item.gauge for item in conditions),
        "condition_provenance": tuple(item.provenance for item in conditions),
        "geometry_group_ptr": torch.tensor(geometry_group_ptr, dtype=torch.long),
        "group_member_ptr": torch.tensor(group_member_ptr, dtype=torch.long),
        "group_atom_index": torch.tensor(group_atom_index, dtype=torch.long),
        "group_name": tuple(name for payload in payloads for name in payload.group_names),
        "group_role": tuple(role for payload in payloads for role in payload.group_roles),
        "labels": labels,
        "multiplier_ptr": labels.get("constraint_multiplier", {}).get(
            "ptr", torch.zeros(len(conditions) + 1, dtype=torch.long)
        ),
        "condition_response_ptr": torch.tensor(condition_response_ptr, dtype=torch.long),
        "response_value_ptr": torch.tensor(response_value_ptr, dtype=torch.long),
        "response_shape": torch.tensor(
            [response.value.shape for response in responses], dtype=torch.long
        ).reshape(-1, 2),
        "response_values": torch.cat([item.value.reshape(-1) for item in responses])
        if responses
        else torch.tensor([], dtype=torch.get_default_dtype()),
        "response_input_ref": tuple(item.input_ref for item in responses),
        "response_output_ref": tuple(item.output_ref for item in responses),
        "response_unit": tuple(item.unit for item in responses),
        "response_method": tuple(item.method for item in responses),
        "response_gauge": tuple(item.gauge for item in responses),
        "response_provenance": tuple(item.provenance for item in responses),
    }


def condition_input(batch: Dict[str, Any], name: str, index: int) -> Optional[torch.Tensor]:
    """Return a live condition input view, or None for the legacy fallback.

    Call before building the energy and reuse this view for differentiation;
    making a new view afterwards does not recover its upstream autograd node.
    """
    entry = batch.get("condition_inputs", {}).get(name)
    if entry is None or not bool(entry["present"][index]):
        return None
    start, stop = int(entry["ptr"][index]), int(entry["ptr"][index + 1])
    return entry["values"][start:stop].reshape(entry["shape"][index])
