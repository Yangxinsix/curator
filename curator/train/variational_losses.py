"""Condition-aware losses for geometry-grouped ACVR training batches."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from .structured_output import StructuredModelOutput


def _validated_ptr(ptr, length, value_count, name):
    if not isinstance(ptr, torch.Tensor) or ptr.dtype != torch.long or ptr.ndim != 1:
        raise TypeError("%s ptr must be a one-dimensional torch.long tensor" % name)
    values = ptr.detach().cpu().tolist()
    if len(values) != length + 1 or values[0] != 0 or values[-1] != value_count:
        raise ValueError("%s ptr endpoints or length are inconsistent" % name)
    if any(right < left for left, right in zip(values, values[1:])):
        raise ValueError("%s ptr must be nondecreasing" % name)
    return values


def _reference_tensor(prediction, target):
    stack = [prediction]
    while stack:
        value = stack.pop()
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            return value
        if isinstance(value, Mapping):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    return target["condition_weight"]


def _mask(value, length, name, device, default=True):
    if value is None:
        return torch.full((length,), default, dtype=torch.bool, device=device)
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or value.shape != (length,):
        raise TypeError("%s must be a torch.bool tensor with one value per condition" % name)
    return value.to(device=device)


def _shape(value):
    if value is None:
        return None
    return tuple(int(item) for item in value)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class VariationalJointLoss(nn.Module):
    """Combine supervised observables and variational certificates.

    Predictions must live under ``prediction["variational_prediction"]``.
    This deliberately differs from the target namespace because Curator's legacy
    training loop copies missing batch keys into the model-output dictionary.

    Every positive supervised coefficient requires a physical scale.  Ragged
    scalar errors are first averaged within each condition.  By default,
    conditions are then averaged within each geometry, followed by a weighted
    average over geometries, so a geometry carrying more voltage/charge states
    does not automatically dominate training.

    KKT certificates must be assembled from the learned functional, predicted
    state/multipliers, and compiled boundary.  A freely learned "residual head"
    is not a valid input to this loss.
    """

    def __init__(
        self,
        supervised_weights: Mapping[str, float],
        supervised_scales: Mapping[str, float],
        required_labels: Iterable[str] = (),
        required_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
        certificate_weights: Optional[Mapping[str, float]] = None,
        response_weight: float = 0.0,
        response_scales: Optional[Mapping[str, float]] = None,
        stability_weight: float = 0.0,
        stability_margin: float = 0.0,
        balance_mode: str = "geometry",
    ) -> None:
        super().__init__()
        self.supervised_weights = self._weights(supervised_weights, "supervised")
        self.supervised_scales = {
            str(key): self._positive(value, "supervised scale")
            for key, value in supervised_scales.items()
        }
        missing_scales = {
            key for key, weight in self.supervised_weights.items()
            if weight > 0 and key not in self.supervised_scales
        }
        if missing_scales:
            raise ValueError("positive supervised terms require scales: %s" % sorted(missing_scales))
        self.required_labels = frozenset(str(key) for key in required_labels)
        unknown_required = self.required_labels - set(self.supervised_weights)
        if unknown_required:
            raise ValueError("required labels need supervised weights: %s" % sorted(unknown_required))
        zero_required = {key for key in self.required_labels if self.supervised_weights[key] <= 0}
        if zero_required:
            raise ValueError("required labels need positive supervised weights: %s" % sorted(zero_required))
        self.required_metadata = {
            str(key): dict(value) for key, value in (required_metadata or {}).items()
        }
        unknown_metadata = set(self.required_metadata) - set(self.supervised_weights)
        if unknown_metadata:
            raise ValueError("metadata requirements need supervised weights: %s" % sorted(unknown_metadata))
        self.certificate_weights = self._weights(certificate_weights or {}, "certificate")
        self.response_weight = self._nonnegative(response_weight, "response_weight")
        self.response_scales = {
            str(key): self._positive(value, "response scale")
            for key, value in (response_scales or {}).items()
        }
        self.stability_weight = self._nonnegative(stability_weight, "stability_weight")
        self.stability_margin = float(stability_margin)
        if not torch.isfinite(torch.tensor(self.stability_margin)):
            raise ValueError("stability_margin must be finite")
        if balance_mode not in {"geometry", "condition"}:
            raise ValueError("balance_mode must be 'geometry' or 'condition'")
        self.balance_mode = balance_mode

    @staticmethod
    def _nonnegative(value, name):
        value = float(value)
        if not torch.isfinite(torch.tensor(value)) or value < 0:
            raise ValueError("%s must be finite and nonnegative" % name)
        return value

    @staticmethod
    def _positive(value, name):
        value = float(value)
        if not torch.isfinite(torch.tensor(value)) or value <= 0:
            raise ValueError("%s must be finite and positive" % name)
        return value

    @classmethod
    def _weights(cls, values, name):
        return {
            str(key): cls._nonnegative(value, "%s weight" % name)
            for key, value in values.items()
        }

    @staticmethod
    def _batch_axes(batch, target, reference):
        condition_weight = target["condition_weight"].to(
            device=reference.device, dtype=reference.dtype
        )
        if condition_weight.ndim != 1 or not torch.isfinite(condition_weight).all() or torch.any(condition_weight < 0):
            raise ValueError("condition_weight must be a finite nonnegative vector")
        geometry_index = target["condition_geometry_index"].to(device=reference.device)
        if geometry_index.dtype != torch.long or geometry_index.shape != condition_weight.shape:
            raise TypeError("condition_geometry_index must be torch.long with one value per condition")
        if "weight" in batch:
            geometry_weight = batch["weight"].to(device=reference.device, dtype=reference.dtype)
        else:
            condition_ptr = target["condition_ptr"]
            geometry_weight = torch.ones(
                condition_ptr.numel() - 1, device=reference.device, dtype=reference.dtype
            )
        if geometry_weight.ndim != 1 or not torch.isfinite(geometry_weight).all() or torch.any(geometry_weight < 0):
            raise ValueError("geometry weights must be a finite nonnegative vector")
        if geometry_index.numel() and (
            int(geometry_index.min()) < 0 or int(geometry_index.max()) >= geometry_weight.numel()
        ):
            raise ValueError("condition_geometry_index is out of range")
        return condition_weight, geometry_index, geometry_weight

    def _reduce(self, values, active, axes):
        condition_weight, geometry_index, geometry_weight = axes
        if values.shape != condition_weight.shape or active.shape != condition_weight.shape:
            raise ValueError("condition losses and masks must have one value per condition")
        active = active & (condition_weight > 0) & (geometry_weight[geometry_index] > 0)
        zero = values.sum() * 0
        if self.balance_mode == "condition":
            weight = condition_weight * geometry_weight[geometry_index] * active.to(values.dtype)
            denominator = weight.sum()
            numerator = (values * weight).sum()
            used_geometries = torch.unique(geometry_index[active]).numel()
        else:
            numerator = zero
            denominator = zero
            used_geometries = 0
            for index in range(geometry_weight.numel()):
                selected = active & (geometry_index == index)
                inner_weight = condition_weight * selected.to(values.dtype)
                inner_denominator = inner_weight.sum()
                if float(inner_denominator.detach().cpu()) <= 0:
                    continue
                geometry_loss = (values * inner_weight).sum() / inner_denominator
                numerator = numerator + geometry_weight[index] * geometry_loss
                denominator = denominator + geometry_weight[index]
                used_geometries += 1
        used_conditions = int(active.sum().detach().cpu())
        if float(denominator.detach().cpu()) <= 0:
            return zero, zero, zero, 0, 0
        return numerator / denominator, numerator, denominator, used_conditions, int(used_geometries)

    @staticmethod
    def _term(coefficient, reduced, n_conditions):
        loss, numerator, denominator, used_conditions, used_geometries = reduced
        return {
            "coefficient": coefficient,
            "loss": coefficient * loss,
            "unweighted": loss,
            "numerator": numerator,
            "denominator": denominator,
            "n_conditions": used_conditions,
            "n_geometries": used_geometries,
            "coverage": used_conditions / max(1, n_conditions),
            "status": "active" if used_conditions else "inactive",
        }

    def _label_errors(self, name, predicted, target, reference):
        n_conditions = target["present"].numel()
        pred_values = predicted["values"].reshape(-1)
        target_values = target["values"].to(
            device=pred_values.device, dtype=pred_values.dtype
        ).reshape(-1)
        pred_ptr = _validated_ptr(predicted["ptr"], n_conditions, pred_values.numel(), name + " prediction")
        target_ptr = _validated_ptr(target["ptr"], n_conditions, target_values.numel(), name + " target")
        if predicted.get("unit") != target.get("unit"):
            raise ValueError("%s prediction and target units do not match" % name)
        pred_shapes = predicted.get("shape")
        target_shapes = target.get("shape")
        if not isinstance(pred_shapes, Sequence) or len(pred_shapes) != n_conditions:
            raise ValueError("%s prediction shape metadata must cover every condition" % name)
        if not isinstance(target_shapes, Sequence) or len(target_shapes) != n_conditions:
            raise ValueError("%s target shape metadata must cover every condition" % name)
        present = _mask(target["present"], n_conditions, name + " present", pred_values.device)
        eligible = _mask(target.get("eligible"), n_conditions, name + " eligible", pred_values.device)
        if name in self.required_labels and torch.any(eligible & ~present):
            raise ValueError("required label %s is absent for an eligible condition" % name)
        active = present & eligible
        if name in self.required_labels and not torch.any(active):
            raise ValueError("required label %s has no eligible target" % name)
        metadata = target.get("metadata")
        if not isinstance(metadata, Sequence) or len(metadata) != n_conditions:
            raise ValueError("%s target metadata must cover every condition" % name)
        requirements = self.required_metadata.get(name, {})
        errors = reference.new_zeros(n_conditions)
        scale = self.supervised_scales[name]
        for index in range(n_conditions):
            pred_slice = pred_values[pred_ptr[index] : pred_ptr[index + 1]]
            target_slice = target_values[target_ptr[index] : target_ptr[index + 1]]
            pred_shape = _shape(pred_shapes[index])
            target_shape = _shape(target_shapes[index])
            if pred_shape is not None:
                expected_count = 1
                for dimension in pred_shape:
                    expected_count *= dimension
                if pred_slice.numel() != expected_count:
                    raise ValueError("%s prediction shape metadata is inconsistent at condition %d" % (name, index))
            if not active[index]:
                continue
            if pred_shape != target_shape or pred_slice.numel() != target_slice.numel() or not pred_slice.numel():
                raise ValueError("%s prediction shape does not match condition %d" % (name, index))
            missing = {
                key: expected for key, expected in requirements.items()
                if metadata[index].get(key) != expected
            }
            if missing:
                raise ValueError("%s metadata requirement failed at condition %d: %s" % (name, index, missing))
            errors[index] = ((pred_slice - target_slice) / scale).square().mean()
        return errors, active

    @staticmethod
    def _compiled_eligible(entry, target, n_conditions, device, name):
        eligible = _mask(entry.get("eligible"), n_conditions, name + " eligible", device)
        boundaries = target.get("boundary_spec")
        if not isinstance(boundaries, Sequence) or len(boundaries) != n_conditions:
            raise ValueError("boundary_spec must cover every condition")
        for index, boundary in enumerate(boundaries):
            status = boundary.get("compile_status")
            if bool(eligible[index]) and status not in {None, "compiled"}:
                raise ValueError("%s is not eligible for boundary compile_status=%s" % (name, status))
        return eligible

    def _certificate_errors(self, name, entry, target, reference):
        if entry.get("source") != "compiled_variational_operator":
            raise ValueError("%s certificate must come from the compiled variational operator" % name)
        if not entry.get("normalization"):
            raise ValueError("%s certificate requires explicit normalization metadata" % name)
        n_conditions = target["condition_weight"].numel()
        values = entry["values"].reshape(-1)
        ptr = _validated_ptr(entry["ptr"], n_conditions, values.numel(), name)
        eligible = self._compiled_eligible(entry, target, n_conditions, values.device, name)
        errors = reference.new_zeros(n_conditions)
        for index in range(n_conditions):
            if not eligible[index]:
                continue
            segment = values[ptr[index] : ptr[index + 1]]
            if not segment.numel():
                raise ValueError("eligible %s certificate is empty at condition %d" % (name, index))
            errors[index] = segment.square().mean()
        return errors, eligible

    @staticmethod
    def _response_key(condition, input_ref, output_ref, shape, unit, gauge, boundary_hash):
        return (
            int(condition), str(input_ref), str(output_ref), tuple(int(item) for item in shape),
            str(unit), _canonical(gauge), str(boundary_hash),
        )

    def _response_errors(self, predicted, target, reference):
        n_conditions = target["condition_weight"].numel()
        target_condition_ptr = _validated_ptr(
            target["condition_response_ptr"], n_conditions,
            len(target["response_unit"]), "condition response target",
        )
        target_values = target["response_values"]
        target_block_count = len(target["response_unit"])
        target_fields = (
            "response_input_ref", "response_output_ref", "response_shape",
            "response_unit", "response_gauge",
        )
        for field in target_fields:
            if len(target[field]) != target_block_count:
                raise ValueError("response target %s must cover every block" % field)
        target_value_ptr = _validated_ptr(
            target["response_value_ptr"], target_block_count,
            target_values.numel(), "response target",
        )
        values = predicted["values"].reshape(-1)
        block_count = len(predicted["unit"])
        value_ptr = _validated_ptr(predicted["ptr"], block_count, values.numel(), "response prediction")
        fields = ("condition_index", "input_ref", "output_ref", "shape", "unit", "gauge", "boundary_hash")
        for field in fields:
            if field not in predicted or len(predicted[field]) != block_count:
                raise ValueError("response prediction %s must cover every block" % field)
        predicted_blocks = {}
        for index in range(block_count):
            key = self._response_key(
                predicted["condition_index"][index], predicted["input_ref"][index],
                predicted["output_ref"][index], predicted["shape"][index],
                predicted["unit"][index], predicted["gauge"][index],
                predicted["boundary_hash"][index],
            )
            if key in predicted_blocks:
                raise ValueError("duplicate response prediction identity")
            predicted_blocks[key] = values[value_ptr[index] : value_ptr[index + 1]]
        condition_blocks = [[] for _ in range(n_conditions)]
        boundary_hash = target["boundary_hash"]
        for condition in range(n_conditions):
            for block in range(target_condition_ptr[condition], target_condition_ptr[condition + 1]):
                shape = tuple(int(item) for item in target["response_shape"][block])
                key = self._response_key(
                    condition, target["response_input_ref"][block],
                    target["response_output_ref"][block], shape,
                    target["response_unit"][block], target["response_gauge"][block],
                    boundary_hash[condition],
                )
                if key not in predicted_blocks:
                    raise KeyError("prediction is missing response block %s" % (key,))
                predicted_block = predicted_blocks[key]
                expected = target_values[
                    target_value_ptr[block] : target_value_ptr[block + 1]
                ].to(device=values.device, dtype=values.dtype)
                if predicted_block.numel() != expected.numel() or predicted_block.numel() != shape[0] * shape[1]:
                    raise ValueError("response prediction shape does not match block %d" % block)
                unit = target["response_unit"][block]
                if unit not in self.response_scales:
                    raise ValueError("response scale is required for unit %s" % unit)
                condition_blocks[condition].append(
                    ((predicted_block - expected) / self.response_scales[unit]).square().mean()
                )
        errors = reference.new_zeros(n_conditions)
        active = torch.zeros(n_conditions, dtype=torch.bool, device=reference.device)
        for condition, blocks in enumerate(condition_blocks):
            if blocks:
                errors[condition] = torch.stack(blocks).mean()
                active[condition] = True
        return errors, active

    def _stability_errors(self, entry, target, reference):
        if entry.get("source") != "reduced_lagrangian_hessian":
            raise ValueError("stability must use the reduced Lagrangian Hessian")
        if not entry.get("normalization"):
            raise ValueError("stability requires explicit metric/units normalization")
        n_conditions = target["condition_weight"].numel()
        curvature = entry["minimum_curvature"]
        if curvature.shape != (n_conditions,):
            raise ValueError("minimum_curvature must have one value per condition")
        eligible = self._compiled_eligible(entry, target, n_conditions, curvature.device, "stability")
        errors = torch.relu(self.stability_margin - curvature).square()
        return errors.to(device=reference.device, dtype=reference.dtype), eligible

    def forward(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        if "variational" not in batch:
            raise KeyError("batch is missing variational targets")
        if "variational_prediction" not in prediction:
            raise KeyError("model output is missing the independent variational_prediction namespace")
        target = batch["variational"]
        predicted = prediction["variational_prediction"]
        if predicted.get("value_space") != "physical":
            raise ValueError("variational predictions must declare value_space='physical'")
        reference = _reference_tensor(predicted, target)
        axes = self._batch_axes(batch, target, reference)
        n_conditions = axes[0].numel()
        zero = reference.sum() * 0
        total = zero
        terms = {}

        predicted_labels = predicted.get("labels", {})
        for name, coefficient in self.supervised_weights.items():
            if coefficient == 0:
                terms[name] = self._term(0.0, (zero, zero, zero, 0, 0), n_conditions)
                terms[name]["status"] = "disabled"
                continue
            target_entry = target.get("labels", {}).get(name)
            if target_entry is None:
                if name in self.required_labels:
                    raise KeyError("required variational label %s is missing" % name)
                reduced = (zero, zero, zero, 0, 0)
            else:
                if name not in predicted_labels:
                    raise KeyError("prediction is missing variational label %s" % name)
                errors, active = self._label_errors(
                    name, predicted_labels[name], target_entry, reference
                )
                reduced = self._reduce(errors, active, axes)
            terms[name] = self._term(coefficient, reduced, n_conditions)
            total = total + terms[name]["loss"]

        certificates = predicted.get("certificates", {})
        for name, coefficient in self.certificate_weights.items():
            if coefficient == 0:
                key = "certificate:" + name
                terms[key] = self._term(0.0, (zero, zero, zero, 0, 0), n_conditions)
                terms[key]["status"] = "disabled"
                continue
            if name not in certificates:
                raise KeyError("prediction is missing variational certificate %s" % name)
            errors, active = self._certificate_errors(
                name, certificates[name], target, reference
            )
            key = "certificate:" + name
            terms[key] = self._term(coefficient, self._reduce(errors, active, axes), n_conditions)
            total = total + terms[key]["loss"]

        if self.response_weight:
            if "responses" not in predicted:
                raise KeyError("prediction is missing variational responses")
            errors, active = self._response_errors(predicted["responses"], target, reference)
            terms["response"] = self._term(
                self.response_weight, self._reduce(errors, active, axes), n_conditions
            )
            total = total + terms["response"]["loss"]

        if self.stability_weight:
            if "stability" not in predicted:
                raise KeyError("prediction is missing stability certificate")
            errors, active = self._stability_errors(predicted["stability"], target, reference)
            terms["stability"] = self._term(
                self.stability_weight, self._reduce(errors, active, axes), n_conditions
            )
            total = total + terms["stability"]["loss"]

        return {"loss": total, "terms": terms, "balance_mode": self.balance_mode}

    @property
    def term_coefficients(self) -> Dict[str, float]:
        values = dict(self.supervised_weights)
        values.update({"certificate:" + key: value for key, value in self.certificate_weights.items()})
        if self.response_weight:
            values["response"] = self.response_weight
        if self.stability_weight:
            values["stability"] = self.stability_weight
        return values


class VariationalModelOutput(StructuredModelOutput):
    """Adapter that makes :class:`VariationalJointLoss` usable by Curator's trainer."""

    def __init__(self, loss_fn: VariationalJointLoss, name: str = "variational", *, loss_weight: float = 1.0) -> None:
        if not isinstance(loss_fn, VariationalJointLoss):
            raise TypeError("loss_fn must be a VariationalJointLoss")
        super().__init__(loss_fn, name, loss_fn.term_coefficients, loss_weight)

    def calculate_loss(self, prediction, batch=None, return_num_obs=True, *, subset=None):
        if batch is None:
            raise ValueError("VariationalModelOutput requires the complete batch")
        result = self.loss_fn(prediction, batch)
        return self._record_result(result, return_num_obs, subset)
