"""Training losses for backend-independent amortized variational proposals."""

import math
from typing import Dict, Optional

import torch
from torch import nn

from curator.data import properties
from curator.variational.amortized_training import AmortizedTrainingBatch
from .structured_output import StructuredModelOutput


class AmortizedVariationalLoss(nn.Module):
    """Train a projected proposal against an exact teacher and KKT residual.

    The loss consumes an :class:`AmortizedTrainingBatch`; it never constructs a
    physical Hessian and therefore has no dependency on Ewald or any other
    plugin backend.
    """

    def __init__(
        self,
        *,
        state_weight: float = 1.0,
        kkt_weight: float = 1.0,
        energy_gap_weight: float = 0.0,
        multiplier_weight: float = 0.0,
        state_scale: float = 1.0,
        energy_scale: float = 1.0,
        multiplier_scale: float = 1.0,
        balance_mode: str = "geometry",
    ) -> None:
        super().__init__()
        self.weights = {
            "proposal_state": self._nonnegative(state_weight, "state_weight"),
            "proposal_kkt": self._nonnegative(kkt_weight, "kkt_weight"),
            "proposal_energy_gap": self._nonnegative(
                energy_gap_weight, "energy_gap_weight"
            ),
            "proposal_multiplier": self._nonnegative(
                multiplier_weight, "multiplier_weight"
            ),
        }
        self.state_scale = self._positive(state_scale, "state_scale")
        self.energy_scale = self._positive(energy_scale, "energy_scale")
        self.multiplier_scale = self._positive(
            multiplier_scale, "multiplier_scale"
        )
        if balance_mode not in {"geometry", "condition"}:
            raise ValueError("balance_mode must be 'geometry' or 'condition'")
        self.balance_mode = balance_mode

    @staticmethod
    def _nonnegative(value, name):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("%s must be finite and nonnegative" % name)
        return value

    @staticmethod
    def _positive(value, name):
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("%s must be finite and positive" % name)
        return value

    @staticmethod
    def _axes(training, condition_geometry_index, condition_weight, geometry_weight):
        n_conditions = training.n_conditions
        device = training.projected_state.device
        dtype = training.projected_state.dtype
        if condition_geometry_index is None:
            condition_geometry_index = torch.arange(
                n_conditions, dtype=torch.long, device=device
            )
        else:
            condition_geometry_index = condition_geometry_index.to(device=device)
        if condition_geometry_index.dtype != torch.long or condition_geometry_index.shape != (n_conditions,):
            raise TypeError("condition_geometry_index must be torch.long with one entry per condition")
        if condition_geometry_index.numel() and int(condition_geometry_index.min()) < 0:
            raise ValueError("condition_geometry_index cannot be negative")
        n_geometries = (
            int(condition_geometry_index.max().item()) + 1
            if condition_geometry_index.numel()
            else 0
        )
        if condition_weight is None:
            condition_weight = torch.ones(n_conditions, dtype=dtype, device=device)
        else:
            condition_weight = condition_weight.to(device=device, dtype=dtype)
        if geometry_weight is None:
            geometry_weight = torch.ones(n_geometries, dtype=dtype, device=device)
        else:
            geometry_weight = geometry_weight.to(device=device, dtype=dtype)
        if condition_weight.shape != (n_conditions,) or not torch.isfinite(condition_weight).all() or torch.any(condition_weight < 0):
            raise ValueError("condition_weight must be a finite nonnegative condition vector")
        if geometry_weight.shape != (n_geometries,) or not torch.isfinite(geometry_weight).all() or torch.any(geometry_weight < 0):
            raise ValueError("geometry_weight must be a finite nonnegative geometry vector")
        return condition_geometry_index, condition_weight, geometry_weight

    def _reduce(self, values, active, axes):
        geometry_index, condition_weight, geometry_weight = axes
        active = active & (condition_weight > 0) & (geometry_weight[geometry_index] > 0)
        zero = values.sum() * 0.0
        if not bool(active.any().item()):
            return zero, zero, zero, 0, 0
        if self.balance_mode == "condition":
            weights = condition_weight * geometry_weight[geometry_index] * active.to(values.dtype)
            numerator = (values * weights).sum()
            denominator = weights.sum()
            n_geometries = torch.unique(geometry_index[active]).numel()
        else:
            numerator = zero
            denominator = zero
            n_geometries = 0
            for geometry in range(geometry_weight.numel()):
                selected = active & (geometry_index == geometry)
                inner_weights = condition_weight * selected.to(values.dtype)
                inner_denominator = inner_weights.sum()
                if float(inner_denominator.detach().cpu()) <= 0:
                    continue
                numerator = numerator + geometry_weight[geometry] * (
                    values * inner_weights
                ).sum() / inner_denominator
                denominator = denominator + geometry_weight[geometry]
                n_geometries += 1
        return (
            numerator / denominator,
            numerator,
            denominator,
            int(active.sum().item()),
            int(n_geometries),
        )

    @staticmethod
    def _term(coefficient, reduced, n_conditions):
        value, numerator, denominator, used_conditions, used_geometries = reduced
        return {
            "coefficient": coefficient,
            "loss": coefficient * value,
            "unweighted": value,
            "numerator": numerator,
            "denominator": denominator,
            "n_conditions": used_conditions,
            "n_geometries": used_geometries,
            "coverage": used_conditions / max(1, n_conditions),
            "status": "active" if used_conditions and coefficient else "disabled",
        }

    @staticmethod
    def _ragged_mean_square(values, ptr, n_conditions):
        errors = values.new_zeros(n_conditions)
        active = torch.zeros(n_conditions, dtype=torch.bool, device=values.device)
        for index in range(n_conditions):
            segment = values[ptr[index] : ptr[index + 1]]
            if segment.numel():
                errors[index] = segment.square().mean()
                active[index] = True
        return errors, active

    def forward(
        self,
        training: AmortizedTrainingBatch,
        *,
        condition_geometry_index: Optional[torch.Tensor] = None,
        condition_weight: Optional[torch.Tensor] = None,
        geometry_weight: Optional[torch.Tensor] = None,
    ) -> Dict:
        if not isinstance(training, AmortizedTrainingBatch):
            raise TypeError("training must be an AmortizedTrainingBatch")
        axes = self._axes(
            training,
            condition_geometry_index,
            condition_weight,
            geometry_weight,
        )
        n_conditions = training.n_conditions
        errors = {}
        errors["proposal_state"] = self._ragged_mean_square(
            (training.projected_state - training.teacher_state) / self.state_scale,
            training.state_ptr,
            n_conditions,
        )
        errors["proposal_kkt"] = self._ragged_mean_square(
            training.normalized_projected_kkt,
            training.kkt_ptr,
            n_conditions,
        )
        errors["proposal_energy_gap"] = (
            (training.energy_gap / self.energy_scale).square(),
            torch.ones(n_conditions, dtype=torch.bool, device=training.projected_state.device),
        )
        errors["proposal_multiplier"] = self._ragged_mean_square(
            (training.projected_multipliers - training.teacher_multipliers)
            / self.multiplier_scale,
            training.multiplier_ptr,
            n_conditions,
        )

        total = training.projected_state.sum() * 0.0
        terms = {}
        for name, coefficient in self.weights.items():
            values, active = errors[name]
            reduced = self._reduce(values, active, axes)
            terms[name] = self._term(coefficient, reduced, n_conditions)
            total = total + terms[name]["loss"]
        return {"loss": total, "terms": terms, "balance_mode": self.balance_mode}


class AmortizedTrainingModelOutput(StructuredModelOutput):
    """Adapter for Curator trainers using ``prediction['amortized_training']``."""

    def __init__(
        self,
        loss_fn: AmortizedVariationalLoss,
        name: str = "amortized_variational",
        *,
        loss_weight: float = 1.0,
    ) -> None:
        if not isinstance(loss_fn, AmortizedVariationalLoss):
            raise TypeError("loss_fn must be an AmortizedVariationalLoss")
        super().__init__(loss_fn, name, loss_fn.weights, loss_weight)

    def calculate_loss(self, prediction, batch=None, return_num_obs=True, *, subset=None):
        if "amortized_training" not in prediction:
            raise KeyError("model output is missing amortized_training")
        if batch is None:
            raise ValueError("amortized training requires variational condition axes")
        training = prediction["amortized_training"]
        if not isinstance(training, AmortizedTrainingBatch):
            raise TypeError("training must be an AmortizedTrainingBatch")
        if "variational" in batch:
            axes = batch["variational"]
        else:
            n_atoms = batch.get(properties.n_atoms)
            if not (
                isinstance(n_atoms, torch.Tensor)
                and n_atoms.ndim == 1
                and n_atoms.numel() == training.n_conditions
            ):
                raise ValueError(
                    "amortized training requires variational condition axes "
                    "unless there is exactly one condition per geometry"
                )
            axes = {
                "condition_geometry_index": torch.arange(
                    training.n_conditions, device=training.projected_state.device
                ),
            }
        result = self.loss_fn(
            training,
            condition_geometry_index=axes["condition_geometry_index"],
            condition_weight=axes.get("condition_weight"),
            geometry_weight=batch.get("weight"),
        )
        return self._record_result(result, return_num_obs, subset)


__all__ = ["AmortizedTrainingModelOutput", "AmortizedVariationalLoss"]
