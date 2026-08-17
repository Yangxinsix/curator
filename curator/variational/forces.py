"""Explicit, full-path, and adjoint-corrected variational derivatives."""

from dataclasses import dataclass
import math
from typing import Dict, Literal, Mapping, Optional

import torch

from .validation import SingularKKTError


ForceMode = Literal["envelope", "full_differentiable", "adjoint_corrected"]


@dataclass(frozen=True)
class VariationalDerivative:
    mode: str
    gradients: Dict[str, torch.Tensor]
    explicit_gradients: Dict[str, torch.Tensor]
    response_correction: Dict[str, torch.Tensor]
    forces: Optional[torch.Tensor]
    stationarity_residual: torch.Tensor
    constraint_residual: torch.Tensor
    adjoint_state: Optional[torch.Tensor]
    adjoint_multiplier: Optional[torch.Tensor]
    kkt_certified: bool
    conservative_for_reported_energy: bool


def _coordinate_gradients(
    scalar: torch.Tensor,
    coordinates: Mapping[str, torch.Tensor],
    *,
    create_graph: bool,
    retain_graph: bool,
) -> Dict[str, torch.Tensor]:
    names = tuple(coordinates)
    values = tuple(coordinates.values())
    if not scalar.requires_grad:
        return {name: torch.zeros_like(value) for name, value in coordinates.items()}
    gradients = torch.autograd.grad(
        scalar,
        values,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return {
        name: torch.zeros_like(value) if gradient is None else gradient
        for name, value, gradient in zip(names, values, gradients)
    }


def _subtract(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {name: left[name] - right[name] for name in left}


def differentiate_variational_energy(
    *,
    objective_fn,
    constraint_fn,
    coordinates: Mapping[str, torch.Tensor],
    state: torch.Tensor,
    multipliers: torch.Tensor,
    mode: ForceMode,
    residual_tolerance: float = 1.0e-8,
    feasibility_tolerance: float = 1.0e-10,
    create_graph: bool = False,
) -> VariationalDerivative:
    r"""Differentiate a constrained variational energy in one explicit mode.

    The frozen sign convention is ``L = objective + lambda.T @ constraint``.
    ``envelope`` holds state and multiplier fixed, ``full_differentiable``
    follows the supplied state's computation graph, and ``adjoint_corrected``
    applies a first-order residual correction toward the relaxed derivative.
    """

    if mode not in ("envelope", "full_differentiable", "adjoint_corrected"):
        raise ValueError("unsupported variational force mode: %s" % mode)
    if mode == "adjoint_corrected" and create_graph:
        raise ValueError("adjoint_corrected currently supports first derivatives only")
    if state.ndim != 1 or multipliers.ndim != 1:
        raise ValueError("state and multipliers must be rank-1 tensors")
    if not coordinates:
        raise ValueError("at least one coordinate tensor is required")
    for name, coordinate in coordinates.items():
        if not coordinate.is_floating_point() or not coordinate.requires_grad:
            raise ValueError("coordinate %s must be floating point with gradients" % name)
        if coordinate.dtype != state.dtype or coordinate.device != state.device:
            raise ValueError("coordinates must match state dtype and device")
    if multipliers.dtype != state.dtype or multipliers.device != state.device:
        raise ValueError("multipliers must match state dtype and device")
    if not bool(torch.isfinite(state).all().item()) or not bool(
        torch.isfinite(multipliers).all().item()
    ):
        raise ValueError("state and multipliers must be finite")
    if (
        not math.isfinite(residual_tolerance)
        or not math.isfinite(feasibility_tolerance)
        or residual_tolerance < 0.0
        or feasibility_tolerance < 0.0
    ):
        raise ValueError("force tolerances must be finite and non-negative")

    fixed_state = state.detach().requires_grad_(True)
    fixed_multipliers = multipliers.detach()
    objective = objective_fn(fixed_state)
    constraints = constraint_fn(fixed_state)
    if not isinstance(objective, torch.Tensor) or objective.numel() != 1:
        raise ValueError("objective_fn must return a scalar tensor")
    if not isinstance(constraints, torch.Tensor) or constraints.shape != multipliers.shape:
        raise ValueError("constraint_fn must return one value per multiplier")
    if objective.dtype != state.dtype or objective.device != state.device:
        raise ValueError("objective must match state dtype and device")
    if constraints.dtype != state.dtype or constraints.device != state.device:
        raise ValueError("constraints must match state dtype and device")
    lagrangian = objective.reshape(()) + fixed_multipliers @ constraints
    stationarity = torch.autograd.grad(
        lagrangian,
        fixed_state,
        create_graph=mode == "adjoint_corrected",
        retain_graph=True,
    )[0]
    if not bool(torch.isfinite(stationarity).all().item()) or not bool(
        torch.isfinite(constraints).all().item()
    ):
        raise ValueError("variational residuals must be finite")
    explicit = _coordinate_gradients(
        lagrangian,
        coordinates,
        create_graph=create_graph,
        retain_graph=True,
    )

    adjoint_state = None
    adjoint_multiplier = None
    if mode == "envelope":
        gradients = explicit
    elif mode == "full_differentiable":
        full_objective = objective_fn(state)
        if not isinstance(full_objective, torch.Tensor) or full_objective.numel() != 1:
            raise ValueError("objective_fn must return a scalar tensor")
        if full_objective.dtype != state.dtype or full_objective.device != state.device:
            raise ValueError("objective must match state dtype and device")
        gradients = _coordinate_gradients(
            full_objective.reshape(()),
            coordinates,
            create_graph=create_graph,
            retain_graph=True,
        )
    else:
        primal_norm = constraints.abs().max() if constraints.numel() else constraints.new_zeros(())
        if bool((primal_norm > feasibility_tolerance).item()):
            raise ValueError("adjoint correction requires a feasible state")
        n_state = state.numel()
        n_constraints = multipliers.numel()
        hessian = torch.autograd.functional.hessian(
            lambda value: objective_fn(value)
            + fixed_multipliers @ constraint_fn(value),
            fixed_state,
            create_graph=False,
        ).detach()
        hessian = 0.5 * (hessian + hessian.transpose(0, 1))
        if n_constraints:
            jacobian = torch.autograd.functional.jacobian(
                constraint_fn, fixed_state, create_graph=False
            ).detach()
        else:
            jacobian = state.new_zeros((0, n_state))
        kkt = torch.cat(
            (
                torch.cat((hessian, jacobian.transpose(0, 1)), dim=1),
                torch.cat(
                    (
                        jacobian,
                        state.new_zeros((n_constraints, n_constraints)),
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        )
        right = torch.cat((stationarity.detach(), state.new_zeros(n_constraints)))
        try:
            adjoint = torch.linalg.solve(kkt.transpose(0, 1), right)
        except RuntimeError as error:
            raise SingularKKTError("adjoint force correction failed") from error
        adjoint_state = adjoint[:n_state].detach()
        adjoint_multiplier = adjoint[n_state:].detach()
        surrogate = (
            lagrangian
            - adjoint_state @ stationarity
            - adjoint_multiplier @ constraints
        )
        gradients = _coordinate_gradients(
            surrogate,
            coordinates,
            create_graph=create_graph,
            retain_graph=True,
        )

    correction = _subtract(gradients, explicit)
    stationarity_norm = (
        stationarity.abs().max() if stationarity.numel() else stationarity.new_zeros(())
    )
    primal_norm = constraints.abs().max() if constraints.numel() else constraints.new_zeros(())
    certified = bool(
        (stationarity_norm <= residual_tolerance).item()
        and (primal_norm <= feasibility_tolerance).item()
    )
    forces = -gradients["positions"] if "positions" in gradients else None
    return VariationalDerivative(
        mode=mode,
        gradients=gradients,
        explicit_gradients=explicit,
        response_correction=correction,
        forces=forces,
        stationarity_residual=stationarity.detach(),
        constraint_residual=constraints.detach(),
        adjoint_state=adjoint_state,
        adjoint_multiplier=adjoint_multiplier,
        kkt_certified=certified,
        conservative_for_reported_energy=mode == "full_differentiable",
    )


__all__ = [
    "ForceMode",
    "VariationalDerivative",
    "differentiate_variational_energy",
]
