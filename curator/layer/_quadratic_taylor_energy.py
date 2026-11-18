"""Quadratic Taylor energy at an explicit geometry-dependent expansion state."""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn


class QuadraticTaylorEnergy(nn.Module):
    """Quadraticize an existing scalar energy at an explicit expansion state.

    With q0 = context['expansion_state'] (zero when absent), d = q - q0,
    and partial state derivatives g0, H0 of base U evaluated at q0:

      full_hvp:       U(q0) + g0.d + .5 d.H0.d
      diagonal_ewald: U(q0) + g0.d + .5 sum_i H0_ii d_i^2 + .5 q.J.q

    ``subtract_u0=True`` subtracts U(context, zero charge), NOT U(context, q0).
    This preserves the moving anchor's energy and total geometry/parameter
    derivatives while matching a separately supplied zero-charge baseline. No
    Ewald anchor energy or linear term is subtracted: J acts on actual q, so the
    diagonal mode's partial state gradient includes J q, including J q0 at q0.

    The expansion state may depend on geometry and model parameters but must
    remain independent of the current proposal correction. For each fixed
    context the energy is strictly quadratic in q. A differentiable clone of
    q0 is the independent state variable for partial derivatives, avoiding
    accidental differentiation of context paths with respect to shared q0.
    Its graph back to the predicted q0 remains intact for total derivatives.

    By default Hessian actions use reverse-over-reverse AD, never a full Hessian, Gram
    parameterization or spectral projection. The exact diagonal uses the base's
    optional state_hessian_diagonal interface or an elementwise AD fallback.
    Only diagonal_ewald constructs the existing Ewald kernel. Neither mode adds
    trainable parameters. Geometry, base-parameter and direction gradients stay
    live, including higher derivatives needed by force/residual training. The
    base must support these derivatives. No positivity is imposed. Boundaries
    and final state projection belong to the surrounding ACVR compiler/head.

    ``derivative_backend="analytic"`` opts into the base energy's exact
    ``state_taylor_terms`` protocol. It evaluates the SAME derivatives by the
    chain rule without nested state-autograd calls; it is neither a fitted
    curvature head nor an approximation. Unsupported energies raise an error.
    Backend selection adds no parameters and does not change the mathematical
    checkpoint contract; old pickled modules default to the autograd path.
    """

    def __init__(self, base_energy: nn.Module, subtract_u0: bool = True,
                 curvature_mode: str = "full_hvp", ewald=None,
                 derivative_backend: str = "autograd"):
        super().__init__()
        if not isinstance(base_energy, nn.Module):
            raise TypeError("base_energy must be a torch.nn.Module")
        if curvature_mode not in {"full_hvp", "diagonal_ewald"}:
            raise ValueError("curvature_mode must be full_hvp or diagonal_ewald")
        if curvature_mode == "full_hvp" and ewald is not None:
            raise ValueError("full_hvp does not use an Ewald kernel")
        if derivative_backend not in {"autograd", "analytic"}:
            raise ValueError("derivative_backend must be autograd or analytic")
        self.derivative_backend = derivative_backend
        self.curvature_mode = curvature_mode
        self.ewald = None
        if curvature_mode == "diagonal_ewald":
            from ._ewald import EwaldSummation
            self.ewald = EwaldSummation() if ewald is None else ewald
        self.base_energy = base_energy
        self.subtract_u0 = bool(subtract_u0)

    def bind(self, **kwargs):
        if hasattr(self.base_energy, "bind"):
            self.base_energy.bind(**kwargs)
        return self

    def initial_state(self, context):
        if hasattr(self.base_energy, "initial_state"):
            return self.base_energy.initial_state(context)
        return torch.zeros_like(context["origin"])

    def geometry_context(self, data):
        from curator.data import properties
        contexts = (self.base_energy.geometry_context(data)
                    if hasattr(self.base_energy, "geometry_context")
                    else [{} for _ in data[properties.n_atoms]])
        if self.curvature_mode == "diagonal_ewald":
            from curator.variational.backends.ewald import EwaldChargeOperator
            kernels = EwaldChargeOperator(self.ewald).geometry_kernels(data)
            if len(contexts) != len(kernels):
                raise ValueError("base geometry contexts and Ewald kernels differ")
            contexts = [dict(context, ewald_kernel=kernel)
                        for context, kernel in zip(contexts, kernels)]
        return contexts

    def _diagonal(self, context, origin, gradient0):
        if hasattr(self.base_energy, "state_hessian_diagonal"):
            result = self.base_energy.state_hessian_diagonal(context, origin, create_graph=True)
            if (result.shape != origin.shape or result.dtype != origin.dtype
                    or result.device != origin.device):
                raise ValueError("exact Hessian diagonal must match state shape, dtype and device")
            return result
        # General exact fallback: each derivative is a state-sized vector and
        # only its corresponding diagonal component is retained. Never stack
        # Hessian rows or materialize a state-by-state Hessian. Structured base
        # energies may expose the exact faster interface above.
        diagonal = []
        flat_gradient = gradient0.reshape(-1)
        flat_origin = origin.reshape(-1)
        for index in range(origin.numel()):
            row = torch.autograd.grad(flat_gradient[index] + flat_origin[index] * 0,
                                      origin, create_graph=True, retain_graph=True)[0]
            diagonal.append(row.reshape(-1)[index])
        return torch.stack(diagonal).reshape_as(origin)

    def forward(self, context, state):
        if not isinstance(state, torch.Tensor) or not state.is_floating_point():
            raise TypeError("state must be a floating-point tensor")
        if state.numel() == 0:
            raise ValueError("state must be nonempty")
        backend = getattr(self, "derivative_backend", "autograd")
        # Exact chain-rule arithmetic needs no internal AD for energy-only use.
        with (nullcontext() if backend == "analytic" else torch.enable_grad()):
            anchor = context.get("expansion_state")
            if anchor is None:
                origin = torch.zeros_like(state, requires_grad=backend != "analytic")
            else:
                if (not isinstance(anchor, torch.Tensor) or anchor.shape != state.shape
                        or anchor.dtype != state.dtype or anchor.device != state.device):
                    raise ValueError("expansion_state must match state shape, dtype and device")
                # Distinct AD node for partial state differentiation, retaining
                # the identity edge back to q0's geometry/parameter predictor.
                origin = anchor.clone()
                if backend != "analytic":
                    origin.requires_grad_(True)
            direction = state - origin
            if backend == "analytic":
                if not hasattr(self.base_energy, "state_taylor_terms"):
                    raise TypeError("analytic backend requires exact state_taylor_terms")
                value0, gradient0, action, reference = self.base_energy.state_taylor_terms(
                    context, origin, direction, diagonal=self.curvature_mode == "diagonal_ewald")
                if not self.subtract_u0:
                    reference = value0.new_zeros(())
            elif backend == "autograd":
                value0 = self.base_energy(context, origin)
                if not isinstance(value0, torch.Tensor) or value0.ndim != 0:
                    raise ValueError("base_energy must return a scalar tensor")
                if value0.dtype != state.dtype or value0.device != state.device:
                    raise ValueError("base energy and state must share dtype and device")
                # Zero paths make constant/linear base energies well-defined without
                # discarding parameter/geometry graphs from their lower derivatives.
                gradient0 = torch.autograd.grad(
                    value0 + origin.sum() * 0, origin, create_graph=True,
                )[0]
                if self.curvature_mode == "full_hvp":
                    action = torch.autograd.grad(
                        gradient0 + origin * 0, origin, grad_outputs=direction,
                        create_graph=True,
                    )[0]
                else:
                    if state.ndim != 2 or state.shape[1] != 1:
                        raise ValueError("diagonal_ewald requires scalar charge [n_atoms, 1]")
                    kernel = context["ewald_kernel"]
                    if kernel.shape != (state.shape[0], state.shape[0]):
                        raise ValueError("Ewald kernel must match atom count")
                    diagonal = self._diagonal(context, origin, gradient0)
                    action = diagonal * direction
                # grad_outputs acts as the cotangent during the HVP but remains part
                # of the returned AD graph. Detaching it loses half the state
                # derivative of the quadratic term and corrupts force training.
                reference = (self.base_energy(context, torch.zeros_like(state))
                             if self.subtract_u0 else value0.new_zeros(()))
            else:
                raise ValueError("derivative_backend must be autograd or analytic")
            energy = value0 - reference + (gradient0 * direction).sum() + 0.5 * (direction * action).sum()
            if self.curvature_mode == "diagonal_ewald":
                if state.ndim != 2 or state.shape[1] != 1:
                    raise ValueError("diagonal_ewald requires scalar charge [n_atoms, 1]")
                kernel = context["ewald_kernel"]
                if kernel.shape != (state.shape[0], state.shape[0]):
                    raise ValueError("Ewald kernel must match atom count")
                charge = state[:, 0]
                energy = energy + 0.5 * charge @ kernel @ charge
            return energy

    def get_extra_state(self):
        result = {"schema": "context_anchor_quadratic_taylor_v2",
                  "expansion_point": "context_expansion_state_else_zero",
                  "subtract_u0": self.subtract_u0, "reference_state": "zero_charge",
                  "ewald_state": "actual_charge", "curvature_mode": self.curvature_mode}
        if self.ewald is not None:
            result["ewald"] = {"alpha": self.ewald.alpha, "cutoff": self.ewald.cutoff,
                               "k_cutoff": self.ewald.k_cutoff,
                               "neutralizing_background": self.ewald.neutralizing_background}
        return result

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("quadratic Taylor expansion contract changed")
