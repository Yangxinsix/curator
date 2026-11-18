"""Relaxed energy derivatives at an independently validated stable KKT solution."""

from __future__ import annotations

import torch

from .functional_response import solve_functional_linear_response
from .functionals import FunctionalProblem
from .observables import EnergyObservables


class StationaryEnergyObservables(EnergyObservables):
    """Evaluate first/second derivatives including the implicit state response.

    The caller must supply a converged stable solution of ``problem``. State and
    multipliers become independent leaves; the energy and linear constraints
    retain their live dependence on the named perturbations. Thus geometry-
    dependent constraint matrices and varying constraint targets are included.

    Only evaluation is supported: results are detached, including when the
    training head normally retains property graphs. Linearized KKT systems use
    Hessian-vector products, never a dense state Hessian. Dense observable
    derivatives retain the parent's small-first-input limit and presentation.
    """

    def __init__(self, problem: FunctionalProblem, state, multipliers, inputs, *,
                 normalizers=None, max_jacobian_rows=9, tolerance=1e-10,
                 create_graph=False):
        if create_graph:
            raise ValueError("stationary observables are evaluation-only; create_graph must be False")
        problem._validate_state(state)
        if multipliers.shape != (problem.n_constraints,):
            raise ValueError("multipliers must match the problem's equality constraints")
        if multipliers.dtype != state.dtype or multipliers.device != state.device:
            raise ValueError("multipliers must share state dtype and device")
        self.problem = problem
        self.state = state.detach().requires_grad_(True)
        self.multipliers = multipliers.detach().requires_grad_(True)
        self.tolerance = tolerance
        self._responses = {}
        with torch.enable_grad():
            energy = problem.energy(self.state)
            self.constraint = problem.constraints(self.state)
            self.lagrangian = energy + self.multipliers @ self.constraint
            self.stationarity = self._gradient(self.lagrangian, self.state, True)
        super().__init__(energy, inputs, state_mode="converged", normalizers=normalizers,
                         max_jacobian_rows=max_jacobian_rows)

    def _check_mode(self, order, create_graph):
        # Internal partial derivatives need graphs to form the KKT RHS. Public
        # methods below disallow exposing those partial graphs as relaxed ones.
        if order > 2:
            raise ValueError("stationary observables support derivative orders zero, one and two")

    def _response(self, partials):
        state_rhs = torch.stack([-self._gradient(value, self.state, False)
                                 for value in partials], dim=1)
        equality_rhs = torch.stack([-self._gradient(value, self.multipliers, False)
                                    for value in partials], dim=1)
        response = solve_functional_linear_response(
            self.problem, self.state, state_rhs, equality_rhs,
            tolerance=self.tolerance, create_graph=False,
        )
        if not response.converged:
            raise RuntimeError("stationary observable linear response failed: " + response.status)
        return response

    def _raw_derivative(self, variables, create_graph):
        if not variables:
            return self.energy
        self._check_mode(len(variables), create_graph)
        key = (variables, create_graph)
        if key in self._cache:
            return self._cache[key]
        variable = self._input(variables[-1])
        if len(variables) == 1:
            result = self._gradient(self.lagrangian, variable, create_graph)
        else:
            first = self._input(variables[0])
            if first.numel() == 0 or variable.numel() == 0:
                result = first.new_zeros(first.shape + variable.shape)
            else:
                if first.numel() > self.max_jacobian_rows:
                    raise ValueError("dense stationary second derivative exceeds max_jacobian_rows; "
                                     "put the small perturbation first, or request hvp")
                partials = self._raw_derivative(variables[:1], True).reshape(-1)
                if variables[0] not in self._responses:
                    self._responses[variables[0]] = self._response(partials)
                response = self._responses[variables[0]]
                rows = []
                for index, partial in enumerate(partials):
                    correction = (self.stationarity @ response.state[:, index]
                                  + self.constraint @ response.multipliers[:, index])
                    rows.append(self._gradient(partial + correction, variable, False))
                result = torch.stack(rows).reshape(first.shape + variable.shape)
        self._cache[key] = result
        return result

    def derivative(self, variables, *, create_graph=None):
        if create_graph:
            raise ValueError("stationary observables are evaluation-only; create_graph must be False")
        return super().derivative(variables, create_graph=False)

    def hvp(self, variable, direction, *, create_graph=None):
        """Relaxed HVP using one linearized KKT solve, with no dense input Hessian."""
        if create_graph:
            raise ValueError("stationary observables are evaluation-only; create_graph must be False")
        point = self._input(variable)
        if not isinstance(direction, torch.Tensor) or direction.shape != point.shape:
            raise ValueError("HVP direction must have the same shape as the input")
        if direction.dtype != point.dtype or direction.device != point.device:
            raise ValueError("HVP direction must share the input dtype and device")
        if point.numel() == 0:
            return torch.zeros_like(point)
        with torch.enable_grad():
            partial = (self._raw_derivative((variable,), True) * direction.detach()).sum()
            response = self._response(partial.reshape(1))
            correction = (self.stationarity @ response.state[:, 0]
                          + self.constraint @ response.multipliers[:, 0])
            return self._gradient(partial + correction, point, False).detach()


__all__ = ["StationaryEnergyObservables"]
