"""Energy-first, nonquadratic charge functional without a prescribed Coulomb kernel."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class NeuralChargeEnergy(nn.Module):
    """Convex neural local energy plus learned nonlinear global charge modes.

    This is an initial general-functional baseline, not a universal long-range
    electrostatics model. Geometry supplies invariant features; boundary controls
    enter only through the surrounding variational compiler. Gradient/HVP are
    obtained from this scalar energy, never from separate chi/hardness heads.

    Global modes sum over the atoms of ONE geometry. Their normalization keeps
    energy extensive for replicated identical environments. Softplus of affine
    charge functions with positive weights and a quadratic floor guarantees
    convexity for a fixed geometry. Mixed curvature is learned and state-dependent.
    """
    def __init__(self, scalar_channels=None, hidden_features=16, curvature_floor=20.0):
        super().__init__()
        if hidden_features < 1 or not math.isfinite(curvature_floor) or curvature_floor <= 0:
            raise ValueError("positive width and finite positive curvature_floor required")
        self.hidden_features = int(hidden_features)
        self.curvature_floor = float(curvature_floor)
        self.scalar_channels = None
        self.geometry = None
        if scalar_channels is not None:
            self.bind(scalar_channels)

    def bind(self, scalar_channels, vector_channels=0, components=1, state_kind="scalar", **kwargs):
        if state_kind != "scalar" or components != 1:
            raise ValueError("NeuralChargeEnergy requires one scalar charge per atom")
        if scalar_channels < 1:
            raise ValueError("scalar geometric features required")
        if self.geometry is not None:
            if self.scalar_channels != scalar_channels:
                raise ValueError("energy already bound to different features")
            return self
        self.scalar_channels = int(scalar_channels)
        width = self.hidden_features
        self.geometry = nn.Sequential(nn.Linear(scalar_channels, width), nn.SiLU(),
                                      nn.Linear(width, 2 * width))
        self.local_slopes = nn.Parameter(torch.linspace(-1, 1, width))
        self.local_weights = nn.Parameter(torch.zeros(width))
        self.global_bias = nn.Parameter(torch.linspace(-0.5, 0.5, width))
        self.global_weights = nn.Parameter(torch.zeros(width))
        return self

    def forward(self, context, state):
        if self.geometry is None:
            raise RuntimeError("bind energy before evaluation")
        if state.ndim != 2 or state.shape[1] != 1:
            raise ValueError("charge state must have shape [n_atoms, 1]")
        bias, modes = self.geometry(context["scalar"]).chunk(2, dim=-1)
        local = F.softplus(bias + state * self.local_slopes) - F.softplus(bias)
        # O(N * width) global coupling; no Ewald and no dense Hessian required.
        collective = (torch.tanh(modes) * state).mean(dim=0)
        global_energy = F.softplus(self.global_bias + collective) - F.softplus(self.global_bias)
        return (0.5 * self.curvature_floor * state.square().sum()
                + (local * F.softplus(self.local_weights)).sum() / self.hidden_features
                + state.shape[0] * (global_energy * F.softplus(self.global_weights)).sum()
                / self.hidden_features)

    @torch.enable_grad()
    def state_hessian_diagonal(self, context, state, *, create_graph=True):
        """Exact charge-Hessian diagonal in O(N * width), with higher-order AD.

        Local charge terms are atomwise independent. Global channels are also
        separable, and each collective coordinate is linear in all charges.
        Differentiate those scalar energies, then apply the exact chain rule;
        no dense Hessian, stochastic estimator, or charge-coupling term is lost.
        """
        if self.geometry is None:
            raise RuntimeError("bind energy before evaluation")
        if state.ndim != 2 or state.shape[1] != 1:
            raise ValueError("charge state must have shape [n_atoms, 1]")
        charge = state if state.requires_grad else state.detach().requires_grad_(True)
        bias, modes = self.geometry(context["scalar"]).chunk(2, dim=-1)
        local = F.softplus(bias + charge * self.local_slopes) - F.softplus(bias)
        local_energy = (0.5 * self.curvature_floor * charge.square().sum()
                        + (local * F.softplus(self.local_weights)).sum()
                        / self.hidden_features)
        local_gradient, = torch.autograd.grad(local_energy, charge, create_graph=True)
        local_diagonal, = torch.autograd.grad(
            local_gradient.sum(), charge, create_graph=True)
        weights = torch.tanh(modes) / charge.shape[0]
        collective = (torch.tanh(modes) * charge).mean(dim=0)
        global_energy = (charge.shape[0]
                         * ((F.softplus(self.global_bias + collective)
                             - F.softplus(self.global_bias))
                            * F.softplus(self.global_weights)).sum()
                         / self.hidden_features)
        global_gradient, = torch.autograd.grad(
            global_energy, collective, create_graph=True)
        global_diagonal, = torch.autograd.grad(
            global_gradient.sum(), collective, create_graph=True)
        diagonal = local_diagonal + (weights.square() * global_diagonal).sum(
            dim=-1, keepdim=True)
        return diagonal if create_graph else diagonal.detach()

    @staticmethod
    def _softplus_derivatives(argument):
        """Exact derivatives of torch softplus (beta=1, threshold=20)."""
        sigmoid = torch.sigmoid(argument)
        # Native pointwise derivative matches softplus's rounding and its
        # one-sided higher-derivative convention exactly at the threshold.
        # This is one tensor kernel, not an autograd.grad / nested AD pass.
        first = torch.ops.aten.softplus_backward(torch.ones_like(argument), argument, 1, 20)
        second = torch.where(argument < 20, sigmoid * (1 - sigmoid), torch.zeros_like(argument))
        return first, second

    def state_taylor_terms(self, context, state, direction, *, diagonal=False):
        """Return U, dU/dq, exact curvature action, U(0), without nested AD.

        This is the chain rule for this energy, not a second parameterization.
        All returned tensors retain geometry, parameter, state and direction
        derivatives. Geometry intermediates belong only to this call's graph.
        """
        if self.geometry is None:
            raise RuntimeError('bind energy before evaluation')
        if state.ndim != 2 or state.shape[1] != 1 or direction.shape != state.shape:
            raise ValueError('state and direction must have shape [n_atoms, 1]')
        bias, modes = self.geometry(context['scalar']).chunk(2, dim=-1)
        argument = bias + state * self.local_slopes
        weights = F.softplus(self.local_weights) / self.hidden_features
        first, second = self._softplus_derivatives(argument)
        value = (0.5 * self.curvature_floor * state.square().sum()
                 + ((F.softplus(argument) - F.softplus(bias)) * weights).sum())
        gradient = self.curvature_floor * state + (weights * first * self.local_slopes).sum(-1, keepdim=True)
        curvature = self.curvature_floor + (weights * second * self.local_slopes.square()).sum(-1, keepdim=True)
        action = curvature * direction
        # Each collective coordinate is affine in all charges.
        n = state.shape[0]
        mode = torch.tanh(modes)
        argument = self.global_bias + (mode * state).mean(0)
        weights = n * F.softplus(self.global_weights) / self.hidden_features
        first, second = self._softplus_derivatives(argument)
        value = value + ((F.softplus(argument) - F.softplus(self.global_bias)) * weights).sum()
        jacobian = mode / n
        gradient = gradient + (jacobian * (weights * first)).sum(-1, keepdim=True)
        if diagonal:
            action = action + (jacobian.square() * (weights * second)).sum(-1, keepdim=True) * direction
        else:
            collective_direction = (jacobian * direction).sum(0)
            action = action + (jacobian * (weights * second * collective_direction)).sum(-1, keepdim=True)
        # This energy is identically zero at q=0, for every geometry/parameter.
        return value, gradient, action, value.new_zeros(())

    def get_extra_state(self):
        return {"schema": 1, "curvature_floor": self.curvature_floor,
                "hidden_features": self.hidden_features, "units": ("e", "eV"),
                "global_mode_normalization": "mean_extensive"}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("neural charge energy contract changed")
