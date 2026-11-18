"""Opt-in convex scalar charge energy with differentiable all-pairs geometry."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ._neural_charge_state_energy import NeuralChargeStateEnergy


class GeometricNeuralChargeStateEnergy(NeuralChargeStateEnergy):
    """Add geometry-aware scalar pair energies; derive all curvature from U.

    For i < j and channel c, the added energy is
      sum w_ijc [softplus(b_ijc + a_ic q_i + a_jc q_j) - softplus(b_ijc)] / C.
    Both b and positive w depend on invariant local features and all-pairs
    distances. They are NOT independent charge Hessian predictions. Exact HVPs
    still differentiate this scalar energy. An analytic chain-rule diagonal
    avoids a separate backward for every atom in the diagonal+Ewald comparison.

    Smooth radial Gaussians have scales from 0.5 to max_length Angstrom, without
    a hard cutoff. The widest Gaussian e_ij also supplies normalization
    e_ij / sqrt(d_i*d_j), d_i=1+sum_{j!=i} e_ij, limiting atom-count growth.
    Disconnected subsystems recover independent normalization as separation
    tends to infinity. This is a learned geometric descriptor, not an Ewald
    kernel or a prescribed Coulomb law. Periodic distances use smooth lattice
    phases, sharing the proposal's convention; they are not an image sum.

    Convexity in q follows from positive weights and affine softplus arguments,
    plus the inherited fixed curvature floor. This restricted convex neural
    architecture is a starting implementation, not an unrestricted functional.
    """
    def __init__(self, scalar_channels=None, hidden_features=64, curvature_floor=20.0,
                 pair_features=16, radial_features=8, max_length=32.0):
        super().__init__(None, hidden_features, curvature_floor)
        if pair_features < 1 or radial_features < 2 or not math.isfinite(max_length) or max_length <= 0.5:
            raise ValueError('positive pair width, radial width >= 2, and max_length > 0.5 required')
        self.pair_features = int(pair_features)
        self.radial_features = int(radial_features)
        self.max_length = float(max_length)
        self.register_buffer('radial_lengths', torch.logspace(math.log10(0.5), math.log10(max_length), radial_features))
        self.pair_embed = None
        self.pair_geometry = None
        if scalar_channels is not None:
            self.bind(scalar_channels)

    def bind(self, scalar_channels, **kwargs):
        super().bind(scalar_channels, **kwargs)
        if self.pair_embed is None:
            width = self.pair_features
            self.pair_embed = nn.Linear(scalar_channels, width)
            self.pair_geometry = nn.Sequential(
                nn.Linear(2 * width + self.radial_features, 2 * width), nn.SiLU(),
                nn.Linear(2 * width, 2 * width))
        return self

    def _pair_terms(self, context, state):
        n_atoms = state.shape[0]
        positions = context['positions']
        left, right = torch.triu_indices(n_atoms, n_atoms, offset=1, device=state.device)
        displacement = positions[right] - positions[left]
        pbc = context['pbc']
        if bool(pbc.any()):
            cell = context['cell']
            fractional = torch.linalg.solve(cell.T, displacement.T).T
            phase = 2 * math.pi * fractional
            metric = torch.where(pbc, 2 * (1 - torch.cos(phase)) / (2 * math.pi) ** 2,
                                 fractional.square())
            squared_distance = (metric * cell.square().sum(-1)).sum(-1)
        else:
            squared_distance = displacement.square().sum(-1)
        radial = torch.exp(-squared_distance[:, None] / self.radial_lengths.square())
        envelope = radial[:, -1]
        degree = torch.ones(n_atoms, dtype=state.dtype, device=state.device)
        degree = degree.index_add(0, left, envelope).index_add(0, right, envelope)
        normalization = envelope / torch.sqrt(degree[left] * degree[right])
        embedding = torch.tanh(self.pair_embed(context['scalar']))
        pair_input = torch.cat((embedding[left] + embedding[right],
                                embedding[left] * embedding[right], radial), dim=-1)
        bias, raw_weight = self.pair_geometry(pair_input).chunk(2, dim=-1)
        weight = F.softplus(raw_weight) * normalization[:, None] / self.pair_features
        argument = bias + embedding[left] * state[left] + embedding[right] * state[right]
        return left, right, embedding, bias, argument, weight

    def forward(self, context, state):
        base = super().forward(context, state)
        _, _, _, bias, argument, weight = self._pair_terms(context, state)
        return base + (weight * (F.softplus(argument) - F.softplus(bias))).sum()

    @torch.enable_grad()
    def state_hessian_diagonal(self, context, state, *, create_graph=True):
        diagonal = super().state_hessian_diagonal(context, state, create_graph=True)
        left, right, embedding, _, argument, weight = self._pair_terms(context, state)
        sigmoid = torch.sigmoid(argument)
        # Match torch softplus's linear branch above its default threshold.
        second_derivative = torch.where(argument > 20, torch.zeros_like(argument),
                                        sigmoid * (1 - sigmoid))
        curvature = weight * second_derivative
        left_diagonal = (curvature * embedding[left].square()).sum(-1, keepdim=True)
        right_diagonal = (curvature * embedding[right].square()).sum(-1, keepdim=True)
        diagonal = diagonal.index_add(0, left, left_diagonal).index_add(0, right, right_diagonal)
        return diagonal if create_graph else diagonal.detach()

    def state_taylor_terms(self, context, state, direction, *, diagonal=False):
        """Exact affine-softplus pair chain rule; no dense charge Hessian."""
        value, gradient, action, reference = super().state_taylor_terms(
            context, state, direction, diagonal=diagonal)
        left, right, embedding, bias, argument, weight = self._pair_terms(context, state)
        first, second = self._softplus_derivatives(argument)
        a, b = embedding[left], embedding[right]
        value = value + (weight * (F.softplus(argument) - F.softplus(bias))).sum()
        gradient = gradient.index_add(0, left, (weight * first * a).sum(-1, keepdim=True))
        gradient = gradient.index_add(0, right, (weight * first * b).sum(-1, keepdim=True))
        curvature = weight * second
        if diagonal:
            left_action = (curvature * a.square()).sum(-1, keepdim=True) * direction[left]
            right_action = (curvature * b.square()).sum(-1, keepdim=True) * direction[right]
        else:
            pair_direction = a * direction[left] + b * direction[right]
            left_action = (curvature * a * pair_direction).sum(-1, keepdim=True)
            right_action = (curvature * b * pair_direction).sum(-1, keepdim=True)
        action = action.index_add(0, left, left_action).index_add(0, right, right_action)
        return value, gradient, action, reference

    def get_extra_state(self):
        return {**super().get_extra_state(), 'geometry_schema': 'all_pair_affine_softplus_v1',
                'pair_features': self.pair_features, 'radial_features': self.radial_features,
                'max_length_angstrom': self.max_length,
                'pair_normalization': 'gaussian_symmetric_degree_plus_one',
                'periodic_metric': 'smooth_lattice_phase_cell_lengths'}
