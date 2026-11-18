"""Euclidean fast attention with the original Curator kernel and checkpoint layout.

ERoPE, spherical integration and the raw-lattice periodic specialization follow
Frank et al., arXiv:2412.08541. Small structures may use an equivalent pairwise
contraction; large structures retain the atom-count-linear contraction.
"""
import itertools
import math
from typing import NamedTuple

import torch
from torch import nn

from .geometry import encoding_grid, rotary_encoding


class PreparedAttention(NamedTuple):
    """Single-structure ERoPE tensors, reusable within one forward graph."""

    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor


def _lebedev_grid(num):
    """Normalized standard Lebedev-Laikov rules, expanded from symmetry orbits.

    These numerical quadrature constants also appear in e3x's Lebedev tables
    (Apache-2.0; google-research/e3x, e3x/so3/_lebedev_grids.npz).
    """
    axis = ((0., 0., 1.),)
    corner = ((1 / math.sqrt(3),) * 3,)
    rules = {
        50: [
            (axis[0], .012698412698412974),
            ((0., 1 / math.sqrt(2), 1 / math.sqrt(2)), .022574955908288955),
            (corner[0], .02109374999999996),
            ((.30151134457776, .30151134457776, .90453403373329), .02017333553791896),
        ],
        110: [
            (axis[0], .003828270494937007),
            (corner[0], .009793737512488018),
            ((.18511563534474, .18511563534474, .96512403508659), .008211737283191014),
            ((.21595729184585, .69042104838229, .69042104838229), .009942814891178018),
            ((.39568947305594, .39568947305594, .82876998125259), .009595471336071018),
            ((0., .47836902881215, .87815891060407), .009694996361663018),
        ],
        194: [
            (axis[0], .0017823404472450107),
            ((0., 1 / math.sqrt(2), 1 / math.sqrt(2)), .005716905949977035),
            (corner[0], .005573383178849034),
            ((.31419699418259, .67129734426952, .67129734426952), .005608704082588034),
            ((.28924656275754, .28924656275754, .91250909686747), .005158237711805031),
            ((.44469331787174, .44469331787174, .77749321931477), .005518771467274033),
            ((.12993354476501, .12993354476501, .98297230270725), .004106777028169024),
            ((0., .34577021976113, .93831921813759), .00505184606461503),
            ((.15904171053835, .52511857244364, .83603601548246), .005530248916233033),
        ],
    }
    if num not in rules:
        raise ValueError("num_directions must be 50, 110 or 194")
    points, weights = [], []
    for xyz, weight in rules[num]:
        orbit = sorted({tuple(a * b for a, b in zip(p, signs))
                        for p in itertools.permutations(xyz)
                        for signs in itertools.product((-1., 1.), repeat=3)})
        points.extend(orbit)
        weights.extend([weight] * len(orbit))
    if len(points) != num:
        raise RuntimeError("invalid Lebedev symmetry orbit")
    return torch.tensor(points, dtype=torch.float64), torch.tensor(weights, dtype=torch.float64)


class EuclideanFastAttention(nn.Module):
    """EFA ERoPE core with equivalent linear and pairwise contractions.

    Queries/keys are [N, angular_components, even_channels]. Rotary encoding
    acts on channel pairs and commutes with rotations of angular components.
    Values have shape [N, value_channels]. Different structures are handled
    independently by the adapter. All atoms, including self, enter the sum.
    ``auto`` compares contraction multiply-add counts and caps the pairwise
    matrix at ``pairwise_max_atoms ** 2`` entries. No softmax is applied.
    """

    def __init__(self, hidden_features=32, num_directions=110,
                 max_frequency=2 * math.pi, max_length=32.0,
                 contraction="auto", pairwise_max_atoms=256):
        super().__init__()
        if hidden_features < 2 or hidden_features % 2:
            raise ValueError("hidden_features must be positive and even")
        if not math.isfinite(max_length) or max_length <= 0:
            raise ValueError("max_length must be finite and positive")
        if not math.isfinite(max_frequency) or max_frequency <= 0:
            raise ValueError("max_frequency must be finite and positive")
        if contraction not in ("linear", "pairwise", "auto"):
            raise ValueError("contraction must be linear, pairwise or auto")
        if not isinstance(pairwise_max_atoms, int) or pairwise_max_atoms < 1:
            raise ValueError("pairwise_max_atoms must be a positive integer")
        safe_max = {50: math.pi, 110: 2.5 * math.pi, 194: 4 * math.pi}
        if num_directions is None:
            grid = torch.empty((0, 3), dtype=torch.float64)
            weights = torch.empty(0, dtype=torch.float64)
        else:
            grid, weights = _lebedev_grid(num_directions)
            if max_frequency > safe_max[num_directions]:
                raise ValueError("frequency exceeds the official quadrature guideline; increase num_directions")
        self.hidden_features = hidden_features
        self.num_directions = num_directions
        self.max_frequency = float(max_frequency)
        self.max_length = float(max_length)
        self.contraction = contraction
        self.pairwise_max_atoms = int(pairwise_max_atoms)
        self.register_buffer("grid", grid)
        self.register_buffer("weights", weights)
        frequency = (torch.linspace(0., max_frequency, hidden_features // 2,
                                    dtype=torch.float64) if hidden_features > 2
                     else torch.tensor([max_frequency], dtype=torch.float64))
        self.register_buffer("frequencies", frequency / max_length)

    def rotary(self, values, positions, cell=None):
        """Encode positions using spherical directions or a raw row-wise cell."""
        directions, _ = encoding_grid(positions, cell, self.grid, self.weights)
        return rotary_encoding(values, positions, directions, self.frequencies)

    def prepare(self, query, key, positions, cell=None):
        """Prepare geometry once; retain autograd for every subsequent use."""
        directions, weights = encoding_grid(positions, cell, self.grid, self.weights)
        q = rotary_encoding(query, positions, directions, self.frequencies)
        q = q / math.sqrt(query.shape[1] * query.shape[2])
        k = rotary_encoding(key, positions, directions, self.frequencies)
        return PreparedAttention(q, k, weights)

    def _contraction_order(self, query, value):
        # Full-model checkpoints predating this option keep their exact old path.
        order = getattr(self, "contraction", "linear")
        limit = getattr(self, "pairwise_max_atoms", 256)
        n, directions, channels = query.shape
        if order == "pairwise" and n > limit:
            raise ValueError("pairwise atom count exceeds pairwise_max_atoms")
        if order != "auto":
            return order
        width = value.shape[-1]
        # Linear: 2*N*M*D*V; pairwise: N*N*(M*D+V) multiply-adds.
        use_pairs = (n <= limit and
                     n * (directions * channels + width)
                     < 2 * directions * channels * width)
        return "pairwise" if use_pairs else "linear"

    def aggregate(self, prepared, value):
        """Apply the same EFA kernel to any Value width without state caching."""
        q, k, weights = prepared
        if self._contraction_order(q, value) == "pairwise":
            # Integrate directions in this GEMM: never materialize [N, N, M].
            weighted_query = q * weights[None, :, None]
            kernel = weighted_query.flatten(1) @ k.flatten(1).T
            return kernel @ value
        kv = torch.einsum("nmd,nv->mdv", k, value)
        return torch.einsum("nmd,mdv,m->nv", q, kv, weights)

    def forward(self, query, key, value, positions, cell=None):
        return self.aggregate(self.prepare(query, key, positions, cell), value)

