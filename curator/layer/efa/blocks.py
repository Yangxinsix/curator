"""Equivariant EFA feature updates with optional directional tensor integration."""
import math

import torch
from torch import nn
from e3nn import o3

from .core import EuclideanFastAttention
from .geometry import encoding_grid, rotary_encoding


class EquivariantEFA(nn.Module):
    """Map state-conditioned irreps features to global equivariant features.

    All three learned projections read ``features``. Queries/keys use the same
    even channel count for every input irrep type, packed as [N, sum(2l+1), H].
    All values and outputs use flat e3nn ordering. With tensor integration,
    values retain ``input_irreps`` and the final tensor product maps them to
    ``output_irreps``. Without it, the value projection directly uses
    ``output_irreps``.

    With tensor integration, each direction-resolved message is coupled to
    spherical harmonics of that direction before quadrature. This can create
    directional output types that a linear value projection alone cannot.
    The largest attention intermediates scale linearly with atom count; no
    atom-by-atom attention matrix is constructed.

    Finite spherical quadrature gives approximate O(3) covariance. The legacy
    raw-lattice branch is available only without tensor integration; periodic
    directional tensor integration is not implemented here.
    """

    def __init__(self, input_irreps, output_irreps, hidden_features=32,
                 num_directions=110, max_frequency=2 * math.pi, max_length=32.0,
                 tensor_integration=True, integration_lmax=2):
        super().__init__()
        self.input_irreps = o3.Irreps(input_irreps)
        self.output_irreps = o3.Irreps(output_irreps)
        if not self.input_irreps.dim or not self.output_irreps.dim:
            raise ValueError("EFA input and output irreps must be nonempty")
        if not isinstance(integration_lmax, int) or integration_lmax < 0:
            raise ValueError("integration_lmax must be a nonnegative integer")
        if tensor_integration and num_directions is None:
            raise ValueError("tensor integration requires spherical num_directions")
        self.hidden_features = hidden_features
        self.tensor_integration = bool(tensor_integration)
        self.integration_lmax = integration_lmax
        types = sorted({ir for _, ir in self.input_irreps})
        self.query_irreps = o3.Irreps([(hidden_features, ir) for ir in types])
        # Retain source types until their directional coupling. Projecting V
        # straight to output types would erase e.g. vector -> scalar TP paths.
        self.value_irreps = self.input_irreps if self.tensor_integration else self.output_irreps
        self.attention = EuclideanFastAttention(hidden_features, num_directions,
                                               max_frequency, max_length)
        self.query = o3.Linear(self.input_irreps, self.query_irreps, biases=True)
        self.key = o3.Linear(self.input_irreps, self.query_irreps, biases=True)
        self.value = o3.Linear(self.input_irreps, self.value_irreps, biases=True)
        self.direction_irreps = o3.Irreps.spherical_harmonics(integration_lmax)
        self.tensor_product = (
            o3.FullyConnectedTensorProduct(self.value_irreps, self.direction_irreps,
                                           self.output_irreps)
            if self.tensor_integration else None
        )

    def _angular_channels(self, features):
        return torch.cat([
            features[:, sl].reshape(len(features), self.hidden_features, ir.dim)
            .transpose(1, 2)
            for (_, ir), sl in zip(self.query_irreps, self.query_irreps.slices())
        ], dim=1)

    def forward(self, features, positions, cell=None):
        if features.ndim != 2 or features.shape[1] != self.input_irreps.dim:
            raise ValueError("features must have shape (n_atoms, input_irreps.dim)")
        if positions.shape != (len(features), 3) or len(features) == 0:
            raise ValueError("positions must describe the same nonempty atom set")
        if self.tensor_integration and cell is not None:
            raise NotImplementedError("directional tensor-integration EFA requires nonperiodic geometry")
        query = self._angular_channels(self.query(features))
        key = self._angular_channels(self.key(features))
        value = self.value(features)
        if not self.tensor_integration:
            return self.attention(query, key, value, positions, cell=cell)

        directions, weights = encoding_grid(positions, None, self.attention.grid,
                                             self.attention.weights)
        q = rotary_encoding(query, positions, directions, self.attention.frequencies)
        q = q / math.sqrt(query.shape[1] * query.shape[2])
        k = rotary_encoding(key, positions, directions, self.attention.frequencies)
        kv = torch.einsum("nmd,nv->mdv", k, value)
        directional = torch.einsum("nmd,mdv->nmv", q, kv)
        harmonics = o3.spherical_harmonics(self.direction_irreps, directions,
                                          normalize=True, normalization="component")
        coupled = self.tensor_product(directional,
                                      harmonics.unsqueeze(0).expand(len(features), -1, -1))
        return torch.einsum("nmv,m->nv", coupled, weights)
