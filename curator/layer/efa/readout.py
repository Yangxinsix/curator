"""Invariant energy readout with learned local/global tensor contractions."""

import torch
from e3nn import o3
from torch import nn


def invariant_features(values, irreps):
    """Keep 0e channels and form smooth squared norms of all other irreps."""
    parts = []
    for (multiplicity, irrep), section in zip(irreps, irreps.slices()):
        block = values[..., section].reshape(*values.shape[:-1], multiplicity, irrep.dim)
        parts.append(block[..., 0] if irrep == o3.Irrep("0e") else block.square().mean(-1))
    return torch.cat(parts, dim=-1)


def scalar_features(values, irreps):
    """Extract only even scalar channels, preserving their original order."""
    return torch.cat([values[..., section] for (_, irrep), section in zip(irreps, irreps.slices())
                      if irrep == o3.Irrep("0e")], dim=-1)


class InvariantEnergyReadout(nn.Module):
    """epsilon(z, g): direct scalars + learned tensor products + a smooth MLP.

    Independent equivariant channel mixtures of z and g are contracted into
    0e scalars. This includes local/global directional cross terms; taking only
    separate vector norms would discard their relative orientation. Compressing
    each irrep type before the tensor product limits its parameter count.
    """

    def __init__(self, local_irreps, global_irreps, hidden_features=64, coupling_features=16):
        super().__init__()
        if hidden_features < 1 or coupling_features < 1:
            raise ValueError("readout widths must be positive")
        self.local_irreps = o3.Irreps(local_irreps)
        self.global_irreps = o3.Irreps(global_irreps)
        self.input_irreps = self.local_irreps + self.global_irreps
        scalar_width = sum(mul for mul, ir in self.input_irreps if ir == o3.Irrep("0e"))
        if not scalar_width:
            raise ValueError("energy readout requires even scalar channels")
        types = sorted({ir for _, ir in self.input_irreps})
        coupling_irreps = o3.Irreps([(coupling_features, ir) for ir in types])
        self.mix_a = o3.Linear(self.input_irreps, coupling_irreps, biases=True)
        self.mix_b = o3.Linear(self.input_irreps, coupling_irreps, biases=True)
        self.product = o3.FullyConnectedTensorProduct(
            coupling_irreps, coupling_irreps, f"{hidden_features}x0e")
        self.mlp = nn.Sequential(
            nn.Linear(scalar_width + hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1, bias=False),
        )

    def forward(self, local, global_features):
        if local.ndim != 2 or local.shape[-1] != self.local_irreps.dim:
            raise ValueError("local features do not match readout irreps")
        if global_features.shape != (local.shape[0], self.global_irreps.dim):
            raise ValueError("global features do not match readout irreps")
        values = torch.cat((local, global_features), dim=-1)
        couplings = self.product(self.mix_a(values), self.mix_b(values))
        scalars = torch.cat((scalar_features(values, self.input_irreps), couplings), dim=-1)
        return self.mlp(scalars).squeeze(-1)


__all__ = ["InvariantEnergyReadout"]
