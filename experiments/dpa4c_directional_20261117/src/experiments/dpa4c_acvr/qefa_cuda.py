"""CUDA ERoPE and factorized communication for nonperiodic Small+ACVR."""

from dataclasses import dataclass
from math import sqrt

import torch
from curator.layer.efa.core import EuclideanFastAttention
from curator.layer.efa.reciprocal_rsa import PeriodicRSA


@dataclass
class MolecularFactors:
    """Batched factors, with quadrature weights already included in the query."""

    is_molecular_factorization = True
    q: torch.Tensor
    k: torch.Tensor

    def apply_batch(self, values, transpose=False):
        left, right = (self.k, self.q) if transpose else (self.q, self.k)
        return left @ (right.mT @ values)

    def apply(self, values, transpose=False):
        return self.apply_batch(values.unsqueeze(0), transpose).squeeze(0)


class MolecularKernels:
    def supports_fused(self, query, cell=None):
        return (
            cell is None
            and query.is_cuda
            and query.dtype in (torch.float32, torch.float64)
            and query.shape[-1] <= 128
            and self.grid.shape[0] <= 256
            and query.shape[-2] <= 16
            and query.numel() > 0
            and self.grid.numel() > 0
            and not self.weights.requires_grad
        )

    def prepare_batch(self, query, key, positions):
        from rotary_cuda import rotated_factors

        batch, atoms, _ = positions.shape
        xyz = positions - positions.mean(dim=1, keepdim=True)
        phase = (xyz @ self.grid.to(xyz).T)[..., None] * self.frequencies.to(xyz)
        weights = self.weights.to(xyz) / sqrt(query.shape[-2] * query.shape[-1])
        q, k = rotated_factors(query, key, phase.flatten(0, 1), weights)
        return MolecularFactors(
            q.reshape(batch, atoms, -1), k.reshape(batch, atoms, -1)
        )

    def prepare(self, query, key, positions, cell=None):
        if self.supports_fused(query, cell):
            return self.prepare_batch(query, key, positions.unsqueeze(0))
        return super().prepare(query, key, positions, cell)

    def aggregate(self, prepared, value):
        if isinstance(prepared, MolecularFactors):
            return prepared.apply(value)
        return super().aggregate(prepared, value)


class MolecularFastAttention(MolecularKernels, EuclideanFastAttention):
    pass


class MolecularPeriodicRSA(MolecularKernels, PeriodicRSA):
    """Use the standard RSA implementation when a periodic cell is supplied."""
