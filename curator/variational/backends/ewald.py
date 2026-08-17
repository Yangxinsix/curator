"""Adapter exposing Curator's differentiable Ewald kernel to ACVR."""

from typing import Dict, List

import torch

from curator.data import properties
from curator.layer._ewald import EwaldSummation


class EwaldChargeOperator:
    """Per-structure quadratic charge operator backed by ``EwaldSummation``."""

    def __init__(self, ewald: EwaldSummation):
        self.ewald = ewald

    def kernels(self, data: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        kernels = self.ewald.get_ewald_kernel(
            data[properties.cell],
            data[properties.n_atoms],
            data[properties.positions],
            data[properties.edge_dist],
            data[properties.edge_idx],
        )
        return [0.5 * (kernel + kernel.transpose(0, 1)) for kernel in kernels]

    def energy(self, data: Dict[str, torch.Tensor], charge: torch.Tensor) -> torch.Tensor:
        updated = data.copy()
        updated[properties.atomic_charge] = charge
        return self.ewald(updated)

    def matvec(self, data: Dict[str, torch.Tensor], charge: torch.Tensor) -> torch.Tensor:
        counts = data[properties.n_atoms]
        if int(counts.sum().item()) != charge.numel():
            raise ValueError("charge length does not match n_atoms")
        products = []
        offset = 0
        for count, kernel in zip(counts.tolist(), self.kernels(data)):
            products.append(kernel @ charge[offset : offset + count])
            offset += count
        return torch.cat(products) if products else charge.new_zeros((0,))
