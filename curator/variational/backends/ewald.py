"""Adapter exposing Curator's differentiable Ewald kernel to ACVR."""

from typing import Dict, List

import torch

from curator.data import properties
from curator.layer._ewald import EwaldSummation

from ..operators import DenseQuadraticOperator


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
        if counts.ndim != 1:
            raise ValueError("n_atoms must be a rank-1 tensor")
        if int(counts.sum().item()) != charge.numel():
            raise ValueError("charge length does not match n_atoms")
        kernels = self.kernels(data)
        if len(kernels) != counts.numel():
            raise ValueError("Ewald kernel count does not match n_atoms")
        products = []
        offset = 0
        for count, kernel in zip(counts.tolist(), kernels):
            if kernel.shape != (count, count):
                raise ValueError("Ewald kernel shape does not match n_atoms")
            if kernel.dtype != charge.dtype or kernel.device != charge.device:
                raise ValueError("Ewald kernels must match charge dtype and device")
            products.append(kernel @ charge[offset : offset + count])
            offset += count
        if offset != charge.numel():
            raise ValueError("Ewald batch metadata did not consume every charge")
        return torch.cat(products) if products else charge.new_zeros((0,))

    def bind(
        self,
        data: Dict[str, torch.Tensor],
        structure_index: int = 0,
    ) -> DenseQuadraticOperator:
        """Bind one reference structure to the state-only operator protocol.

        The Torch reference backend materializes its small dense Ewald kernel.
        Production LAMMPS/PPPM bindings will implement the same ``energy`` and
        ``matvec`` protocol without materializing that kernel.
        """

        kernels = self.kernels(data)
        if structure_index < 0:
            structure_index += len(kernels)
        if structure_index < 0 or structure_index >= len(kernels):
            raise IndexError(structure_index)
        return DenseQuadraticOperator(kernels[structure_index])
