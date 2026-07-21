from typing import Optional

import torch
from torch import nn

from curator.data import properties

HuberLoss = nn.HuberLoss


class VectorL2MAELoss(nn.Module):
    """Mean L2 error over the final (Cartesian) dimension."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'.")
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.linalg.vector_norm(prediction - target, dim=-1)
        return loss if self.reduction == "none" else getattr(loss, self.reduction)()


class VectorHuberLoss(nn.Module):
    """Huber loss applied to the L2 error of each Cartesian vector."""

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        if delta <= 0:
            raise ValueError("delta must be positive.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'.")
        self.delta = float(delta)
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = torch.linalg.vector_norm(prediction - target, dim=-1)
        loss = torch.where(
            error <= self.delta,
            0.5 * error.square(),
            self.delta * (error - 0.5 * self.delta),
        )
        return loss if self.reduction == "none" else getattr(loss, self.reduction)()


def _per_observation(loss: torch.Tensor) -> torch.Tensor:
    if loss.dim() == 0:
        raise ValueError("Balanced losses require a nested loss with reduction='none'.")
    return loss.reshape(loss.shape[0], -1).mean(dim=1)


class StructureBalancedLoss(nn.Module):
    """Average atomwise errors within each structure, then across structures."""

    requires_batch = True

    def __init__(self, loss_fn: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.loss_fn = loss_fn or VectorL2MAELoss(reduction="none")

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        loss = _per_observation(self.loss_fn(prediction, target))
        n_atoms = batch[properties.n_atoms].reshape(-1).to(torch.long)
        if loss.shape[0] == n_atoms.shape[0]:
            return loss.mean()
        total_atoms = int(n_atoms.sum().item())
        if total_atoms == 0 or loss.shape[0] % total_atoms:
            raise ValueError("Loss observations do not align with batch n_atoms.")
        repeat = loss.shape[0] // total_atoms
        return torch.stack([
            part.mean() for part in torch.split(loss, (n_atoms * repeat).tolist())
        ]).mean()


class SpeciesBalancedLoss(nn.Module):
    """Average atomwise errors within each species, then across species."""

    requires_batch = True

    def __init__(self, loss_fn: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.loss_fn = loss_fn or VectorL2MAELoss(reduction="none")

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        loss = _per_observation(self.loss_fn(prediction, target))
        n_atoms = batch[properties.n_atoms].reshape(-1).to(torch.long)
        atomic_numbers = batch[properties.Z].reshape(-1)
        total_atoms = int(n_atoms.sum().item())
        if total_atoms == 0 or loss.shape[0] % total_atoms:
            raise ValueError("Loss observations do not align with batch atomic numbers.")
        repeat = loss.shape[0] // total_atoms
        if repeat > 1:
            atomic_numbers = torch.cat([
                values.repeat(repeat)
                for values in torch.split(atomic_numbers, n_atoms.tolist())
            ])
        return torch.stack([
            loss[atomic_numbers == species].mean()
            for species in torch.unique(atomic_numbers)
        ]).mean()
