from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn

try:
    from torch_scatter import scatter_add, scatter_mean
except ImportError:
    from curator.utils import scatter_add, scatter_mean

from .common import ExtractedFeatures, Reduction


class FeatureAggregator(nn.Module):
    """Aggregate atom-level features into structure-level features."""

    def aggregate(self, atomic_features: torch.Tensor, extracted: ExtractedFeatures) -> torch.Tensor:
        raise NotImplementedError


class SumAggregator(FeatureAggregator):
    def aggregate(self, atomic_features: torch.Tensor, extracted: ExtractedFeatures) -> torch.Tensor:
        return scatter_add(atomic_features, extracted.image_idx, dim=0)


class MeanAggregator(FeatureAggregator):
    def aggregate(self, atomic_features: torch.Tensor, extracted: ExtractedFeatures) -> torch.Tensor:
        return scatter_mean(atomic_features, extracted.image_idx, dim=0)


class KMEAggregator(FeatureAggregator):
    """Kernel mean embedding style aggregation with optional species-wise blocks."""

    def __init__(
        self,
        pooling: Reduction = "sum",
        kernel: str = "linear",
        species_wise: bool = False,
        species: Optional[Sequence[int]] = None,
        num_rff_features: int = 256,
        sigma: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if pooling not in {"sum", "mean"}:
            raise ValueError(f"Unsupported pooling '{pooling}'.")
        if kernel not in {"linear", "rbf"}:
            raise ValueError(f"Unsupported KME kernel '{kernel}'.")
        if kernel == "rbf" and num_rff_features <= 0:
            raise ValueError("num_rff_features must be positive for rbf KME.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        if species_wise and not species:
            raise ValueError("species_wise KME requires an explicit species list.")
        self.pooling = pooling
        self.kernel = kernel
        self.species_wise = species_wise
        self.species = tuple(int(s) for s in species) if species is not None else None
        self.num_rff_features = int(num_rff_features)
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.register_buffer("_rff_weight", torch.empty(0, 0))
        self.register_buffer("_rff_bias", torch.empty(0))
        self._input_dim: Optional[int] = None

    def aggregate(self, atomic_features: torch.Tensor, extracted: ExtractedFeatures) -> torch.Tensor:
        transformed = self._transform(atomic_features)
        num_structures = self._num_structures(extracted)
        if not self.species_wise:
            return self._reduce(transformed, extracted.image_idx, num_structures)

        if extracted.atomic_numbers is None:
            raise ValueError("species_wise KME requires atomic_numbers in extracted features.")

        blocks = []
        for species in self.species or ():
            mask = extracted.atomic_numbers == species
            if mask.any():
                block = self._reduce(
                    transformed[mask],
                    extracted.image_idx[mask],
                    num_structures,
                )
            else:
                block = transformed.new_zeros((num_structures, transformed.shape[1]))
            blocks.append(block)
        return torch.cat(blocks, dim=1) if blocks else transformed.new_zeros((num_structures, 0))

    def _transform(self, atomic_features: torch.Tensor) -> torch.Tensor:
        if self.kernel == "linear":
            return atomic_features
        self._ensure_rff_parameters(
            input_dim=atomic_features.shape[1],
            device=atomic_features.device,
            dtype=atomic_features.dtype,
        )
        projected = atomic_features @ self._rff_weight + self._rff_bias
        scale = math.sqrt(2.0 / self.num_rff_features)
        return scale * torch.cos(projected)

    def _ensure_rff_parameters(
        self,
        input_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if (
            self._input_dim == input_dim
            and self._rff_weight.numel() > 0
            and self._rff_weight.device == device
            and self._rff_weight.dtype == dtype
        ):
            return
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        weight = torch.randn(input_dim, self.num_rff_features, generator=generator, dtype=dtype)
        weight = weight / self.sigma
        bias = 2.0 * math.pi * torch.rand(self.num_rff_features, generator=generator, dtype=dtype)
        self._rff_weight = weight.to(device=device, dtype=dtype)
        self._rff_bias = bias.to(device=device, dtype=dtype)
        self._input_dim = input_dim

    def _reduce(
        self,
        transformed: torch.Tensor,
        image_idx: torch.Tensor,
        num_structures: int,
    ) -> torch.Tensor:
        out = transformed.new_zeros((num_structures, transformed.shape[1]))
        reduced = scatter_add(transformed, image_idx, dim=0, out=out)
        if self.pooling == "sum":
            return reduced
        count_out = transformed.new_zeros((num_structures, 1))
        counts = scatter_add(
            torch.ones((image_idx.shape[0], 1), device=image_idx.device, dtype=transformed.dtype),
            image_idx,
            dim=0,
            out=count_out,
        )
        return reduced / torch.clamp(counts, min=1.0)

    @staticmethod
    def _num_structures(extracted: ExtractedFeatures) -> int:
        if extracted.num_atoms is not None:
            return int(extracted.num_atoms.shape[0])
        if extracted.image_idx.numel() == 0:
            return 0
        return int(extracted.image_idx.max().item()) + 1
