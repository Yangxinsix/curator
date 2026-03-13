from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    from torch_scatter import scatter_add, scatter_mean
except ImportError:
    from curator.utils import scatter_add, scatter_mean

from .common import KernelName, Reduction


class DistanceMetrics:
    """Distance metrics over feature tensors."""

    def __init__(self, regularization: float = 1e-6, reduction: Optional[Reduction] = None) -> None:
        self.regularization = regularization
        self.reduction = reduction
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.precision: Optional[torch.Tensor] = None
        self.reference_distances: Optional[torch.Tensor] = None

    def fit(
        self,
        features: torch.Tensor,
        image_idx: Optional[torch.Tensor] = None,
        reduction: Optional[Reduction] = None,
    ) -> None:
        feats = self._prepare_features(features)
        mean = torch.mean(feats, dim=0)
        std = torch.std(feats, dim=0)
        std = torch.where(std == 0, torch.ones_like(std), std)
        norm = (feats - mean) / std
        denom = max(norm.shape[0] - 1, 1)
        covariance = norm.T @ norm / denom
        eye = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
        covariance = covariance + self.regularization * eye
        self.mean = mean
        self.std = std
        self.precision = torch.linalg.inv(covariance)
        dist_sq = torch.einsum("bi,ij,bj->b", norm, self.precision, norm)
        distances = torch.sqrt(torch.clamp(dist_sq, min=0.0))
        self.reference_distances = self._reduce(distances, image_idx, reduction)

    def fit_from_stats(
        self,
        stats,
        kernel: KernelName,
        image_idx: Optional[torch.Tensor] = None,
        reduction: Optional[Reduction] = None,
        streaming: bool = False,
    ) -> None:
        if streaming:
            self.fit_from_stats_streaming(stats, kernel, image_idx=image_idx, reduction=reduction)
            return
        features = stats.get_features(normalize=False)
        if kernel not in features:
            raise ValueError(f"Kernel '{kernel}' is not available in FeatureStatistics.")
        self.fit(features[kernel], image_idx=image_idx, reduction=reduction)

    def fit_from_stats_streaming(
        self,
        stats,
        kernel: KernelName,
        image_idx: Optional[torch.Tensor] = None,
        reduction: Optional[Reduction] = None,
    ) -> None:
        mean, std, precision = self._streaming_stats(stats, kernel)
        distances = self._streaming_distances(stats, kernel, mean, std, precision)
        self.mean = mean
        self.std = std
        self.precision = precision
        self.reference_distances = self._reduce(distances, image_idx, reduction)

    def _streaming_stats(self, stats, kernel: KernelName) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = 0
        sum_x = None
        sum_x2 = None
        sum_xxT = None
        for feats in stats.iter_kernel_features(kernel):
            feats = self._prepare_features(feats.detach().cpu())
            if feats.numel() == 0:
                continue
            batch_n = feats.shape[0]
            batch_sum = feats.sum(dim=0)
            batch_sum2 = feats.pow(2).sum(dim=0)
            batch_sum_xxT = feats.T @ feats
            if sum_x is None:
                sum_x = batch_sum
                sum_x2 = batch_sum2
                sum_xxT = batch_sum_xxT
            else:
                sum_x += batch_sum
                sum_x2 += batch_sum2
                sum_xxT += batch_sum_xxT
            count += batch_n
        if sum_x is None or sum_x2 is None or sum_xxT is None or count == 0:
            raise RuntimeError("No features available for streaming statistics.")
        denom = max(count - 1, 1)
        mean = sum_x / count
        var = (sum_x2 - count * mean.pow(2)) / denom
        std = torch.sqrt(var)
        std = torch.where(std == 0, torch.ones_like(std), std)
        covariance = (sum_xxT - count * torch.outer(mean, mean)) / denom
        eye = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
        covariance = covariance + self.regularization * eye
        precision = torch.linalg.inv(covariance)
        return mean, std, precision

    def _streaming_distances(
        self,
        stats,
        kernel: KernelName,
        mean: torch.Tensor,
        std: torch.Tensor,
        precision: torch.Tensor,
    ) -> torch.Tensor:
        distances = []
        for feats in stats.iter_kernel_features(kernel):
            feats = self._prepare_features(feats.detach().cpu())
            if feats.numel() == 0:
                continue
            norm = (feats - mean) / std
            dist_sq = torch.einsum("bi,ij,bj->b", norm, precision, norm)
            distances.append(torch.sqrt(torch.clamp(dist_sq, min=0.0)))
        if not distances:
            raise RuntimeError("No features available for streaming distances.")
        return torch.cat(distances, dim=0)

    def score(
        self,
        features: torch.Tensor,
        image_idx: Optional[torch.Tensor] = None,
        reduction: Optional[Reduction] = None,
    ) -> torch.Tensor:
        if self.mean is None or self.std is None or self.precision is None:
            raise RuntimeError("DistanceMetrics must be fit before scoring.")
        feats = self._prepare_features(features)
        norm = (feats - self.mean) / self.std
        dist_sq = torch.einsum("bi,ij,bj->b", norm, self.precision, norm)
        distances = torch.sqrt(torch.clamp(dist_sq, min=0.0))
        return self._reduce(distances, image_idx, reduction)

    def score_euclidean(
        self,
        features: torch.Tensor,
        image_idx: Optional[torch.Tensor] = None,
        reduction: Optional[Reduction] = None,
    ) -> torch.Tensor:
        distances = torch.norm(self._prepare_features(features), dim=1)
        return self._reduce(distances, image_idx, reduction)

    def score_cosine(
        self,
        features: torch.Tensor,
        image_idx: Optional[torch.Tensor] = None,
        reduction: Optional[Reduction] = None,
    ) -> torch.Tensor:
        feats = self._prepare_features(features)
        norms = torch.norm(feats, dim=1)
        distances = 1.0 - torch.sum(feats, dim=1) / torch.clamp(norms, min=1e-12)
        return self._reduce(distances, image_idx, reduction)

    def score_from_stats(
        self,
        stats,
        kernel: KernelName,
        image_idx: Optional[torch.Tensor] = None,
        reduction: Optional[Reduction] = None,
    ) -> torch.Tensor:
        features = stats.get_features(normalize=False)
        if kernel not in features:
            raise ValueError(f"Kernel '{kernel}' is not available in FeatureStatistics.")
        return self.score(features[kernel], image_idx=image_idx, reduction=reduction)

    @staticmethod
    def _prepare_features(features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3:
            features = features.mean(dim=0)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.dim() != 2:
            raise ValueError("Features must be 2D or 3D.")
        return features

    def _reduce(
        self,
        distances: torch.Tensor,
        image_idx: Optional[torch.Tensor],
        reduction: Optional[Reduction],
    ) -> torch.Tensor:
        reduction = reduction or self.reduction
        if reduction is None:
            return distances
        if image_idx is None:
            raise ValueError("image_idx is required for distance reduction.")
        if reduction == "mean":
            return scatter_mean(distances, image_idx, dim=0)
        if reduction == "sum":
            return scatter_add(distances, image_idx, dim=0)
        raise ValueError(f"Unsupported reduction '{reduction}'.")
