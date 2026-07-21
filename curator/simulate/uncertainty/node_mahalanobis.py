from __future__ import annotations

from typing import Dict, Optional

import torch

from curator.data import properties
from curator.data._uncertainty import UncertaintyModule


class NodeFeatureMahalanobis(UncertaintyModule):
    """Scriptable Mahalanobis scorer based on representation node features.

    This module is intended for deployment formats that cannot carry Python
    forward/backward hooks, such as TorchScript pair_style curator models.
    It consumes ``properties.node_feat`` produced by the representation.
    """

    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        precision: torch.Tensor,
        reference_distances: torch.Tensor,
        *,
        local: bool = True,
        output_per_atom: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("feature_mean", feature_mean.detach().clone())
        self.register_buffer("feature_std", feature_std.detach().clone().clamp_min(1e-12))
        self.register_buffer("precision", precision.detach().clone())
        self.register_buffer("maha_dist", reference_distances.detach().clone())
        self.local = bool(local)
        self.output_per_atom = bool(output_per_atom)
        self.model_outputs = [properties.maha_dist]
        per_atom_keys = []
        if self.local and self.output_per_atom:
            self.model_outputs.append(properties.maha_dist_per_atom)
            per_atom_keys.append(properties.maha_dist_per_atom)
        self.set_uncertainty_outputs(
            scalar_keys=[properties.maha_dist],
            per_atom_keys=per_atom_keys,
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        node_features = data[properties.node_feat]
        image_idx = data[properties.image_idx].to(dtype=torch.long)
        if self.local:
            per_atom = self._distance(node_features)
            data[properties.maha_dist] = self._scatter_mean_1d(per_atom, image_idx)
            if self.output_per_atom:
                data[properties.maha_dist_per_atom] = per_atom
            return data

        pooled = self._scatter_mean_2d(node_features, image_idx)
        data[properties.maha_dist] = self._distance(pooled)
        return data

    def _distance(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std
        projected = torch.matmul(normalized, self.precision)
        dist_sq = torch.sum(projected * normalized, dim=-1)
        return torch.sqrt(torch.clamp(dist_sq, min=0.0))

    @staticmethod
    def _scatter_mean_1d(values: torch.Tensor, image_idx: torch.Tensor) -> torch.Tensor:
        n_images = int(torch.max(image_idx).item()) + 1
        out = torch.zeros((n_images,), dtype=values.dtype, device=values.device)
        count = torch.zeros((n_images,), dtype=values.dtype, device=values.device)
        out.scatter_add_(0, image_idx, values)
        count.scatter_add_(0, image_idx, torch.ones_like(values))
        return out / torch.clamp(count, min=1.0)

    @staticmethod
    def _scatter_mean_2d(values: torch.Tensor, image_idx: torch.Tensor) -> torch.Tensor:
        n_images = int(torch.max(image_idx).item()) + 1
        out = torch.zeros((n_images, values.shape[-1]), dtype=values.dtype, device=values.device)
        count = torch.zeros((n_images, 1), dtype=values.dtype, device=values.device)
        expanded = image_idx.view(-1, 1).expand(-1, values.shape[-1])
        out.scatter_add_(0, expanded, values)
        count.scatter_add_(0, image_idx.view(-1, 1), torch.ones((values.shape[0], 1), dtype=values.dtype, device=values.device))
        return out / torch.clamp(count, min=1.0)


def fit_node_feature_mahalanobis(
    model: torch.nn.Module,
    dataset: str,
    *,
    local: bool,
    output_per_atom: bool,
    max_structures: Optional[int] = None,
    regularization: float = 1e-6,
) -> NodeFeatureMahalanobis:
    """Fit a scriptable Mahalanobis scorer from frozen representation features."""

    from curator.data import AseDataset
    from curator.layer.utils import find_layer_by_name_recursive

    device = next(model.parameters()).device
    cutoff = find_layer_by_name_recursive(model, "cutoff") or 5.0
    data = AseDataset(dataset, cutoff=cutoff)
    limit = len(data) if max_structures is None else min(int(max_structures), len(data))

    feature_blocks = []
    image_blocks = []
    with torch.no_grad():
        for index in range(limit):
            batch = dict(data[index])
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            for module in model.input_modules:
                batch = module(batch)
            batch = model.representation(batch)
            features = batch[properties.node_feat].detach()
            if local:
                feature_blocks.append(features)
                image_blocks.append(torch.full((features.shape[0],), index, dtype=torch.long, device=device))
            else:
                image_idx = batch[properties.image_idx].to(dtype=torch.long)
                pooled = NodeFeatureMahalanobis._scatter_mean_2d(features, image_idx)
                feature_blocks.append(pooled)
                image_blocks.append(torch.arange(pooled.shape[0], dtype=torch.long, device=device) + len(image_blocks))

    if not feature_blocks:
        raise ValueError("Reference dataset is empty; cannot fit Mahalanobis statistics.")

    features = torch.cat(feature_blocks, dim=0)
    image_idx = torch.cat(image_blocks, dim=0)
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-12)
    normalized = (features - mean) / std
    denom = max(1, normalized.shape[0] - 1)
    cov = torch.matmul(normalized.T, normalized) / float(denom)
    cov = cov + torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device) * float(regularization)
    precision = torch.linalg.pinv(cov)
    projected = torch.matmul(normalized, precision)
    row_dist = torch.sqrt(torch.clamp(torch.sum(projected * normalized, dim=-1), min=0.0))
    reference_distances = (
        NodeFeatureMahalanobis._scatter_mean_1d(row_dist, image_idx)
        if local
        else row_dist
    )
    return NodeFeatureMahalanobis(
        mean,
        std,
        precision,
        reference_distances,
        local=local,
        output_per_atom=output_per_atom,
    )
