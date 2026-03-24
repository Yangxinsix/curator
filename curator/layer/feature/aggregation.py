from __future__ import annotations

import torch
from torch import nn

try:
    from torch_scatter import scatter_add, scatter_mean
except ImportError:
    from curator.utils import scatter_add, scatter_mean

from .common import ExtractedFeatures


class FeatureAggregator(nn.Module):
    """Aggregate atom-level features into structure-level features."""

    def aggregate(self, atomic_features: torch.Tensor, extracted: ExtractedFeatures) -> torch.Tensor:
        raise NotImplementedError


class SumAggregator(FeatureAggregator):
    """Structure embedding via sum pooling."""

    def aggregate(self, atomic_features: torch.Tensor, extracted: ExtractedFeatures) -> torch.Tensor:
        return scatter_add(atomic_features, extracted.image_idx, dim=0)


class MeanAggregator(FeatureAggregator):
    """Structure embedding via mean pooling."""

    def aggregate(self, atomic_features: torch.Tensor, extracted: ExtractedFeatures) -> torch.Tensor:
        return scatter_mean(atomic_features, extracted.image_idx, dim=0)
