from __future__ import annotations

from typing import Union

import torch
from torch import nn

from .common import ExtractedFeatures, FeatureSpec, feature_spec_from_object
from .kme import (
    BaseKMEAggregator,
    IdentityKMEAggregator,
    RandomFourierKMEAggregator,
    SketchingKMEAggregator,
)


class FeatureKernel(nn.Module):
    """Parse a feature spec and compute the final feature representation."""

    def __init__(self, spec: Union[FeatureSpec, dict]) -> None:
        super().__init__()
        self.spec = feature_spec_from_object(spec)
        self.kernel = self.spec.kernel_name
        self.local = self.spec.local
        self.kme = self._build_kme(self.spec)

    def compute(self, extracted: ExtractedFeatures) -> torch.Tensor:
        raw_feature = self._resolve_raw_feature(extracted)
        atomic_features = self.kme.transform(raw_feature)
        if self.local:
            return atomic_features
        return self.kme.aggregate(atomic_features, extracted.image_idx)

    def _resolve_raw_feature(self, extracted: ExtractedFeatures):
        if self.spec.source == "full-gradient":
            if not extracted.grads:
                raise ValueError(
                    "full-gradient requires gradient hooks. "
                    "Use a linear target_layer such as 'readout_mlp'."
                )
            return extracted.feats, extracted.grads
        if self.spec.source == "ll-gradient":
            return extracted.feats[-1][:, :-1]
        if self.spec.source == "gnn":
            return extracted.feats[0][:, :-1]
        raise ValueError(f"Unsupported raw_feature '{self.spec.raw_feature}'.")

    @staticmethod
    def _build_kme(spec: FeatureSpec) -> BaseKMEAggregator:
        if spec.mapping == "gaussian_sketch":
            return SketchingKMEAggregator(
                num_features=spec.num_features,
                pooling=spec.pooling,
                layer_combine=spec.layer_combine,
                layer_norm=spec.layer_norm,
                seed=spec.seed,
            )
        if spec.mapping == "rff":
            return RandomFourierKMEAggregator(
                num_features=spec.num_features,
                pooling=spec.pooling,
                layer_combine=spec.layer_combine,
                layer_norm=spec.layer_norm,
                sigma=spec.sigma,
                seed=spec.seed,
            )
        return IdentityKMEAggregator(
            pooling=spec.pooling,
        )
