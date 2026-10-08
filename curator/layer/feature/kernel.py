from __future__ import annotations

from typing import Union

import torch
from torch import nn

from .common import ExtractedFeatures, FeatureSpec, feature_spec_from_object
from .kme import (
    BaseKMEAggregator,
    IdentityKMEAggregator,
    NystromKMEAggregator,
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
        if self.spec.mapping == "nystrom" and self.spec.source == "full-gradient":
            expected_layouts = self.kme.state.metadata.get("readout_layouts")
            if expected_layouts is not None and expected_layouts != extracted.readout_layouts:
                raise ValueError("Nyström state readout_layouts differ from the current extracted readout.")
        raw_feature = self._resolve_raw_feature(extracted)
        if self.local:
            return self.kme.transform(raw_feature)
        return self.kme.structure_features(raw_feature, extracted.image_idx)

    def _resolve_raw_feature(self, extracted: ExtractedFeatures):
        if self.spec.source == "full-gradient":
            if not extracted.grads:
                raise ValueError(
                    "full-gradient requires gradient hooks. "
                    "Use a linear target_layer such as 'readout_mlp'."
                )
            if self.spec.layer_combine == "joint":
                from .readout import parameter_gradient_factors

                pairs = parameter_gradient_factors(
                    extracted.feats, extracted.grads, extracted.readout_layouts
                )
                return [a for a, _ in pairs], [b for _, b in pairs]
            return extracted.feats, extracted.grads
        if self.spec.source == "ll-gradient":
            return extracted.feats[-1][:, :-1]
        if self.spec.source == "gnn":
            return extracted.feats[0][:, :-1]
        raise ValueError(f"Unsupported raw_feature '{self.spec.raw_feature}'.")

    @staticmethod
    def _build_kme(spec: FeatureSpec) -> BaseKMEAggregator:
        if spec.mapping == "nystrom":
            return NystromKMEAggregator(
                state_path=spec.nystrom_state,
                num_features=spec.num_features,
                sigma=spec.sigma,
                pooling=spec.pooling,
                layer_combine=spec.layer_combine,
                raw_feature=spec.source,
            )
        if spec.mapping == "gaussian_sketch":
            return SketchingKMEAggregator(
                num_features=spec.num_features,
                pooling=spec.pooling,
                layer_combine=spec.layer_combine,
                layer_norm=spec.layer_norm,
                seed=spec.seed,
                projection_cache_bytes=spec.projection_cache_bytes,
            )
        if spec.mapping == "rff":
            return RandomFourierKMEAggregator(
                num_features=spec.num_features,
                pooling=spec.pooling,
                layer_combine=spec.layer_combine,
                layer_norm=spec.layer_norm,
                sigma=spec.sigma,
                rff_kernel=spec.rff_kernel,
                seed=spec.seed,
                projection_cache_bytes=spec.projection_cache_bytes,
            )
        return IdentityKMEAggregator(
            pooling=spec.pooling,
        )
