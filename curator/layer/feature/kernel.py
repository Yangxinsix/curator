from __future__ import annotations

import torch

try:
    from torch_scatter import scatter_add
except ImportError:
    from curator.utils import scatter_add

from .aggregation import FeatureAggregator, SumAggregator
from .common import ExtractedFeatures, KernelName, normalize_kernel
from .projector import FeatureProjector


class FeatureKernel:
    """Compute atom-level features and aggregate them into structure features."""

    def __init__(
        self,
        kernel: KernelName,
        projector: FeatureProjector,
        aggregator: FeatureAggregator | None = None,
    ) -> None:
        self.kernel = normalize_kernel(kernel)
        self.projector = projector
        self.aggregator = aggregator if aggregator is not None else SumAggregator()

    def compute(self, extracted: ExtractedFeatures) -> torch.Tensor:
        local = self.kernel.startswith("local_")
        kernel = self.kernel[len("local_") :] if local else self.kernel

        if kernel == "full-gradient":
            if self.projector.num_features == 0:
                raise ValueError("full-gradient requires random projections.")
            atomic = torch.zeros(
                (extracted.image_idx.shape[0], self.projector.num_features),
                device=extracted.image_idx.device,
            )
            for feat, grad, in_proj, out_proj in zip(
                extracted.feats,
                extracted.grads,
                self.projector.in_feat_proj,
                self.projector.out_grad_proj,
            ):
                atomic += (feat @ in_proj) * (grad @ out_proj)
        elif kernel == "ll-gradient":
            if self.projector.num_features != 0:
                atomic = (extracted.feats[-1] @ self.projector.in_feat_proj[-1]) * (
                    extracted.grads[-1] @ self.projector.out_grad_proj[-1]
                )
            else:
                atomic = extracted.feats[-1][:, :-1]
        elif kernel == "gnn":
            if self.projector.num_features != 0:
                atomic = (extracted.feats[0] @ self.projector.in_feat_proj[0]) * (
                    extracted.grads[0] @ self.projector.out_grad_proj[0]
                )
            else:
                atomic = extracted.feats[0][:, :-1]
        else:
            raise RuntimeError(f"Unknown kernel '{self.kernel}'")

        if local:
            return atomic
        return self.aggregator.aggregate(atomic, extracted)
