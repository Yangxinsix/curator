from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Optional

import torch

try:
    from torch_scatter import scatter_add
except ImportError:
    from curator.utils import scatter_add

from .aggregation import FeatureAggregator
from .gaussian import BlockGaussianProjection

_RMS_EPS = 1e-8
_SKETCH_FEAT_SEED_STEP = 13
_SKETCH_GRAD_SEED_STEP = 17
_SKETCH_SIMPLE_SEED_STEP = 19
_RFF_FEAT_SEED_STEP = 29
_RFF_GRAD_SEED_STEP = 31
_RFF_SIMPLE_SEED_STEP = 37


def _num_structures(image_idx: torch.Tensor) -> int:
    if image_idx.numel() == 0:
        return 0
    return int(image_idx.max().item()) + 1


class BaseKMEAggregator(FeatureAggregator):
    """Transform raw layer features once, then aggregate them."""

    def __init__(
        self,
        num_features: int,
        pooling: str = "sum",
        layer_combine: str = "concat",
        layer_norm: str = "none",
    ) -> None:
        super().__init__()
        if pooling not in {"sum", "mean"}:
            raise ValueError(f"Unsupported pooling '{pooling}'.")
        if layer_combine not in {"concat", "sum", "joint"}:
            raise ValueError(f"Unsupported layer_combine '{layer_combine}'.")
        if layer_norm not in {"none", "rms"}:
            raise ValueError(f"Unsupported layer_norm '{layer_norm}'.")
        if layer_combine == "joint" and layer_norm != "none":
            raise ValueError("joint projection requires layer_norm='none'.")
        self.num_features = int(num_features)
        self.pooling = pooling
        self.layer_combine = layer_combine
        self.layer_norm = layer_norm

    def transform(self, raw_feature) -> torch.Tensor:
        if isinstance(raw_feature, tuple):
            feats, grads = raw_feature
            return self._transform_full_gradient(feats, grads)
        return self.transform_simple(raw_feature, 0)

    def aggregate(self, atomic_features: torch.Tensor, image_idx: torch.Tensor) -> torch.Tensor:
        return self.reduce(atomic_features, image_idx)

    def structure_features(self, raw_feature, image_idx: torch.Tensor) -> torch.Tensor:
        return self.aggregate(self.transform(raw_feature), image_idx)

    def reduce(self, transformed: torch.Tensor, image_idx: torch.Tensor) -> torch.Tensor:
        num_structures = _num_structures(image_idx)
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

    def _transform_full_gradient(self, feats: list[torch.Tensor], grads: list[torch.Tensor]) -> torch.Tensor:
        if len(feats) != len(grads):
            raise ValueError("Readout feature and gradient layer counts differ.")
        if self.layer_combine == "joint":
            return self.transform_joint(list(zip(feats, grads)))
        blocks = []
        for idx, (feat, grad) in enumerate(zip(feats, grads)):
            block = self.transform_pair(feat, grad, idx)
            blocks.append(self._maybe_rms_norm(block))
        if not blocks:
            return torch.zeros((0, 0))
        if self.layer_combine == "sum":
            return torch.stack(blocks, dim=0).sum(dim=0)
        return torch.cat(blocks, dim=1)

    def transform_joint(self, pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        raise NotImplementedError

    def transform_pair(self, feat: torch.Tensor, grad: torch.Tensor, layer_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def transform_simple(self, atomic_features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def _maybe_rms_norm(self, block: torch.Tensor) -> torch.Tensor:
        if self.layer_norm != "rms":
            return block
        denom = torch.sqrt(torch.mean(block * block, dim=1, keepdim=True) + _RMS_EPS)
        return block / denom


class IdentityKMEAggregator(BaseKMEAggregator):
    def __init__(self, pooling: str = "sum") -> None:
        super().__init__(
            num_features=0,
            pooling=pooling,
        )

    def transform_pair(self, feat: torch.Tensor, grad: torch.Tensor, layer_idx: int) -> torch.Tensor:
        raise ValueError("IdentityKMEAggregator does not support full-gradient features.")

    def transform_simple(self, atomic_features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return atomic_features


class NystromKMEAggregator(BaseKMEAggregator):
    """Use one frozen Nyström map for every atom and structure in the dataset."""

    def __init__(
        self,
        state_path: str,
        num_features: int,
        sigma: float,
        pooling: str = "mean",
        layer_combine: str = "concat",
        raw_feature: Optional[str] = None,
    ) -> None:
        from .nystrom import NystromMap

        super().__init__(num_features=num_features, pooling=pooling, layer_combine=layer_combine)
        self.state = NystromMap.load(state_path)
        self.state_file_sha256 = hashlib.sha256(Path(state_path).read_bytes()).hexdigest()
        if self.state.num_features != num_features:
            raise ValueError(
                f"Nyström state width {self.state.num_features} differs from num_features={num_features}."
            )
        if not math.isclose(self.state.sigma, sigma, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f"Nyström state sigma={self.state.sigma} differs from feature spec sigma={sigma}.")
        state_source = self.state.metadata.get("raw_feature")
        if state_source is not None and state_source != raw_feature:
            raise ValueError(f"Nyström state raw_feature={state_source!r} differs from {raw_feature!r}.")

    @staticmethod
    def _map_input(raw_feature):
        if isinstance(raw_feature, tuple):
            feats, grads = raw_feature
            if len(feats) != len(grads):
                raise ValueError("Readout feature and gradient layer counts differ.")
            return list(zip(feats, grads))
        return raw_feature

    def _prepare_input(self, raw_feature):
        raw = self._map_input(raw_feature)
        device = raw.device if isinstance(raw, torch.Tensor) else raw[0][0].device
        self.state.to(device=device)
        return raw

    def transform(self, raw_feature) -> torch.Tensor:
        return self.state.transform(self._prepare_input(raw_feature))

    def structure_features(self, raw_feature, image_idx: torch.Tensor) -> torch.Tensor:
        # The Gaussian kernel is applied per atom; only its final linear
        # correction commutes with pooling. This avoids per-atom whitening.
        return self.state.structure_features(self._prepare_input(raw_feature), image_idx, pooling=self.pooling)


class SketchingKMEAggregator(BaseKMEAggregator):
    def __init__(
        self,
        num_features: int,
        pooling: str = "sum",
        layer_combine: str = "concat",
        layer_norm: str = "none",
        seed: int = 0,
        projection_cache_bytes: int = 64 * 1024**2,
    ) -> None:
        if num_features <= 0:
            raise ValueError("num_features must be positive for sketching KME.")
        super().__init__(
            num_features=num_features,
            pooling=pooling,
            layer_combine=layer_combine,
            layer_norm=layer_norm,
        )
        self.seed = int(seed)
        self.projection = BlockGaussianProjection(num_features, self.seed, cache_bytes=projection_cache_bytes)
        self._feat_proj_buffers: list[str] = []
        self._grad_proj_buffers: list[str] = []
        self._simple_proj_buffers: list[str] = []

    def transform_joint(self, pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        return self.finish_projection(self.projection.project_pairs(pairs))

    def finish_projection(self, projected: torch.Tensor) -> torch.Tensor:
        """Finish a shared Gaussian projection of all readout coordinates."""
        return projected / math.sqrt(self.num_features)

    def transform_pair(self, feat: torch.Tensor, grad: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self.layer_combine == "joint":
            raise ValueError("joint projection requires all readout layers together; use transform().")
        feat_proj = self._ensure_projection(
            buffer_names=self._feat_proj_buffers,
            block=feat,
            layer_idx=layer_idx,
            prefix="feat_proj",
            seed=self.seed + _SKETCH_FEAT_SEED_STEP * layer_idx,
        )
        grad_proj = self._ensure_projection(
            buffer_names=self._grad_proj_buffers,
            block=grad,
            layer_idx=layer_idx,
            prefix="grad_proj",
            seed=self.seed + _SKETCH_GRAD_SEED_STEP * layer_idx + 1,
        )
        return ((feat @ feat_proj) * (grad @ grad_proj)) / math.sqrt(self.num_features)

    def transform_simple(self, atomic_features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        proj = self._ensure_projection(
            buffer_names=self._simple_proj_buffers,
            block=atomic_features,
            layer_idx=layer_idx,
            prefix="simple_proj",
            seed=self.seed + _SKETCH_SIMPLE_SEED_STEP * layer_idx + 3,
        )
        return (atomic_features @ proj) / math.sqrt(self.num_features)

    def _ensure_projection(
        self,
        buffer_names: list[str],
        block: torch.Tensor,
        layer_idx: int,
        prefix: str,
        seed: int,
    ) -> torch.Tensor:
        name = self._ensure_buffer_name(buffer_names, layer_idx, prefix)
        self._set_buffer_tensor(
            name,
            self._randn_tensor(block.shape[1], self.num_features, block.device, block.dtype, seed),
        )
        return getattr(self, name)

    @staticmethod
    def _ensure_buffer_name(buffer_names: list[str], idx: int, prefix: str) -> str:
        while len(buffer_names) <= idx:
            buffer_names.append(f"{prefix}_{len(buffer_names)}")
        return buffer_names[idx]

    def _set_buffer_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if hasattr(self, name):
            setattr(self, name, tensor)
        else:
            self.register_buffer(name, tensor)

    @staticmethod
    def _randn_tensor(
        rows: int,
        cols: int,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
    ) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.randn(rows, cols, generator=generator, dtype=dtype).to(device=device, dtype=dtype)


class RandomFourierKMEAggregator(BaseKMEAggregator):
    def __init__(
        self,
        num_features: int,
        pooling: str = "sum",
        layer_combine: str = "concat",
        layer_norm: str = "none",
        sigma: float = 1.0,
        seed: int = 0,
        rff_kernel: str = "rbf",
        projection_cache_bytes: int = 64 * 1024**2,
    ) -> None:
        if num_features <= 0:
            raise ValueError("num_features must be positive for RFF KME.")
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive.")
        if rff_kernel not in {"rbf", "matern32", "matern52", "laplacian_l1"}:
            raise ValueError(f"Unsupported RFF kernel '{rff_kernel}'.")
        if layer_combine == "joint" and rff_kernel != "rbf":
            raise ValueError("joint readout RFF supports only rff_kernel='rbf'.")
        super().__init__(
            num_features=num_features,
            pooling=pooling,
            layer_combine=layer_combine,
            layer_norm=layer_norm,
        )
        self.sigma = float(sigma)
        self.rff_kernel = rff_kernel
        self.seed = int(seed)
        self.projection = BlockGaussianProjection(num_features, self.seed, cache_bytes=projection_cache_bytes)
        self._feat_weight_buffers: list[str] = []
        self._feat_bias_buffers: list[str] = []
        self._grad_weight_buffers: list[str] = []
        self._grad_bias_buffers: list[str] = []
        self._simple_weight_buffers: list[str] = []
        self._simple_bias_buffers: list[str] = []

    def transform_joint(self, pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        return self.finish_projection(self.projection.project_pairs(pairs))

    def finish_projection(self, projected: torch.Tensor) -> torch.Tensor:
        """Apply one global phase and cosine, before any atomic pooling."""
        phase = self.projection.phase(projected.device, projected.dtype)
        return math.sqrt(2.0 / self.num_features) * torch.cos(projected / self.sigma + phase)

    def transform_pair(self, feat: torch.Tensor, grad: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self.layer_combine == "joint":
            raise ValueError("joint projection requires all readout layers together; use transform().")
        feat_weight, feat_bias = self._ensure_rff_parameters(
            weight_buffers=self._feat_weight_buffers,
            bias_buffers=self._feat_bias_buffers,
            block=feat,
            layer_idx=layer_idx,
            prefix="feat",
            seed=self.seed + _RFF_FEAT_SEED_STEP * layer_idx,
        )
        grad_weight, grad_bias = self._ensure_rff_parameters(
            weight_buffers=self._grad_weight_buffers,
            bias_buffers=self._grad_bias_buffers,
            block=grad,
            layer_idx=layer_idx,
            prefix="grad",
            seed=self.seed + _RFF_GRAD_SEED_STEP * layer_idx + 1,
        )
        feat_block = self._rff_map(feat, feat_weight, feat_bias, self.num_features)
        grad_block = self._rff_map(grad, grad_weight, grad_bias, self.num_features)
        return math.sqrt(self.num_features) * (feat_block * grad_block)

    def transform_simple(self, atomic_features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        weight, bias = self._ensure_rff_parameters(
            weight_buffers=self._simple_weight_buffers,
            bias_buffers=self._simple_bias_buffers,
            block=atomic_features,
            layer_idx=layer_idx,
            prefix="simple",
            seed=self.seed + _RFF_SIMPLE_SEED_STEP * layer_idx + 3,
        )
        return self._rff_map(atomic_features, weight, bias, self.num_features)

    @staticmethod
    def _rff_map(
        block: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        output_dim: int,
    ) -> torch.Tensor:
        return math.sqrt(2.0 / output_dim) * torch.cos(block @ weight + bias)

    def _ensure_rff_parameters(
        self,
        weight_buffers: list[str],
        bias_buffers: list[str],
        block: torch.Tensor,
        layer_idx: int,
        prefix: str,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_name = self._ensure_buffer_name(weight_buffers, layer_idx, f"{prefix}_weight")
        bias_name = self._ensure_buffer_name(bias_buffers, layer_idx, f"{prefix}_bias")
        weight, bias = self._sample_rff_parameters(
            block.shape[1],
            self.num_features,
            block.device,
            block.dtype,
            seed,
        )
        self._set_buffer_tensor(weight_name, weight)
        self._set_buffer_tensor(bias_name, bias)
        return getattr(self, weight_name), getattr(self, bias_name)

    @staticmethod
    def _ensure_buffer_name(buffer_names: list[str], idx: int, prefix: str) -> str:
        while len(buffer_names) <= idx:
            buffer_names.append(f"{prefix}_{len(buffer_names)}")
        return buffer_names[idx]

    def _set_buffer_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if hasattr(self, name):
            setattr(self, name, tensor)
        else:
            self.register_buffer(name, tensor)

    def _sample_rff_parameters(
        self,
        input_dim: int,
        output_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        if self.rff_kernel == "rbf":
            weight = torch.randn(input_dim, output_dim, generator=generator, dtype=dtype) / self.sigma
        elif self.rff_kernel == "laplacian_l1":
            uniform = torch.rand(input_dim, output_dim, generator=generator, dtype=dtype)
            eps = torch.finfo(dtype).eps
            weight = torch.tan(math.pi * (uniform.clamp(eps, 1 - eps) - 0.5)) / self.sigma
        else:
            degrees = 3 if self.rff_kernel == "matern32" else 5
            normal = torch.randn(input_dim, output_dim, generator=generator, dtype=dtype)
            chi2 = torch.randn(degrees, output_dim, generator=generator, dtype=dtype).square().sum(dim=0)
            weight = normal * torch.sqrt(degrees / chi2.clamp_min(torch.finfo(dtype).tiny)) / self.sigma
        bias = 2.0 * math.pi * torch.rand(output_dim, generator=generator, dtype=dtype)
        return weight.to(device=device, dtype=dtype), bias.to(device=device, dtype=dtype)
