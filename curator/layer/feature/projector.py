from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


_RMS_EPS = 1e-8


class FeatureProjector(nn.Module):
    """Project raw extracted features to atom-level features."""

    def __init__(
        self,
        raw_feature_source: str,
        num_features: int = 0,
        layer_combine: str = "concat",
        layer_norm: str = "none",
        sigma: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if layer_combine not in {"concat", "sum"}:
            raise ValueError(f"Unsupported layer_combine '{layer_combine}'.")
        if layer_norm not in {"none", "rms"}:
            raise ValueError(f"Unsupported layer_norm '{layer_norm}'.")
        self.raw_feature_source = raw_feature_source
        self.num_features = int(num_features)
        self.layer_combine = layer_combine
        self.layer_norm = layer_norm
        self.sigma = float(sigma)
        self.seed = int(seed)

    def compute(self, feats: list[torch.Tensor], grads: list[torch.Tensor]) -> torch.Tensor:
        if self.raw_feature_source == "full-gradient":
            return self._compute_full_gradient(feats, grads)
        if self.raw_feature_source == "ll-gradient":
            return self.transform_simple(feats[-1][:, :-1], len(feats) - 1)
        if self.raw_feature_source == "gnn":
            return self.transform_simple(feats[0][:, :-1], 0)
        raise ValueError(f"Unsupported raw_feature_source '{self.raw_feature_source}'.")

    def _compute_full_gradient(self, feats: list[torch.Tensor], grads: list[torch.Tensor]) -> torch.Tensor:
        blocks = []
        for idx, (feat, grad) in enumerate(zip(feats, grads)):
            block = self.transform_pair(feat, grad, idx)
            blocks.append(self._maybe_rms_norm(block))
        if not blocks:
            return torch.zeros((0, 0))
        if self.layer_combine == "sum":
            return torch.stack(blocks, dim=0).sum(dim=0)
        return torch.cat(blocks, dim=1)

    def transform_pair(self, feat: torch.Tensor, grad: torch.Tensor, layer_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def transform_simple(self, atomic_features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def _maybe_rms_norm(self, block: torch.Tensor) -> torch.Tensor:
        if self.layer_norm != "rms":
            return block
        denom = torch.sqrt(torch.mean(block * block, dim=1, keepdim=True) + _RMS_EPS)
        return block / denom


class IdentityProjector(FeatureProjector):
    def __init__(self, raw_feature_source: str) -> None:
        super().__init__(raw_feature_source=raw_feature_source, num_features=0)

    def transform_pair(self, feat: torch.Tensor, grad: torch.Tensor, layer_idx: int) -> torch.Tensor:
        raise ValueError("IdentityProjector does not support full-gradient features.")

    def transform_simple(self, atomic_features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return atomic_features


class SketchingProjector(FeatureProjector):
    def __init__(
        self,
        raw_feature_source: str,
        num_features: int,
        layer_combine: str = "concat",
        layer_norm: str = "none",
        seed: int = 0,
    ) -> None:
        if num_features <= 0:
            raise ValueError("num_features must be positive for sketching projector.")
        super().__init__(
            raw_feature_source=raw_feature_source,
            num_features=num_features,
            layer_combine=layer_combine,
            layer_norm=layer_norm,
            seed=seed,
        )
        self._feat_proj_buffers: list[str] = []
        self._grad_proj_buffers: list[str] = []
        self._simple_proj_buffers: list[str] = []

    def transform_pair(self, feat: torch.Tensor, grad: torch.Tensor, layer_idx: int) -> torch.Tensor:
        feat_proj = getattr(self, self._ensure_feat_proj(feat, layer_idx))
        grad_proj = getattr(self, self._ensure_grad_proj(grad, layer_idx))
        return ((feat @ feat_proj) * (grad @ grad_proj)) / math.sqrt(self.num_features)

    def transform_simple(self, atomic_features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        proj = getattr(self, self._ensure_simple_proj(atomic_features, layer_idx))
        return (atomic_features @ proj) / math.sqrt(self.num_features)

    def _ensure_feat_proj(self, feat: torch.Tensor, layer_idx: int) -> str:
        name = self._ensure_buffer_name(self._feat_proj_buffers, layer_idx, "feat_proj")
        self._set_buffer_tensor(
            name,
            self._randn_tensor(feat.shape[1], self.num_features, feat.device, feat.dtype, self.seed + 13 * layer_idx),
        )
        return name

    def _ensure_grad_proj(self, grad: torch.Tensor, layer_idx: int) -> str:
        name = self._ensure_buffer_name(self._grad_proj_buffers, layer_idx, "grad_proj")
        self._set_buffer_tensor(
            name,
            self._randn_tensor(grad.shape[1], self.num_features, grad.device, grad.dtype, self.seed + 17 * layer_idx + 1),
        )
        return name

    def _ensure_simple_proj(self, atomic_features: torch.Tensor, layer_idx: int) -> str:
        name = self._ensure_buffer_name(self._simple_proj_buffers, layer_idx, "simple_proj")
        self._set_buffer_tensor(
            name,
            self._randn_tensor(
                atomic_features.shape[1],
                self.num_features,
                atomic_features.device,
                atomic_features.dtype,
                self.seed + 19 * layer_idx + 3,
            ),
        )
        return name

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


class RandomFourierProjector(FeatureProjector):
    def __init__(
        self,
        raw_feature_source: str,
        num_features: int,
        layer_combine: str = "concat",
        layer_norm: str = "none",
        sigma: float = 1.0,
        seed: int = 0,
    ) -> None:
        if num_features <= 0:
            raise ValueError("num_features must be positive for RFF projector.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        super().__init__(
            raw_feature_source=raw_feature_source,
            num_features=num_features,
            layer_combine=layer_combine,
            layer_norm=layer_norm,
            sigma=sigma,
            seed=seed,
        )
        self._feat_weight_buffers: list[str] = []
        self._feat_bias_buffers: list[str] = []
        self._grad_weight_buffers: list[str] = []
        self._grad_bias_buffers: list[str] = []
        self._simple_weight_buffers: list[str] = []
        self._simple_bias_buffers: list[str] = []

    def transform_pair(self, feat: torch.Tensor, grad: torch.Tensor, layer_idx: int) -> torch.Tensor:
        feat_weight = getattr(self, self._ensure_feat_weight(feat, layer_idx))
        feat_bias = getattr(self, self._ensure_feat_bias(feat, layer_idx))
        grad_weight = getattr(self, self._ensure_grad_weight(grad, layer_idx))
        grad_bias = getattr(self, self._ensure_grad_bias(grad, layer_idx))
        feat_block = self._rff_map(feat, feat_weight, feat_bias, self.num_features)
        grad_block = self._rff_map(grad, grad_weight, grad_bias, self.num_features)
        return math.sqrt(self.num_features) * (feat_block * grad_block)

    def transform_simple(self, atomic_features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        weight = getattr(self, self._ensure_simple_weight(atomic_features, layer_idx))
        bias = getattr(self, self._ensure_simple_bias(atomic_features, layer_idx))
        return self._rff_map(atomic_features, weight, bias, self.num_features)

    @staticmethod
    def _rff_map(
        block: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        output_dim: int,
    ) -> torch.Tensor:
        return math.sqrt(2.0 / output_dim) * torch.cos(block @ weight + bias)

    def _ensure_feat_weight(self, feat: torch.Tensor, layer_idx: int) -> str:
        name = self._ensure_buffer_name(self._feat_weight_buffers, layer_idx, "feat_weight")
        weight, _ = self._sample_rff_parameters(
            feat.shape[1], self.num_features, feat.device, feat.dtype, self.seed + 29 * layer_idx
        )
        self._set_buffer_tensor(name, weight)
        return name

    def _ensure_feat_bias(self, feat: torch.Tensor, layer_idx: int) -> str:
        name = self._ensure_buffer_name(self._feat_bias_buffers, layer_idx, "feat_bias")
        _, bias = self._sample_rff_parameters(
            feat.shape[1], self.num_features, feat.device, feat.dtype, self.seed + 29 * layer_idx
        )
        self._set_buffer_tensor(name, bias)
        return name

    def _ensure_grad_weight(self, grad: torch.Tensor, layer_idx: int) -> str:
        name = self._ensure_buffer_name(self._grad_weight_buffers, layer_idx, "grad_weight")
        weight, _ = self._sample_rff_parameters(
            grad.shape[1], self.num_features, grad.device, grad.dtype, self.seed + 31 * layer_idx + 1
        )
        self._set_buffer_tensor(name, weight)
        return name

    def _ensure_grad_bias(self, grad: torch.Tensor, layer_idx: int) -> str:
        name = self._ensure_buffer_name(self._grad_bias_buffers, layer_idx, "grad_bias")
        _, bias = self._sample_rff_parameters(
            grad.shape[1], self.num_features, grad.device, grad.dtype, self.seed + 31 * layer_idx + 1
        )
        self._set_buffer_tensor(name, bias)
        return name

    def _ensure_simple_weight(self, atomic_features: torch.Tensor, layer_idx: int) -> str:
        name = self._ensure_buffer_name(self._simple_weight_buffers, layer_idx, "simple_weight")
        weight, _ = self._sample_rff_parameters(
            atomic_features.shape[1],
            self.num_features,
            atomic_features.device,
            atomic_features.dtype,
            self.seed + 37 * layer_idx + 3,
        )
        self._set_buffer_tensor(name, weight)
        return name

    def _ensure_simple_bias(self, atomic_features: torch.Tensor, layer_idx: int) -> str:
        name = self._ensure_buffer_name(self._simple_bias_buffers, layer_idx, "simple_bias")
        _, bias = self._sample_rff_parameters(
            atomic_features.shape[1],
            self.num_features,
            atomic_features.device,
            atomic_features.dtype,
            self.seed + 37 * layer_idx + 3,
        )
        self._set_buffer_tensor(name, bias)
        return name

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
        weight = torch.randn(input_dim, output_dim, generator=generator, dtype=dtype) / self.sigma
        bias = 2.0 * math.pi * torch.rand(output_dim, generator=generator, dtype=dtype)
        return weight.to(device=device, dtype=dtype), bias.to(device=device, dtype=dtype)
