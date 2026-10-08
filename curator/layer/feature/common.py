from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Literal, Optional

import torch

KernelName = Literal[
    "full-g",
    "ll-g",
    "local-full-g",
    "local_full-g",
    "local-ll-g",
    "local_ll-g",
    "local-gnn",
    "full-gradient",
    "ll-gradient",
    "gnn",
    "local_full-gradient",
    "local_ll-gradient",
    "local_gnn",
]
Reduction = Literal["mean", "sum"]
MappingName = Literal["identity", "gaussian_sketch", "rff", "nystrom"]
RFFKernelName = Literal["rbf", "matern32", "matern52", "laplacian_l1"]
LayerCombineName = Literal["concat", "sum", "joint"]
LayerNormName = Literal["none", "rms"]

_DEFAULT_KERNEL = "full-g"
_DEFAULT_NUM_FEATURES = 256


def normalize_kernel(kernel: KernelName) -> str:
    aliases = {
        "full-g": "full-gradient",
        "ll-g": "ll-gradient",
        "local-full-g": "local_full-gradient",
        "local_full-g": "local_full-gradient",
        "local-ll-g": "local_ll-gradient",
        "local_ll-g": "local_ll-gradient",
        "local-gnn": "local_gnn",
    }
    return aliases.get(kernel, kernel)


def _normalize_raw_feature(raw_feature: str) -> str:
    normalized = normalize_kernel(raw_feature)  # type: ignore[arg-type]
    return normalized[len("local_") :] if normalized.startswith("local_") else normalized


def _feature_spec_presets() -> Dict[str, Dict[str, Any]]:
    presets = {
        "fg-sketch": {
            "name": "fg-readout-sketch-v1",
            "raw_feature": "full-gradient",
            "mapping": "gaussian_sketch",
            "num_features": _DEFAULT_NUM_FEATURES,
            "layer_combine": "joint",
            "layer_norm": "none",
            "pooling": "mean",
            "sigma": 1.0,
            "seed": 0,
        },
        "fg-sketch-rms": {
            "name": "fg-sketch-rms",
            "raw_feature": "full-gradient",
            "mapping": "gaussian_sketch",
            "num_features": _DEFAULT_NUM_FEATURES,
            "layer_combine": "concat",
            "layer_norm": "rms",
            "pooling": "sum",
            "sigma": 1.0,
            "seed": 0,
        },
        "fg-rff": {
            "name": "fg-readout-rff-v1",
            "raw_feature": "full-gradient",
            "mapping": "rff",
            "num_features": _DEFAULT_NUM_FEATURES,
            "layer_combine": "joint",
            "layer_norm": "none",
            "pooling": "mean",
            "sigma": 1.0,
            "seed": 0,
        },
        "llg-id": {
            "name": "llg-id",
            "raw_feature": "ll-gradient",
            "mapping": "identity",
            "num_features": _DEFAULT_NUM_FEATURES,
            "layer_combine": "concat",
            "layer_norm": "none",
            "pooling": "sum",
            "sigma": 1.0,
            "seed": 0,
        },
        "llg-sketch": {
            "name": "llg-sketch",
            "raw_feature": "ll-gradient",
            "mapping": "gaussian_sketch",
            "num_features": _DEFAULT_NUM_FEATURES,
            "layer_combine": "concat",
            "layer_norm": "none",
            "pooling": "sum",
            "sigma": 1.0,
            "seed": 0,
        },
        "llg-rff": {
            "name": "llg-rff",
            "raw_feature": "ll-gradient",
            "mapping": "rff",
            "num_features": _DEFAULT_NUM_FEATURES,
            "layer_combine": "concat",
            "layer_norm": "none",
            "pooling": "sum",
            "sigma": 1.0,
            "seed": 0,
        },
        "gnn-id": {
            "name": "gnn-id",
            "raw_feature": "gnn",
            "mapping": "identity",
            "num_features": _DEFAULT_NUM_FEATURES,
            "layer_combine": "concat",
            "layer_norm": "none",
            "pooling": "sum",
            "sigma": 1.0,
            "seed": 0,
        },
        "gnn-sketch": {
            "name": "gnn-sketch",
            "raw_feature": "gnn",
            "mapping": "gaussian_sketch",
            "num_features": _DEFAULT_NUM_FEATURES,
            "layer_combine": "concat",
            "layer_norm": "none",
            "pooling": "sum",
            "sigma": 1.0,
            "seed": 0,
        },
        "gnn-rff": {
            "name": "gnn-rff",
            "raw_feature": "gnn",
            "mapping": "rff",
            "num_features": _DEFAULT_NUM_FEATURES,
            "layer_combine": "concat",
            "layer_norm": "none",
            "pooling": "sum",
            "sigma": 1.0,
            "seed": 0,
        },
        "local_fg-sketch": {
            "name": "local_fg-readout-sketch-v1",
            "raw_feature": "full-gradient",
            "mapping": "gaussian_sketch",
            "num_features": _DEFAULT_NUM_FEATURES,
            "layer_combine": "joint",
            "layer_norm": "none",
            "pooling": "mean",
            "sigma": 1.0,
            "seed": 0,
        },
        "local_fg-rff": {
            "name": "local_fg-readout-rff-v1",
            "raw_feature": "full-gradient",
            "mapping": "rff",
            "num_features": _DEFAULT_NUM_FEATURES,
            "layer_combine": "joint",
            "layer_norm": "none",
            "pooling": "mean",
            "sigma": 1.0,
            "seed": 0,
        },
        "local_llg-id": {
            "name": "local_llg-id",
            "raw_feature": "ll-gradient",
            "mapping": "identity",
            "num_features": _DEFAULT_NUM_FEATURES,
            "layer_combine": "concat",
            "layer_norm": "none",
            "pooling": "sum",
            "sigma": 1.0,
            "seed": 0,
        },
        "local_gnn-id": {
            "name": "local_gnn-id",
            "raw_feature": "gnn",
            "mapping": "identity",
            "num_features": _DEFAULT_NUM_FEATURES,
            "layer_combine": "concat",
            "layer_norm": "none",
            "pooling": "sum",
            "sigma": 1.0,
            "seed": 0,
        },
    }
    for short_name, source in (("fg", "full-gradient"), ("llg", "ll-gradient"), ("gnn", "gnn")):
        for prefix in ("", "local_"):
            name = f"{prefix}{short_name}-nystrom"
            presets[name] = {
                "name": name,
                "raw_feature": source,
                "mapping": "nystrom",
                "num_features": _DEFAULT_NUM_FEATURES,
                "layer_combine": "joint" if source == "full-gradient" else "concat",
                "layer_norm": "none",
                "pooling": "mean",
                "sigma": 1.0,
            }
    return presets


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    raw_feature: str
    mapping: MappingName = "identity"
    num_features: int = _DEFAULT_NUM_FEATURES
    layer_combine: LayerCombineName = "concat"
    layer_norm: LayerNormName = "none"
    pooling: Reduction = "sum"
    sigma: float = 1.0
    seed: int = 0
    rff_kernel: RFFKernelName = "rbf"
    projection_cache_bytes: int = 64 * 1024**2
    nystrom_state: Optional[str] = None

    @property
    def kernel_name(self) -> str:
        return normalize_kernel(self.name)  # type: ignore[arg-type]

    @property
    def local(self) -> bool:
        return self.kernel_name.startswith("local_")

    @property
    def source(self) -> str:
        return _normalize_raw_feature(self.raw_feature)


def feature_spec_from_object(obj: Any) -> FeatureSpec:
    if isinstance(obj, FeatureSpec):
        spec = obj
    elif isinstance(obj, dict):
        data = dict(obj)
        preset = data.pop("preset", None)
        if preset is not None:
            presets = _feature_spec_presets()
            if preset not in presets:
                raise ValueError(
                    f"Unknown feature preset '{preset}'. Available presets: {sorted(presets)}"
                )
            preset_data = dict(presets[preset])
            preset_data.update(data)
            if "name" not in preset_data:
                preset_data["name"] = str(preset)
            data = preset_data
        spec = FeatureSpec(**data)
    else:
        raise TypeError("feature spec must be a FeatureSpec or dict.")
    if spec.pooling not in {"sum", "mean"}:
        raise ValueError(f"Unsupported pooling '{spec.pooling}'.")
    if spec.mapping not in {"identity", "gaussian_sketch", "rff", "nystrom"}:
        raise ValueError(f"Unsupported mapping '{spec.mapping}'.")
    if spec.layer_combine not in {"concat", "sum", "joint"}:
        raise ValueError(f"Unsupported layer_combine '{spec.layer_combine}'.")
    if spec.layer_norm not in {"none", "rms"}:
        raise ValueError(f"Unsupported layer_norm '{spec.layer_norm}'.")
    if spec.mapping in {"gaussian_sketch", "rff", "nystrom"} and spec.num_features <= 0:
        raise ValueError("num_features must be positive for gaussian_sketch, rff, and nystrom mappings.")
    if not math.isfinite(spec.sigma) or spec.sigma <= 0:
        raise ValueError("sigma must be positive.")
    if spec.mapping == "rff" and spec.rff_kernel not in {"rbf", "matern32", "matern52", "laplacian_l1"}:
        raise ValueError(f"Unsupported RFF kernel '{spec.rff_kernel}'.")
    source = spec.source
    if spec.mapping == "nystrom":
        if not isinstance(spec.nystrom_state, str) or not spec.nystrom_state.strip():
            raise ValueError("nystrom mapping requires a nonempty nystrom_state file path; prepare the fixed map first.")
        if spec.layer_norm != "none" or spec.rff_kernel != "rbf":
            raise ValueError("nystrom supports only the Gaussian RBF kernel without layer normalization.")
        if source == "full-gradient" and spec.layer_combine != "joint":
            raise ValueError("Readout nystrom requires layer_combine='joint' for the complete gradient Gaussian kernel.")
    if not isinstance(spec.projection_cache_bytes, int) or isinstance(spec.projection_cache_bytes, bool) or spec.projection_cache_bytes < 0:
        raise ValueError("projection_cache_bytes must be a nonnegative integer.")
    if spec.layer_combine == "joint":
        if source != "full-gradient":
            raise ValueError("joint projection is supported only for readout full-gradient features.")
        if spec.layer_norm != "none":
            raise ValueError("joint projection requires layer_norm='none'; no layer normalization is applied.")
        if spec.mapping == "rff" and spec.rff_kernel != "rbf":
            raise ValueError("joint readout RFF supports only rff_kernel='rbf'.")
    if source == "full-gradient":
        if spec.mapping not in {"gaussian_sketch", "rff", "nystrom"}:
            raise ValueError("full-gradient supports only gaussian_sketch, rff, and nystrom mappings.")
    elif source in {"ll-gradient", "gnn"}:
        if spec.mapping not in {"identity", "gaussian_sketch", "rff", "nystrom"}:
            raise ValueError(f"{source} supports only identity, gaussian_sketch, rff, and nystrom mappings.")
    elif source != "atomic":
        raise ValueError(f"Unsupported raw_feature '{spec.raw_feature}'.")
    return spec


@dataclass
class ExtractedFeatures:
    image_idx: torch.Tensor
    feats: List[torch.Tensor]
    grads: List[torch.Tensor]
    atomic_numbers: Optional[torch.Tensor] = None
    num_atoms: Optional[torch.Tensor] = None
    readout_layouts: Optional[List[Dict[str, Any]]] = None
