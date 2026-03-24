from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Optional, Sequence, Union
import warnings

import torch
from torch import nn

from curator.data import properties
from curator.layer import AtomwiseNN
from curator.layer.utils import find_layer_by_name_recursive
from curator.model.base import Representation
from curator.model.external_utils import (
    build_image_index,
    extract_state_dict,
    strip_state_dict_prefix,
)

try:
    from torch_scatter import scatter_add
except ImportError:
    from curator.utils import scatter_add


def _unwrap_hook_payload(payload):
    if isinstance(payload, tuple):
        if len(payload) == 1:
            return payload[0]
        return payload
    return payload


def _batch_get(batch, key: str, default=None):
    if isinstance(batch, dict):
        return batch.get(key, default)
    if hasattr(batch, key):
        return getattr(batch, key)
    try:
        return batch[key]
    except Exception:
        return default


class _FeatureCapture:
    def __init__(self, module: nn.Module, mode: str) -> None:
        self.value = None
        self.mode = mode
        if mode == "input":
            self.handle = module.register_forward_pre_hook(self._capture_input)
        elif mode == "output":
            self.handle = module.register_forward_hook(self._capture_output)
        else:
            raise ValueError(f"Unsupported feature hook mode '{mode}'.")

    def clear(self) -> None:
        self.value = None

    def _capture_input(self, module, args):
        self.value = _unwrap_hook_payload(args)

    def _capture_output(self, module, args, output):
        self.value = _unwrap_hook_payload(output)


class ExternalBackboneRepresentation(Representation):
    state_dict_prefixes: Sequence[str] = (
        "model.representation.backbone.",
        "representation.backbone.",
        "backbone.",
        "module.",
    )
    feature_keys: Sequence[str] = ()
    node_feature_keys: Sequence[str] = ()
    edge_feature_keys: Sequence[str] = ()

    def __init__(
        self,
        backbone: nn.Module,
        cutoff: Optional[float] = None,
        pretrained_path: Optional[str] = None,
        strict_load: bool = True,
        feature_dim: int = 128,
        feature_layer: Optional[str] = None,
        feature_hook: str = "output",
        readout=AtomwiseNN,
        heads: Optional[list] = None,
    ) -> None:
        if heads is None:
            heads = [
                {
                    "key": properties.energy,
                    "is_atomwise": True,
                    "reduction": "sum",
                    "atomwise_key": properties.atomic_energy,
                    "write_atomwise": True,
                }
            ]
        super().__init__(heads=heads)
        if backbone is None:
            raise ValueError("A trainable external representation requires a backbone module.")
        self.backbone = backbone
        self.cutoff = float(cutoff) if cutoff is not None else float(self._infer_cutoff())
        self.feature_dim = int(feature_dim)
        self.feature_layer = feature_layer
        self.feature_hook = feature_hook
        self.projection = nn.LazyLinear(self.feature_dim)
        self._feature_capture = None
        if self.feature_layer is not None:
            target_module = find_layer_by_name_recursive(self.backbone, self.feature_layer)
            if target_module is None:
                raise ValueError(f"Cannot find feature layer '{self.feature_layer}' on {type(self.backbone).__name__}.")
            self._feature_capture = _FeatureCapture(target_module, mode=self.feature_hook)
        self.readout = self._instantiate_readout(
            readout,
            heads=self.heads,
            in_features=self.feature_dim,
        )
        if pretrained_path is not None:
            self.load_backbone_state(pretrained_path, strict=strict_load)

    def _infer_cutoff(self) -> float:
        raise NotImplementedError

    def _build_native_batch(self, data: properties.Type):
        raise NotImplementedError

    def _resolve_feature_tensor(self, native_batch, backbone_output) -> torch.Tensor:
        if self._feature_capture is not None and self._feature_capture.value is not None:
            captured = self._feature_capture.value
            if torch.is_tensor(captured):
                return captured
            if isinstance(captured, (list, tuple)):
                for value in captured:
                    if torch.is_tensor(value):
                        return value
        for container in (backbone_output, native_batch):
            if container is None:
                continue
            value = self._resolve_feature_tensor_from_container(container)
            if value is not None:
                return value
        raise ValueError(
            f"Unable to resolve latent features for {type(self).__name__}. "
            f"Set `feature_layer` to an explicit module name."
        )

    def _resolve_feature_tensor_from_container(self, container) -> Optional[torch.Tensor]:
        for key in self.feature_keys:
            value = _batch_get(container, key)
            if torch.is_tensor(value):
                return value
        return None

    def _coerce_node_features(self, feature_tensor: torch.Tensor, native_batch, data: properties.Type) -> torch.Tensor:
        num_nodes = int(data[properties.atomic_numbers].shape[0])
        if feature_tensor.dim() == 1:
            feature_tensor = feature_tensor.unsqueeze(-1)
        if feature_tensor.dim() != 2:
            raise ValueError(
                f"Expected a rank-2 latent tensor for {type(self).__name__}, got shape {tuple(feature_tensor.shape)}."
            )
        if feature_tensor.shape[0] == num_nodes:
            return feature_tensor

        edge_index = _batch_get(native_batch, "edge_index")
        if edge_index is None:
            raise ValueError(
                f"Latent tensor shape {tuple(feature_tensor.shape)} does not match node count {num_nodes}, "
                "and no edge_index was available for edge-to-node aggregation."
            )
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Unsupported edge_index shape: {tuple(edge_index.shape)}")
        if feature_tensor.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"Latent tensor shape {tuple(feature_tensor.shape)} does not match nodes or edges for {type(self).__name__}."
            )
        target_nodes = edge_index[0].to(torch.long)
        return scatter_add(feature_tensor, target_nodes, dim=0, dim_size=num_nodes)

    def load_backbone_state(self, pretrained_path: str, strict: bool = True):
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state_dict = extract_state_dict(checkpoint)
        state_dict = strip_state_dict_prefix(state_dict, self.state_dict_prefixes)
        result = self.backbone.load_state_dict(state_dict, strict=strict)
        if not strict:
            missing = list(getattr(result, "missing_keys", []))
            unexpected = list(getattr(result, "unexpected_keys", []))
            if missing or unexpected:
                warnings.warn(
                    f"Loaded external backbone from {pretrained_path} with missing keys {missing} "
                    f"and unexpected keys {unexpected}.",
                    RuntimeWarning,
                )
        return result

    def module_groups(self):
        return OrderedDict(
            (
                ("backbone", [self.backbone]),
                ("projection", [self.projection]),
                ("readout", [self.readout]),
            )
        )

    def forward(self, data: properties.Type) -> properties.Type:
        if self._feature_capture is not None:
            self._feature_capture.clear()
        if properties.image_idx not in data:
            data[properties.image_idx] = build_image_index(
                data[properties.n_atoms],
                data[properties.positions].device,
            )
        native_batch = self._build_native_batch(data)
        backbone_output = self.backbone(native_batch)
        feature_tensor = self._resolve_feature_tensor(native_batch, backbone_output)
        node_features = self._coerce_node_features(feature_tensor, native_batch, data)
        projected = self.projection(node_features)
        data[properties.node_embedding] = projected
        data[properties.node_feat] = projected
        data = self.readout(data)
        return data
