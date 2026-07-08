from __future__ import annotations

import logging
from importlib import util
from typing import Callable, List, Optional, Sequence, Union

import torch
from torch import nn

from curator.data import properties
from ..utils import find_layer_by_name_recursive

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """Extract features and gradients from linear-like layers."""

    def __init__(
        self,
        repr_callback: Optional[Callable] = None,
        model_outputs: Optional[List[str]] = None,
        target_layer: str = "readout",
        target_domain: Optional[Union[str, int]] = None,
        num_layers: Optional[Union[int, str]] = None,
        invariants_only: bool = True,
    ) -> None:
        super().__init__()
        self.repr_callback = repr_callback
        self._features: List[torch.Tensor] = []
        self._grads: List[torch.Tensor] = []
        self.hooks = []
        self.model_outputs = model_outputs if model_outputs is not None else ["feature", "gradient"]
        self.target_layer = target_layer
        self.target_domain = target_domain
        self.num_layers = num_layers
        self.invariants_only = invariants_only
        self._linear_types = self._resolve_linear_types()
        if self.repr_callback is not None:
            self.add_hooks()

    def save_feats_hook(self, _, in_feat) -> None:
        new_feat = torch.cat(
            (in_feat[0].detach().clone(), torch.ones_like(in_feat[0][:, 0:1])),
            dim=-1,
        )
        self._features.append(new_feat)

    def save_segmented_feats_hook(self, module, in_feat) -> None:
        feat = in_feat[0]
        if isinstance(feat, dict):
            feat = feat[properties.node_feat]
        feat = self._select_segmented_feat(module, feat)
        self._features.append(feat)

    def save_grads_hook(self, _, __, grad_output) -> None:
        self._grads.append(grad_output[0].detach().clone())

    def unhook(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def attach(self, repr_callback: nn.Module) -> None:
        if self.hooks:
            self.unhook()
        self.repr_callback = repr_callback
        self.add_hooks()

    def detach(self) -> None:
        self.unhook()
        self.repr_callback = None

    def register_repr_callback(self, repr_callback: nn.Module) -> None:
        self.attach(repr_callback)

    def add_hooks(self) -> None:
        if self.repr_callback is None:
            raise RuntimeError("repr_callback is not set.")
        search_root = self.repr_callback
        readout = find_layer_by_name_recursive(self.repr_callback, "readout")
        domain_modules = getattr(readout, "domain_modules", None)
        if self.target_domain is not None:
            if domain_modules is None:
                raise ValueError("target_domain is set but model has no domain_modules.")
            dom = str(self.target_domain)
            if dom not in domain_modules:
                raise ValueError(f"target_domain '{dom}' not found in model domain_modules.")
            search_root = domain_modules[dom]
        elif domain_modules is not None and len(domain_modules) > 1:
            logger.warning(
                "Multiple domains detected in readout; FeatureExtractor defaults to the first discovered domain. "
                "Set target_domain to select explicitly."
            )

        layer = self._resolve_target_layer(search_root)
        assert layer is not None, f"Target layer {self.target_layer} is not found!"
        if self.num_layers is not None:
            if not self._has_segmented_input(layer):
                raise ValueError("num_layers requires a target layer with segmented inputs.")
            self.hooks.append(layer.register_forward_pre_hook(self.save_segmented_feats_hook))
            return

        layer = self._default_target(layer)
        linear_modules = [m for m in layer.modules() if isinstance(m, self._linear_types)]
        if not linear_modules:
            logger.warning("No linear-like submodules found under target layer %s", self.target_layer)
        for child in linear_modules:
            self.hooks.append(child.register_forward_pre_hook(self.save_feats_hook))
            self.hooks.append(child.register_backward_hook(self.save_grads_hook))

    def forward(self, data: properties.Type, predict: bool = False) -> properties.Type:
        if predict:
            data = self.repr_callback(data.copy())
        data[properties.feature] = self._features
        data[properties.gradient] = self._grads[::-1]
        self._reset()
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(target_layer={self.target_layer}, "
            f"target_domain={self.target_domain}, num_layers={self.num_layers}, "
            f"invariants_only={self.invariants_only})"
        )

    def _reset(self) -> None:
        self._features = []
        self._grads = []

    def _select_segmented_feat(self, module: nn.Module, feat: torch.Tensor) -> torch.Tensor:
        widths = [int(width) for width in getattr(module, "in_features_list")]
        start_segment = 0
        if isinstance(self.num_layers, str):
            if self.num_layers.lower() not in {"last", "final"}:
                raise ValueError("num_layers must be positive, -1, or 'last'.")
            start_segment = len(widths) - 1
            num_segments = 1
        elif self.num_layers == -1:
            num_segments = len(widths)
        elif self.num_layers is not None and self.num_layers > 0:
            if self.num_layers > len(widths):
                raise ValueError(f"num_layers={self.num_layers} exceeds available segments {len(widths)}.")
            num_segments = self.num_layers
        else:
            raise ValueError("num_layers must be positive, -1, or 'last'.")
        feat = self._slice_segmented_feat(module, feat, widths, start_segment, num_segments).detach().clone()
        return torch.cat((feat, torch.ones_like(feat[:, 0:1])), dim=-1)

    def _slice_segmented_feat(
        self,
        module: nn.Module,
        feat: torch.Tensor,
        widths: List[int],
        start_segment: int,
        num_segments: int,
    ) -> torch.Tensor:
        end_segment = start_segment + num_segments
        if not self.invariants_only:
            return feat[:, sum(widths[:start_segment]) : sum(widths[:end_segment])]
        invariant_widths = getattr(module, "invariant_features_list", widths)
        if len(invariant_widths) < end_segment:
            raise ValueError("invariant_features_list is shorter than selected segments.")
        out = []
        start = sum(widths[:start_segment])
        for width, invariant_width in zip(widths[start_segment:end_segment], invariant_widths[start_segment:end_segment]):
            out.append(feat[:, start : start + int(invariant_width)])
            start += width
        return torch.cat(out, dim=-1)

    def _resolve_target_layer(self, search_root: nn.Module) -> Optional[nn.Module]:
        layer = find_layer_by_name_recursive(search_root, self.target_layer)
        if layer is not None or self.target_layer != "readout":
            return layer
        if hasattr(search_root, "readouts") or self._has_segmented_input(search_root):
            return search_root
        return None

    @staticmethod
    def _default_target(layer: nn.Module) -> nn.Module:
        readouts = getattr(layer, "readouts", None)
        if readouts is not None and len(readouts) > 0:
            return readouts[-1]
        return layer

    @staticmethod
    def _has_segmented_input(layer: nn.Module) -> bool:
        return hasattr(layer, "in_features_list")

    @staticmethod
    def _resolve_linear_types() -> Sequence[type]:
        types: List[type] = [nn.Linear]
        if util.find_spec("e3nn.o3"):
            from e3nn import o3

            types.append(o3.Linear)
        try:
            has_nequip = util.find_spec("nequip") is not None
        except ModuleNotFoundError:
            has_nequip = False
        if has_nequip:
            try:
                from nequip.nn.mlp import ScalarLinearLayer

                types.append(ScalarLinearLayer)
            except Exception:
                pass
        try:
            from curator.layer.wrappers.cueq import IS_CUET_AVAILABLE, cuet
            from curator.layer.wrappers.cueq_elora import CueqLoRALinear

            if IS_CUET_AVAILABLE and cuet is not None:
                types.append(cuet.Linear)
            types.append(CueqLoRALinear)
        except Exception:
            pass
        return tuple(types)
