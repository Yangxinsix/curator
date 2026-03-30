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
        target_layer: str = "readout_mlp",
        target_domain: Optional[Union[str, int]] = None,
    ) -> None:
        super().__init__()
        self.repr_callback = repr_callback
        self._features: List[torch.Tensor] = []
        self._grads: List[torch.Tensor] = []
        self.hooks = []
        self.model_outputs = model_outputs if model_outputs is not None else ["feature", "gradient"]
        self.target_layer = target_layer
        self.target_domain = target_domain
        self._linear_types = self._resolve_linear_types()
        if self.repr_callback is not None:
            self.add_hooks()

    def save_feats_hook(self, _, in_feat) -> None:
        new_feat = torch.cat(
            (in_feat[0].detach().clone(), torch.ones_like(in_feat[0][:, 0:1])),
            dim=-1,
        )
        self._features.append(new_feat)

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

        layer = find_layer_by_name_recursive(search_root, self.target_layer)
        assert layer is not None, f"Target layer {self.target_layer} is not found!"
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
        return f"{self.__class__.__name__}(target_layer={self.target_layer}, target_domain={self.target_domain})"

    def _reset(self) -> None:
        self._features = []
        self._grads = []

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
