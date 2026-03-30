import inspect
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

import torch
from pytorch_lightning import LightningDataModule
from torch import nn

from curator.data import properties


@dataclass
class ParameterGroup:
    name: str
    params: List[nn.Parameter]
    defaults: Optional[Dict[str, Any]] = None


def collect_unique_parameters(
    items: Iterable[Any],
    *,
    seen: Optional[set[int]] = None,
    require_grad: Optional[bool] = None,
) -> List[nn.Parameter]:
    if seen is None:
        seen = set()
    collected: List[nn.Parameter] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, nn.Parameter):
            iterator = (item,)
        elif isinstance(item, nn.Module):
            iterator = item.parameters()
        else:
            iterator = item
        for param in iterator:
            if not isinstance(param, nn.Parameter):
                continue
            if require_grad is not None and bool(param.requires_grad) != bool(require_grad):
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            collected.append(param)
    return collected

class Representation(nn.Module):
    """
    Shared mixin/base to standardize handling of head configs and readout instantiation
    across representations (MACE/Nequip/PAINN).
    """

    def __init__(self, heads: Optional[list] = None) -> None:
        super().__init__()
        self.heads = heads or []

    def _instantiate_readout(
        self,
        readout: Union[nn.Module, Type[nn.Module], Callable],
        heads: Optional[list] = None,
        **kwargs,
    ) -> nn.Module:
        """Instantiate readout, passing heads when supported."""
        if isinstance(readout, nn.Module):
            return readout

        call = readout
        if isinstance(readout, partial):
            call = readout.func

        sig = inspect.signature(call)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )

        def maybe(name, value):
            return {name: value} if value is not None and (accepts_kwargs or name in sig.parameters) else {}

        if accepts_kwargs:
            init_kwargs = {key: value for key, value in kwargs.items() if value is not None}
        else:
            init_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in sig.parameters and value is not None
            }
        init_kwargs.update(maybe("heads", heads))

        return readout(**init_kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _normalize_readout_factory(
        readout: Union[nn.Module, Type[nn.Module], Callable, Dict[str, Any]],
        *,
        base_cls: Type[nn.Module],
    ) -> Union[nn.Module, Type[nn.Module], Callable]:
        if hasattr(readout, "get") and not isinstance(readout, nn.Module) and not callable(readout):
            readout_kwargs = dict(readout)
            if "_target_" in readout_kwargs:
                return readout
            return partial(base_cls, **readout_kwargs)
        return readout

    @staticmethod
    def _enable_cueq(use_cueq: bool):
        """Helper to enable cuequivariance with a single warning path."""
        if not use_cueq:
            return
        from curator.layer._cuequivariance_wrapper import IS_CUET_AVAILABLE, set_use_cueq
        import warnings

        set_use_cueq(use_cueq)
        if use_cueq and not IS_CUET_AVAILABLE:
            warnings.warn(
                "Requested use_cueq=True but cuequivariance is not available; falling back to e3nn kernels.",
                RuntimeWarning,
            )

    @staticmethod
    def _apply_cutoff_mask(data: properties.Type, cutoff: float):
        """Apply edge cutoff mask in-place. Returns original edges for optional downstream use."""
        try:
            edge_idx = data[properties.edge_idx]
            edge_diff = data[properties.edge_diff]
            edge_dist = data[properties.edge_dist]
        except KeyError:
            return None
        mask = edge_dist < cutoff
        data[properties.edge_idx] = edge_idx[mask]
        data[properties.edge_diff] = edge_diff[mask]
        data[properties.edge_dist] = edge_dist[mask]
        return (edge_idx, edge_diff, edge_dist)

    @staticmethod
    def _restore_cutoff_mask(data: properties.Type, cache):
        """Restore edges previously masked by _apply_cutoff_mask."""
        if cache is None:
            return
        edge_idx, edge_diff, edge_dist = cache
        data[properties.edge_idx] = edge_idx
        data[properties.edge_diff] = edge_diff
        data[properties.edge_dist] = edge_dist

    def module_groups(self) -> "OrderedDict[str, List[nn.Module]]":
        return OrderedDict()

    def parameter_groups(self) -> List[ParameterGroup]:
        seen: set[int] = set()
        groups: List[ParameterGroup] = []
        for name, modules in self.module_groups().items():
            params = collect_unique_parameters(modules, seen=seen)
            if params:
                groups.append(ParameterGroup(name=str(name), params=params))
        return groups

    def export_init_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement export_init_kwargs() for wrapper rebuilds."
        )


class NeuralNetworkPotential(nn.Module):
    """Base class for neural network potentials."""

    def __init__(
        self,
        representation: nn.Module,
        input_modules: Optional[List[nn.Module]] = None,
        output_modules: Optional[List[nn.Module]] = None,
        model_outputs: Optional[List[str]] = None,
        heads: Optional[list] = None,
    ) -> None:
        super().__init__()

        self.representation = representation
        self.model_outputs = model_outputs or []
        self.input_modules = CallbackModuleList(input_modules, on_register_callback=None)
        self.output_modules = CallbackModuleList(output_modules, on_register_callback=self.register_callbacks)
        self.heads = heads
        self._initialized: bool = False
        self.collect_outputs()
        self.register_callbacks()

    def forward(self, data: properties.Type) -> properties.Type:
        data = data.copy()
        for module in self.input_modules:
            data = module(data)

        data = self.representation(data)

        for module in self.output_modules:
            data = module(data)

        return self.extract_outputs(data)

    def initialize_modules(self, datamodule: LightningDataModule) -> None:
        for module in self.modules():
            if hasattr(module, "setup_from_datamodule"):
                module.setup_from_datamodule(datamodule)
            elif hasattr(module, "datamodule"):
                module.datamodule(datamodule)
        self._initialized = True

    def collect_outputs(self) -> None:
        model_outputs = set()
        for module in self.modules():
            if hasattr(module, "model_outputs") and module.model_outputs is not None:
                model_outputs.update(module.model_outputs)
        self.model_outputs = list(set(self.model_outputs + list(model_outputs)))

    def extract_outputs(self, data: properties.Type) -> properties.Type:
        if "all" in self.model_outputs:
            return data
        return {key: data[key] for key in self.model_outputs}

    def register_callbacks(self, target_module: Union[nn.Module, List[nn.Module], None] = None) -> None:
        def register_module(module):
            if hasattr(module, "update_callback"):
                module.update_callback = self.collect_outputs
            if hasattr(module, "repr_callback"):
                module.register_repr_callback(self)
            if hasattr(module, "model_outputs") and module.model_outputs is not None:
                for model_output in module.model_outputs:
                    if model_output not in self.model_outputs:
                        self.model_outputs.append(model_output)

        if target_module is None:
            for module in self.output_modules:
                register_module(module)
        elif isinstance(target_module, list):
            for module in target_module:
                register_module(module)
        else:
            register_module(target_module)

    def clone_with_representation(self, representation: nn.Module) -> "NeuralNetworkPotential":
        return self.__class__(
            representation=representation,
            input_modules=list(self.input_modules),
            output_modules=list(self.output_modules),
            model_outputs=list(self.model_outputs),
            heads=getattr(self, "heads", None),
        )

    def module_groups(self) -> "OrderedDict[str, List[nn.Module]]":
        groups: "OrderedDict[str, List[nn.Module]]" = OrderedDict()

        if len(self.input_modules) > 0:
            groups["input_modules"] = [self.input_modules]

        rep_groups_fn = getattr(self.representation, "module_groups", None)
        rep_groups = rep_groups_fn() if callable(rep_groups_fn) else None
        if rep_groups:
            for name, modules in rep_groups.items():
                groups[str(name)] = list(modules)
        else:
            groups["representation"] = [self.representation]

        if len(self.output_modules) > 0:
            groups["output_modules"] = [self.output_modules]
        return groups

    def parameter_groups(self) -> List[ParameterGroup]:
        groups: List[ParameterGroup] = []
        seen: set[int] = set()

        def append_group(name: str, items: Iterable[Any], defaults: Optional[Dict[str, Any]] = None) -> None:
            params = collect_unique_parameters(items, seen=seen)
            if params:
                group_defaults = dict(defaults) if defaults is not None else None
                groups.append(ParameterGroup(name=str(name), params=params, defaults=group_defaults))

        if len(self.input_modules) > 0:
            append_group("input_modules", [self.input_modules])

        rep_groups_fn = getattr(self.representation, "parameter_groups", None)
        rep_groups = rep_groups_fn() if callable(rep_groups_fn) else None
        if rep_groups:
            for group in rep_groups:
                append_group(group.name, group.params, group.defaults)
        else:
            rep_module_groups_fn = getattr(self.representation, "module_groups", None)
            rep_module_groups = rep_module_groups_fn() if callable(rep_module_groups_fn) else None
            if rep_module_groups:
                for name, modules in rep_module_groups.items():
                    append_group(name, modules)
            else:
                append_group("representation", [self.representation])

        if len(self.output_modules) > 0:
            append_group("output_modules", [self.output_modules])

        append_group("misc", [self])
        return groups


class CallbackModuleList(nn.ModuleList):
    def __init__(self, modules=None, on_register_callback=None):
        super().__init__()
        self.on_register_callback = on_register_callback
        if modules:
            super().extend(modules)

    def append(self, module):
        if self.on_register_callback is not None:
            self.on_register_callback(module)
        super().append(module)

    def extend(self, modules):
        module_list = list(modules)
        if self.on_register_callback is not None:
            self.on_register_callback(module_list)
        super().extend(module_list)

    def insert(self, index, module):
        if self.on_register_callback is not None:
            self.on_register_callback(module)
        super().insert(index, module)

    def __setitem__(self, idx, module):
        if self.on_register_callback is not None:
            self.on_register_callback(module)
        super().__setitem__(idx, module)


def __getattr__(name: str):
    if name == "LitNNP":
        from curator.model.lit_module import LitNNP

        return LitNNP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
