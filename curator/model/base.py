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

        def maybe(name, value):
            return {name: value} if name in sig.parameters and value is not None else {}

        init_kwargs = dict(kwargs)
        init_kwargs.update(maybe("heads", heads))

        return readout(**init_kwargs)  # type: ignore[arg-type]

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

    def forward(self, data: properties.Type, force_domain: Optional[Union[str, int]] = None) -> properties.Type:
        data = data.copy()
        if force_domain is not None:
            dom = torch.tensor([int(force_domain)], dtype=torch.long, device=data[properties.n_atoms].device)
            data[properties.domain] = dom
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

    def module_groups(self) -> "OrderedDict[str, List[nn.Module]]":
        groups: "OrderedDict[str, List[nn.Module]]" = OrderedDict()

        def normalize_modules(modules: Any) -> List[nn.Module]:
            if modules is None:
                return []
            if isinstance(modules, nn.Module):
                return [modules]
            return [module for module in list(modules) if module is not None]

        if len(self.input_modules) > 0:
            groups["input_modules"] = [self.input_modules]

        representation_groups: "OrderedDict[str, List[nn.Module]]" = OrderedDict()
        rep_groups_fn = getattr(self.representation, "module_groups", None)
        rep_groups = rep_groups_fn() if callable(rep_groups_fn) else None
        if rep_groups:
            for name, modules in rep_groups.items():
                module_list = normalize_modules(modules)
                if module_list:
                    representation_groups[str(name)] = module_list
        else:
            representation_groups["representation"] = [self.representation]

        readout = getattr(self.representation, "readout", None)
        readout_domain_modules = getattr(readout, "domain_modules", None)
        if isinstance(readout_domain_modules, nn.ModuleDict) and len(readout_domain_modules) > 0:
            representation_groups.pop("readout", None)
            readout_shared = [
                child for name, child in readout.named_children() if name != "domain_modules"
            ]
            if readout_shared:
                representation_groups["readout_shared"] = readout_shared
            representation_groups["readout_domains"] = [readout_domain_modules]

        groups.update(representation_groups)

        if len(self.output_modules) > 0:
            output_shared: List[nn.Module] = []
            output_domains: List[nn.ModuleDict] = []
            for module in self.output_modules:
                domain_modules = getattr(module, "domain_modules", None)
                if isinstance(domain_modules, nn.ModuleDict) and len(domain_modules) > 0:
                    output_shared.extend(
                        child for name, child in module.named_children() if name != "domain_modules"
                    )
                    output_domains.append(domain_modules)
                else:
                    output_shared.append(module)
            if output_domains:
                if output_shared:
                    groups["output_shared"] = output_shared
                groups["output_domains"] = output_domains
            else:
                groups["output_modules"] = [self.output_modules]
        return groups

    def parameter_groups(self) -> List[ParameterGroup]:
        groups: List[ParameterGroup] = []
        seen: set[int] = set()
        module_groups = self.module_groups()
        consumed_names: set[str] = set()

        def append_group(name: str, items: Iterable[Any], defaults: Optional[Dict[str, Any]] = None) -> None:
            params = collect_unique_parameters(items, seen=seen)
            if params:
                group_defaults = dict(defaults) if defaults is not None else None
                groups.append(ParameterGroup(name=str(name), params=params, defaults=group_defaults))

        if "input_modules" in module_groups:
            append_group("input_modules", module_groups["input_modules"])
            consumed_names.add("input_modules")

        rep_groups_fn = getattr(self.representation, "parameter_groups", None)
        rep_groups = rep_groups_fn() if callable(rep_groups_fn) else None
        if rep_groups:
            for group in rep_groups:
                group_name = str(group.name)
                group_defaults = group.defaults
                if group_name == "readout" and "readout_domains" in module_groups:
                    if "readout_shared" in module_groups:
                        append_group("readout_shared", module_groups["readout_shared"], group_defaults)
                        consumed_names.add("readout_shared")
                    append_group("readout_domains", module_groups["readout_domains"], group_defaults)
                    consumed_names.add("readout_domains")
                    consumed_names.add("readout")
                    continue
                append_group(group_name, group.params, group_defaults)
                consumed_names.add(group_name)
        else:
            if "representation" in module_groups:
                append_group("representation", module_groups["representation"])
                consumed_names.add("representation")

        for group_name, modules in module_groups.items():
            if group_name in consumed_names:
                continue
            append_group(group_name, modules)
            consumed_names.add(group_name)

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
