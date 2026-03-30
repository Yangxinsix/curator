from __future__ import annotations

from collections import OrderedDict
import copy
from typing import Dict, Iterable, List, Literal, Optional, Union

import torch
from torch import nn

from curator.data import properties
from curator.model.base import NeuralNetworkPotential, ParameterGroup, collect_unique_parameters


class MultiDomainPotential(NeuralNetworkPotential):
    """Domain-aware wrapper around a base neural network potential."""

    @staticmethod
    def _domain_readout_modules(representation: nn.Module) -> Optional[nn.ModuleDict]:
        domain_modules = getattr(getattr(representation, "readout", None), "domain_modules", None)
        if isinstance(domain_modules, nn.ModuleDict) and len(domain_modules) > 0:
            return domain_modules
        return None

    def _split_output_modules(self) -> tuple[List[nn.Module], List[nn.ModuleDict]]:
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
        return output_shared, output_domains

    def module_groups(self) -> "OrderedDict[str, List[nn.Module]]":
        groups: "OrderedDict[str, List[nn.Module]]" = OrderedDict()

        if len(self.input_modules) > 0:
            groups["input_modules"] = [self.input_modules]

        rep_groups_fn = getattr(self.representation, "module_groups", None)
        rep_groups = rep_groups_fn() if callable(rep_groups_fn) else None
        readout_domains = self._domain_readout_modules(self.representation)
        if rep_groups:
            for name, modules in rep_groups.items():
                if str(name) == "readout" and readout_domains is not None:
                    groups["readout_domains"] = [readout_domains]
                    continue
                groups[str(name)] = list(modules)
        else:
            groups["representation"] = [self.representation]

        output_shared, output_domains = self._split_output_modules()
        if output_domains:
            if output_shared:
                groups["output_shared"] = output_shared
            groups["output_domains"] = output_domains
        elif len(self.output_modules) > 0:
            groups["output_modules"] = [self.output_modules]
        return groups

    def parameter_groups(self) -> List[ParameterGroup]:
        groups: List[ParameterGroup] = []
        seen: set[int] = set()

        def append_group(name: str, items: Iterable[object], defaults: Optional[Dict[str, object]] = None) -> None:
            params = collect_unique_parameters(items, seen=seen)
            if params:
                group_defaults = dict(defaults) if defaults is not None else None
                groups.append(ParameterGroup(name=str(name), params=params, defaults=group_defaults))

        if len(self.input_modules) > 0:
            append_group("input_modules", [self.input_modules])

        rep_groups_fn = getattr(self.representation, "parameter_groups", None)
        rep_groups = rep_groups_fn() if callable(rep_groups_fn) else None
        readout_domains = self._domain_readout_modules(self.representation)
        if rep_groups:
            for group in rep_groups:
                if str(group.name) == "readout" and readout_domains is not None:
                    append_group("readout_domains", [readout_domains], group.defaults)
                    continue
                append_group(group.name, group.params, group.defaults)
        else:
            rep_module_groups_fn = getattr(self.representation, "module_groups", None)
            rep_module_groups = rep_module_groups_fn() if callable(rep_module_groups_fn) else None
            if rep_module_groups:
                for name, modules in rep_module_groups.items():
                    if str(name) == "readout" and readout_domains is not None:
                        append_group("readout_domains", [readout_domains])
                        continue
                    append_group(name, modules)
            else:
                append_group("representation", [self.representation])

        output_shared, output_domains = self._split_output_modules()
        if output_domains:
            if output_shared:
                append_group("output_shared", output_shared)
            append_group("output_domains", output_domains)
        elif len(self.output_modules) > 0:
            append_group("output_modules", [self.output_modules])

        append_group("misc", [self])
        return groups

    def forward(
        self,
        data: properties.Type,
        force_domain: Optional[Union[str, int]] = None,
    ) -> properties.Type:
        if force_domain is None:
            return super().forward(data)
        data = data.copy()
        data[properties.domain] = torch.tensor(
            [int(force_domain)],
            dtype=torch.long,
            device=data[properties.n_atoms].device,
        )
        return super().forward(data)


def apply_domain_set(
    model: torch.nn.Module,
    domains,
    mode: Literal["extend", "replace"] = "extend",
    template_domain: str = "0",
    init_strategy: Literal["random", "copy"] = "random",
    logger=None,
) -> int:
    domains = [str(domain) for domain in list(domains or [])]
    if not domains:
        return 0
    if mode not in ("extend", "replace"):
        if logger:
            logger.warning("Unknown mode=%s; expected 'extend' or 'replace'.", mode)
        return 0
    if init_strategy not in ("random", "copy"):
        if logger:
            logger.warning("Unknown init_strategy=%s; expected 'random' or 'copy'.", init_strategy)
        init_strategy = "random"

    def reset_params(module: torch.nn.Module) -> None:
        for submodule in module.modules():
            reset = getattr(submodule, "reset_parameters", None)
            if callable(reset):
                reset()

    updated = 0
    for module in model.modules():
        domain_modules = getattr(module, "domain_modules", None)
        if domain_modules is None or not hasattr(domain_modules, "items"):
            continue

        template_key = str(template_domain)
        template_module = None
        if template_key in domain_modules:
            template_module = domain_modules[template_key]
        elif len(domain_modules) > 0:
            template_key, template_module = next(iter(domain_modules.items()))

        if mode == "replace":
            for key in list(domain_modules.keys()):
                if key not in domains:
                    del domain_modules[key]

        for domain in domains:
            if domain in domain_modules:
                continue
            if template_module is None:
                if logger:
                    logger.warning(
                        "No template domain found for module %s; skipping %s.",
                        module.__class__.__name__,
                        domain,
                    )
                continue
            new_module = copy.deepcopy(template_module)
            if init_strategy == "random":
                reset_params(new_module)
            domain_modules[domain] = new_module
            updated += 1

        if hasattr(module, "domains"):
            if mode == "replace":
                module.domains = domains[:]
            else:
                existing = [str(domain) for domain in (getattr(module, "domains") or [])]
                for domain in domains:
                    if domain not in existing:
                        existing.append(domain)
                module.domains = existing
    return updated


__all__ = [
    "MultiDomainPotential",
    "apply_domain_set",
]
