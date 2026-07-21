from __future__ import annotations

import copy
import importlib
import io
import logging
from functools import partial
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.serialization as torch_serialization
from ase.data import chemical_symbols

from curator.data import properties
from curator.data.properties import HeadConfig
from curator.layer import (
    AgnesiTransform,
    AtomwiseNN,
    AtomwiseReduce,
    GlobalRescaleShift,
    GradientOutput,
    MACEAtomwiseNN,
    MultiDomainAtomwiseNN,
    MultiDomainMACEAtomwiseNN,
    PairRepulsionEnergy,
    PairwiseDistance,
    RealAgnosticDensityInteractionBlock,
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    SoftTransform,
    ZBLBasis,
)
from curator.layer._rescale import MultiDomainRescaleShift

from .base import NeuralNetworkPotential
from .mace import MACE
from .multi_domain import MultiDomainPotential, apply_domain_set
from .nequip import Nequip

logger = logging.getLogger(__name__)

def _load_state_dict_by_shape(
    module: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    strict_shapes: bool = False,
    label: Optional[str] = None,
) -> Dict[str, List[str]]:
    current = module.state_dict()
    filtered = {}
    skipped = []
    for name, param in state_dict.items():
        if name in current and current[name].shape == param.shape:
            filtered[name] = param
        else:
            skipped.append(name)
    module.load_state_dict(filtered, strict=False)
    if strict_shapes and skipped:
        context = f" for {label}" if label else ""
        raise ValueError(
            f"Failed to load{context}: {len(skipped)} tensor(s) had missing or mismatched shapes: "
            + ", ".join(skipped[:8])
            + ("..." if len(skipped) > 8 else "")
        )
    return {"loaded": list(filtered.keys()), "skipped": skipped}


def _coerce_state_tensor_shape(
    source_tensor: torch.Tensor,
    target_shape: torch.Size,
) -> Optional[torch.Tensor]:
    if source_tensor.shape == target_shape:
        return source_tensor
    if (
        source_tensor.dim() + 1 == len(target_shape)
        and target_shape[0] == 1
        and source_tensor.shape == target_shape[1:]
    ):
        return source_tensor.unsqueeze(0)
    if (
        source_tensor.dim() == len(target_shape) + 1
        and source_tensor.shape[0] == 1
        and source_tensor.shape[1:] == target_shape
    ):
        return source_tensor.squeeze(0)
    return None


def _load_energy_readout_from_source(
    target_model: torch.nn.Module,
    source_model: torch.nn.Module,
) -> int:
    def unwrap_single_domain_module(module):
        domain_modules = getattr(module, "domain_modules", None)
        if not isinstance(domain_modules, torch.nn.ModuleDict) or len(domain_modules) == 0:
            return module
        if len(domain_modules) != 1:
            return None
        return next(iter(domain_modules.values()))

    target_readout = unwrap_single_domain_module(getattr(getattr(target_model, "representation", None), "readout", None))
    source_readout = unwrap_single_domain_module(getattr(getattr(source_model, "representation", None), "readout", None))
    if target_readout is None or source_readout is None:
        return 0
    if not isinstance(target_readout, AtomwiseNN) or not isinstance(source_readout, AtomwiseNN):
        return 0
    if getattr(source_readout, "separate_heads", False) or not getattr(target_readout, "separate_heads", False):
        return 0

    target_modules: List[torch.nn.Module]
    source_modules: List[torch.nn.Module]
    if isinstance(target_readout, MACEAtomwiseNN) or isinstance(source_readout, MACEAtomwiseNN):
        if not isinstance(target_readout, MACEAtomwiseNN) or not isinstance(source_readout, MACEAtomwiseNN):
            return 0
        if properties.energy not in getattr(target_readout, "readouts_by_head", {}):
            return 0
        if properties.energy not in getattr(target_readout, "final_readouts", {}):
            return 0
        target_modules = list(target_readout.readouts_by_head[properties.energy]) + [
            target_readout.final_readouts[properties.energy]
        ]
        source_modules = list(source_readout.readouts)
    else:
        head_modules = getattr(target_readout, "head_modules", None)
        if not isinstance(head_modules, torch.nn.ModuleDict) or properties.energy not in head_modules:
            return 0
        shared_mlp = getattr(target_readout, "shared_mlp", None)
        shared_layers = list(shared_mlp) if isinstance(shared_mlp, torch.nn.Sequential) else []
        source_layers = list(getattr(source_readout, "readout_mlp", []))
        if not source_layers:
            return 0
        target_modules = shared_layers + [head_modules[properties.energy]]
        source_modules = source_layers

    if len(target_modules) != len(source_modules):
        return 0

    loaded = 0
    for target_module, source_module in zip(target_modules, source_modules):
        loaded += len(_load_state_dict_by_shape(target_module, source_module.state_dict())["loaded"])
    return loaded


def _load_pretrained_weights_from_model_impl(
    target_model: torch.nn.Module,
    source_model: torch.nn.Module,
) -> int:
    loaded = len(_load_state_dict_by_shape(target_model, source_model.state_dict())["loaded"])
    loaded += _load_energy_readout_from_source(target_model, source_model)
    return loaded

def _available_model_domains(model: torch.nn.Module) -> List[str]:
    readout = getattr(getattr(model, "representation", None), "readout", None)
    domain_modules = getattr(readout, "domain_modules", None)
    if isinstance(domain_modules, torch.nn.ModuleDict) and len(domain_modules) > 0:
        return [str(domain) for domain in domain_modules.keys()]

    for module in getattr(model, "output_modules", []):
        domain_modules = getattr(module, "domain_modules", None)
        if isinstance(domain_modules, torch.nn.ModuleDict) and len(domain_modules) > 0:
            return [str(domain) for domain in domain_modules.keys()]
    return []


def _resolve_domain_selector(
    available: List[str],
    selector: Optional[Union[str, int]],
) -> str:
    if not available:
        raise ValueError("At least one domain must be available.")

    if selector is None:
        return available[-1]
    if isinstance(selector, int):
        if selector < 0 or selector >= len(available):
            raise ValueError(
                f"Domain index {selector} out of range for available domains {available}."
            )
        return available[selector]

    token = str(selector).strip()
    if token == "" or token == "last":
        return available[-1]
    if token in available:
        return token

    try:
        index = int(token)
    except ValueError as exc:
        raise ValueError(
            f"Domain selector {token!r} not found in available domains {available}."
        ) from exc

    if 0 <= index < len(available):
        return available[index]
    raise ValueError(
        f"Domain selector {token!r} not found in available domains {available}."
    )


def _resolve_model_domains(
    model: torch.nn.Module,
    domains: Optional[Union[str, int, List[Union[str, int]]]] = None,
) -> List[str]:
    available = _available_model_domains(model)
    if not available:
        raise TypeError(f"Expected a multi-domain model, got {type(model)}.")
    if domains is None:
        return [available[-1]]

    raw_domains: List[Union[str, int]]
    if isinstance(domains, (str, int)):
        raw_domains = [domains]
    else:
        raw_domains = list(domains)

    selected: List[str] = []
    for raw in raw_domains:
        if isinstance(raw, str) and raw.strip() == "":
            continue
        resolved = _resolve_domain_selector(available, raw)
        if resolved not in selected:
            selected.append(resolved)

    if not selected:
        raise ValueError("At least one domain must be selected.")
    return selected

def _convert_single_to_multi_domain_impl(
    model: torch.nn.Module,
    *,
    template_domain: str = "0",
    domains: Optional[List[Union[str, int]]] = None,
    heads_by_domain: Optional[Dict[str, List[Any]]] = None,
) -> MultiDomainPotential:
    raw_domains = domains if domains is not None else [template_domain]
    target_domains = [str(domain).strip() for domain in raw_domains]
    target_domains = [domain for idx, domain in enumerate(target_domains) if domain and domain not in target_domains[:idx]]
    if not target_domains:
        raise ValueError("At least one target domain must be provided.")

    if isinstance(model, MultiDomainPotential):
        if target_domains == _available_model_domains(model):
            return model
        converted = copy.deepcopy(model)
        template = _resolve_domain_selector(_available_model_domains(model), template_domain)
        apply_domain_set(
            converted,
            target_domains,
            mode="replace",
            template_domain=template,
            init_strategy="copy",
        )
        converted._initialized = getattr(model, "_initialized", False)
        converted.train(model.training)
        return converted

    normalized_heads_by_domain = (
        {str(domain): list(heads) for domain, heads in heads_by_domain.items()}
        if heads_by_domain
        else None
    )
    representation = getattr(model, "representation", None)
    if representation is None:
        raise TypeError("Expected a model with a `representation` attribute.")

    readout = getattr(representation, "readout", None)
    domain_modules = getattr(readout, "domain_modules", None)
    if not isinstance(domain_modules, torch.nn.ModuleDict) or len(domain_modules) == 0:
        export_fn = getattr(representation, "export_init_kwargs", None)
        if not callable(export_fn):
            raise TypeError(
                f"{representation.__class__.__name__} does not implement export_init_kwargs()."
            )

        readout_cls = (
            MultiDomainMACEAtomwiseNN
            if representation.__class__.__name__ == "MACE"
            else MultiDomainAtomwiseNN
        )
        rep_kwargs = dict(export_fn())
        rep_kwargs.setdefault("heads", list(getattr(representation, "heads", None) or []))
        template_heads = list(getattr(readout, "heads", None) or [])
        domain_heads = {
            domain: list((normalized_heads_by_domain or {}).get(domain, template_heads))
            for domain in target_domains
        }
        readout_kwargs: Dict[str, Any] = {"domains": target_domains}
        if any(domain_heads.values()):
            readout_kwargs["heads_by_domain"] = domain_heads
        rep_kwargs["readout"] = partial(readout_cls, **readout_kwargs)

        new_representation = representation.__class__(**rep_kwargs)
        rep_state_dict = {}
        for key, value in representation.state_dict().items():
            if key.startswith("readout.domain_modules."):
                rep_state_dict[key] = value
            elif key.startswith("readout."):
                suffix = key[len("readout.") :]
                for domain in target_domains:
                    rep_state_dict[f"readout.domain_modules.{domain}.{suffix}"] = value
        for name, child in representation.named_children():
            if not name.startswith("readout"):
                setattr(new_representation, name, child)
        new_representation.train(representation.training)
        new_representation.load_state_dict(rep_state_dict, strict=False)
        base_model = model.clone_with_representation(new_representation)
        base_model._initialized = getattr(model, "_initialized", False)
        base_model.train(model.training)
    else:
        base_model = model

    output_modules = []
    for module in getattr(base_model, "output_modules", []):
        if isinstance(module, GlobalRescaleShift):
            multi_domain_module = MultiDomainRescaleShift(heads=list(module.heads))
            for domain in target_domains:
                multi_domain_module.domain_modules[domain] = copy.deepcopy(module)
            multi_domain_module.train(module.training)
            output_modules.append(multi_domain_module)
        else:
            output_modules.append(module)

    converted = MultiDomainPotential(
        representation=base_model.representation,
        input_modules=list(base_model.input_modules),
        output_modules=output_modules,
        model_outputs=list(base_model.model_outputs),
        heads=getattr(base_model, "heads", None),
    )
    current_domains = _available_model_domains(converted)
    if current_domains != target_domains:
        apply_domain_set(
            converted,
            target_domains,
            mode="replace",
            template_domain=current_domains[0] if current_domains else target_domains[0],
            init_strategy="copy",
        )
    converted._initialized = getattr(model, "_initialized", False)
    converted.train(model.training)
    if not isinstance(converted, MultiDomainPotential):
        raise TypeError(f"Expected MultiDomainPotential after promotion, got {type(converted)}")
    return converted


def _convert_multi_to_single_domain_impl(
    model: torch.nn.Module,
    *,
    domain: Optional[Union[str, int]] = "last",
) -> NeuralNetworkPotential:
    selected_domain = _resolve_model_domains(model, domains=domain)[0]

    representation = copy.deepcopy(model.representation)
    readout_domains = getattr(getattr(representation, "readout", None), "domain_modules", None)
    if not isinstance(readout_domains, torch.nn.ModuleDict) or len(readout_domains) == 0 or selected_domain not in readout_domains:
        raise TypeError("Expected multi-domain representation readout for single-domain conversion.")
    representation.readout = copy.deepcopy(readout_domains[selected_domain])

    output_modules: List[torch.nn.Module] = []
    for module in getattr(model, "output_modules", []):
        domain_modules = getattr(module, "domain_modules", None)
        if not isinstance(domain_modules, torch.nn.ModuleDict) or len(domain_modules) == 0:
            output_modules.append(copy.deepcopy(module))
            continue
        if selected_domain not in domain_modules:
            raise ValueError(
                f"Domain {selected_domain!r} not found in output module {module.__class__.__name__}."
            )
        output_modules.append(copy.deepcopy(domain_modules[selected_domain]))

    single_model = NeuralNetworkPotential(
        representation=representation,
        input_modules=[copy.deepcopy(module) for module in getattr(model, "input_modules", [])],
        output_modules=output_modules,
        model_outputs=list(getattr(model, "model_outputs", [])),
        heads=getattr(model, "heads", None),
    )
    single_model._initialized = getattr(model, "_initialized", False)
    single_model.train(model.training)
    return single_model


def _convert_multi_to_selected_domains_impl(
    model: torch.nn.Module,
    *,
    domains: Optional[Union[str, int, List[Union[str, int]]]] = None,
) -> Union[NeuralNetworkPotential, MultiDomainPotential]:
    selected_domains = _resolve_model_domains(model, domains=domains)
    if len(selected_domains) == 1:
        return convert_multi_to_single_domain(model, domain=selected_domains[0])

    if not isinstance(model, MultiDomainPotential):
        raise TypeError(f"Expected MultiDomainPotential, got {type(model)}")

    converted = copy.deepcopy(model)
    from .multi_domain import apply_domain_set

    apply_domain_set(
        converted,
        selected_domains,
        mode="replace",
        template_domain=selected_domains[0],
        init_strategy="copy",
    )
    converted._initialized = getattr(model, "_initialized", False)
    converted.train(model.training)
    return converted

def _load_external_mapping(
    module: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    label: str,
    required_prefixes: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    result = _load_state_dict_by_shape(module, state_dict)
    loaded = result["loaded"]
    if required_prefixes:
        missing = [
            prefix
            for prefix in required_prefixes
            if not any(key == prefix or key.startswith(prefix) for key in loaded)
        ]
        if missing:
            raise ValueError(
                f"Incomplete {label} mapping; missing required tensors for: {', '.join(missing)}."
            )
    elif not loaded:
        raise ValueError(f"Failed to map {label}; no compatible tensors were loaded.")

    skipped = result["skipped"]
    if skipped:
        logger.warning(
            "Partial external conversion load for %s: loaded=%d skipped=%d first_skipped=%s",
            label,
            len(loaded),
            len(skipped),
            ", ".join(skipped[:5]),
        )
    return result


def _load_official_mace_energy_head(
    target_readout,
    mace_model,
    *,
    head_idx: int,
    num_mace_heads: int,
    mlp_count_irreps: int,
) -> None:
    if getattr(target_readout, "separate_heads", False):
        target_hidden_readouts = list(target_readout.readouts_by_head[properties.energy])
        target_final_readout = target_readout.final_readouts.get(properties.energy)
    else:
        target_hidden_readouts = list(target_readout.readouts[:-1])
        target_final_readout = target_readout.readouts[-1]
    if len(target_hidden_readouts) != len(mace_model.readouts) - 1:
        raise ValueError(
            "Curator MACE readout depth does not match official MACE readout depth."
        )
    for idx, src_readout in enumerate(mace_model.readouts):
        state_dict = {key: value.detach().cpu() for key, value in src_readout.state_dict().items()}
        if num_mace_heads > 1:
            sliced: Dict[str, torch.Tensor] = {}
            for name, param in state_dict.items():
                if "linear_1.weight" in name and mlp_count_irreps:
                    sliced[name] = param.reshape(-1, num_mace_heads, mlp_count_irreps)[:, head_idx, :].flatten()
                elif "linear_2.weight" in name:
                    sliced[name] = (
                        param.reshape(num_mace_heads, -1, num_mace_heads)[head_idx, :, head_idx].flatten()
                        / (num_mace_heads**0.5)
                    )
                elif "linear.weight" in name:
                    sliced[name] = param.reshape(-1, num_mace_heads)[:, head_idx].flatten()
                elif "output_mask" in name and param.numel() == num_mace_heads:
                    sliced[name] = param[head_idx : head_idx + 1]
                elif "bias" in name and param.numel() > 0 and param.numel() % num_mace_heads == 0:
                    sliced[name] = param.reshape(-1, num_mace_heads)[:, head_idx].flatten()
                else:
                    sliced[name] = param
            state_dict = sliced
        if idx == len(mace_model.readouts) - 1:
            if target_final_readout is None:
                raise ValueError("Target Curator MACE readout does not expose an energy final branch.")
            dst_readout = target_final_readout
            state_dict = {
                key.replace("linear_1.", "0.linear.").replace("linear_2.", "1.linear."): value
                for key, value in state_dict.items()
            }
        else:
            dst_readout = target_hidden_readouts[idx]
        module_state = dst_readout.state_dict()
        adapted_state: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key not in module_state:
                adapted_state[key] = value
                continue
            coerced = _coerce_state_tensor_shape(value, module_state[key].shape)
            adapted_state[key] = value if coerced is None else coerced
        _load_external_mapping(
            dst_readout,
            adapted_state,
            label=f"official MACE readout {idx}",
        )


def _create_curator_mace_representation(
    mace_model,
) -> MACE:
    interaction_map = {
        "RealAgnosticDensityInteractionBlock": RealAgnosticDensityInteractionBlock,
        "RealAgnosticDensityResidualInteractionBlock": RealAgnosticDensityResidualInteractionBlock,
        "RealAgnosticInteractionBlock": RealAgnosticInteractionBlock,
        "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
    }
    num_mace_heads = max(len(list(getattr(mace_model, "heads", []) or [])), 1)
    first_interaction_name = mace_model.interactions[0].__class__.__name__
    if first_interaction_name not in interaction_map:
        raise NotImplementedError(
            f"Unsupported MACE interaction block '{first_interaction_name}' in first interaction."
        )
    interaction_cls_first = interaction_map[first_interaction_name]
    if len(mace_model.interactions) > 1:
        interaction_name = mace_model.interactions[1].__class__.__name__
        if interaction_name not in interaction_map:
            raise NotImplementedError(
                f"Unsupported MACE interaction block '{interaction_name}' in interaction 1."
            )
        interaction_cls = interaction_map[interaction_name]
    else:
        interaction_cls = interaction_cls_first
    for idx, interaction in enumerate(mace_model.interactions[2:], start=2):
        interaction_name = interaction.__class__.__name__
        if interaction_name not in interaction_map:
            raise NotImplementedError(
                f"Unsupported MACE interaction block '{interaction_name}' in interaction {idx}."
            )
        if interaction_map[interaction_name] is not interaction_cls:
            raise NotImplementedError(
                "Curator MACE conversion expects all non-first interaction blocks to share "
                f"one class, but interaction 1 is '{mace_model.interactions[1].__class__.__name__}' "
                f"and interaction {idx} is '{interaction_name}'."
            )

    distance_transform = None
    transform = getattr(getattr(mace_model, "radial_embedding", None), "distance_transform", None)
    if transform is not None:
        transform_cls = {
            "AgnesiTransform": AgnesiTransform,
            "SoftTransform": SoftTransform,
        }.get(transform.__class__.__name__)
        if transform_cls is None:
            raise ValueError(
                f"Unsupported distance_transform '{transform.__class__.__name__}' in MACE model"
            )
        distance_transform = transform_cls()
        distance_transform.load_state_dict(
            {key: value.detach().cpu() for key, value in transform.state_dict().items()},
            strict=False,
        )

    readout_head = HeadConfig(
        key=properties.energy,
        is_atomwise=True,
        reduction="sum",
        atomwise_key=properties.atomic_energy,
        write_atomwise=False,
        dim=1,
    )
    correlation_list = [
        product.symmetric_contractions.contractions[0].correlation
        for product in mace_model.products
    ]
    try:
        from e3nn import o3

        total_mlp_irreps = o3.Irreps(getattr(mace_model.readouts[-1], "hidden_irreps", "16x0e"))
        total_mul = int(total_mlp_irreps.count(o3.Irrep(0, 1)))
        mlp_irreps = o3.Irreps(
            f"{total_mul // num_mace_heads if num_mace_heads > 0 and total_mul % num_mace_heads == 0 else total_mul}x0e"
        )
    except Exception:
        mlp_irreps = getattr(mace_model.readouts[-1], "hidden_irreps", None)

    curator_mace = MACE(
        cutoff=float(mace_model.r_max),
        num_interactions=len(mace_model.interactions),
        correlation=correlation_list,
        species=[chemical_symbols[i] for i in mace_model.atomic_numbers],
        hidden_irreps=mace_model.interactions[0].hidden_irreps,
        edge_sh_irreps=mace_model.spherical_harmonics.irreps_out,
        avg_num_neighbors=mace_model.interactions[0].avg_num_neighbors,
        MLP_irreps=mlp_irreps,
        num_basis=len(mace_model.radial_embedding.bessel_fn.bessel_weights),
        power=float(mace_model.radial_embedding.cutoff_fn.p),
        interaction_cls=interaction_cls,
        interaction_cls_first=interaction_cls_first,
        distance_transform=distance_transform,
        filter_forbidden_irreps=False,
        readout=partial(
            MACEAtomwiseNN,
            heads=[readout_head],
            activation=torch.nn.functional.silu,
        ),
        heads=[readout_head],
    )
    curator_mace.embeddings.radial_basis.basis.load_state_dict(
        {
            key: value.detach().cpu()
            for key, value in mace_model.radial_embedding.bessel_fn.state_dict().items()
        },
        strict=False,
    )
    curator_mace.embeddings.chemical_embedding.linear.load_state_dict(
        {
            key: value.detach().cpu()
            for key, value in mace_model.node_embedding.linear.state_dict().items()
        },
        strict=False,
    )
    for idx in range(len(mace_model.interactions)):
        curator_mace.interactions[idx].avg_num_neighbors = torch.tensor(
            mace_model.interactions[idx].avg_num_neighbors
        )
        _load_state_dict_by_shape(
            curator_mace.interactions[idx],
            {
                key: value.detach().cpu()
                for key, value in mace_model.interactions[idx].state_dict().items()
            },
            strict_shapes=True,
            label=f"official MACE interaction {idx}",
        )
        _load_state_dict_by_shape(
            curator_mace.products[idx],
            {key: value.detach().cpu() for key, value in mace_model.products[idx].state_dict().items()},
        )
    return curator_mace


def _build_curator_mace_model(
    mace_model,
    domain_to_head_idx: Dict[str, int],
    *,
    mode: Literal["single", "multi"],
) -> torch.nn.Module:
    heads = list(getattr(mace_model, "heads", []) or [])
    num_mace_heads = len(heads) if heads else 1
    curator_mace = _create_curator_mace_representation(mace_model)
    try:
        from e3nn import o3

        mlp_count_irreps = int(o3.Irreps(curator_mace.MLP_irreps).count(o3.Irrep(0, 1)))
    except Exception:
        mlp_count_irreps = 0

    def build_energy_head(head_idx: int) -> HeadConfig:
        scale_all = torch.atleast_1d(
            getattr(mace_model.scale_shift, "scale", torch.tensor([1.0]))
        ).detach().cpu()
        shift_all = torch.atleast_1d(
            getattr(mace_model.scale_shift, "shift", torch.tensor([0.0]))
        ).detach().cpu()
        atomic_energies = getattr(mace_model.atomic_energies_fn, "atomic_energies", None)
        if atomic_energies is None:
            atomic_energies = torch.zeros((len(mace_model.atomic_numbers),), dtype=torch.get_default_dtype())
        elif torch.is_tensor(atomic_energies) and atomic_energies.ndim > 1 and heads:
            if atomic_energies.shape[0] == len(heads):
                atomic_energies = atomic_energies[head_idx]
            elif atomic_energies.shape[1] == len(heads):
                atomic_energies = atomic_energies[:, head_idx]
            else:
                atomic_energies = atomic_energies[0]
        atomic_energies = (
            atomic_energies if torch.is_tensor(atomic_energies) else torch.as_tensor(atomic_energies)
        ).detach().cpu().squeeze()

        def pick(values: torch.Tensor) -> float:
            flat = values.reshape(-1)
            if heads and flat.numel() == len(heads):
                return float(flat[head_idx].item())
            return float(flat[0].item())

        return HeadConfig(
            key=properties.energy,
            is_atomwise=True,
            reduction="sum",
            atomwise_key=properties.atomic_energy,
            write_atomwise=False,
            dim=1,
            scale_by=pick(scale_all),
            shift_by=pick(shift_all),
            atomwise_shift=False,
            atomwise_normalization=True,
            per_species_shift={
                int(z): float(e)
                for z, e in zip(mace_model.atomic_numbers, atomic_energies.reshape(-1).tolist())
            },
        )

    template_head_idx = next(iter(domain_to_head_idx.values()))
    if len(mace_model.readouts) != len(curator_mace.interactions):
        raise NotImplementedError(
            "Curator MACE conversion currently supports official MACE checkpoints with "
            "one readout per interaction. Use the official ASE calculator path for this checkpoint."
        )
    _load_official_mace_energy_head(
        curator_mace.readout,
        mace_model,
        head_idx=template_head_idx,
        num_mace_heads=num_mace_heads,
        mlp_count_irreps=mlp_count_irreps,
    )

    output_modules: List[torch.nn.Module] = []
    pair_fn = getattr(mace_model, "pair_repulsion_fn", None)
    if pair_fn is not None:
        basis = ZBLBasis()
        basis.load_state_dict(
            {key: value.detach().cpu() for key, value in pair_fn.state_dict().items()},
            strict=False,
        )
        output_modules.append(PairRepulsionEnergy(basis, atomic_numbers=curator_mace.atomic_numbers))
    output_modules.append(GlobalRescaleShift(heads=[build_energy_head(template_head_idx)]))
    output_modules.append(
        GradientOutput(
            model_outputs=[properties.energy, properties.forces],
            grad_on_edge_diff=True,
            grad_on_positions=False,
        )
    )

    model = NeuralNetworkPotential(
        input_modules=[PairwiseDistance()],
        representation=curator_mace,
        output_modules=output_modules,
    )
    try:
        model = model.to(dtype=next(mace_model.parameters()).dtype)
    except StopIteration:
        pass
    template_buffer = io.BytesIO()
    torch.save(copy.deepcopy(mace_model).cpu(), template_buffer)
    model.representation._official_mace_template_bytes = template_buffer.getvalue()

    if mode == "single":
        return model

    model = convert_single_to_multi_domain(model)
    apply_domain_set(
        model,
        domain_to_head_idx.keys(),
        mode="replace",
        template_domain="0",
        init_strategy="copy",
    )
    if not isinstance(model, MultiDomainPotential):
        raise TypeError(f"Expected MultiDomainPotential after promotion, got {type(model)}")
    readout_domains = getattr(getattr(model.representation, "readout", None), "domain_modules", None)
    if not isinstance(readout_domains, torch.nn.ModuleDict) or len(readout_domains) == 0:
        raise TypeError("Expected promoted MACE readout to expose domain_modules.")

    for domain, head_idx in domain_to_head_idx.items():
        if str(domain) not in readout_domains:
            raise ValueError(
                f"Converted MACE model is missing promoted domain {domain!r}; available={list(readout_domains.keys())}"
            )
        _load_official_mace_energy_head(
            readout_domains[str(domain)],
            mace_model,
            head_idx=head_idx,
            num_mace_heads=num_mace_heads,
            mlp_count_irreps=mlp_count_irreps,
        )

    for module in model.output_modules:
        if isinstance(module, MultiDomainRescaleShift):
            for domain, head_idx in domain_to_head_idx.items():
                module.domain_modules[str(domain)] = GlobalRescaleShift(
                    heads=[build_energy_head(head_idx)]
                )
    model.representation._official_mace_template_bytes = template_buffer.getvalue()
    return model


def _create_model_from_mace_impl(
    mace_model,
    head: Optional[Union[str, int]] = None,
    *,
    mode: Literal["auto", "single", "multi"] = "auto",
):
    heads = list(getattr(mace_model, "heads", []) or [])
    if mode not in {"auto", "single", "multi"}:
        raise ValueError(f"Unsupported MACE conversion mode {mode!r}.")
    if mode == "auto":
        mode = "multi" if head is None and len(heads) > 1 else "single"

    if head is None:
        if len(heads) <= 1:
            domain_to_head_idx = {"0": 0}
        else:
            domain_to_head_idx: Dict[str, int] = {}
            for idx, official_head in enumerate(heads):
                domain = str(official_head).strip()
                if not domain or domain in domain_to_head_idx:
                    domain = str(idx)
                domain_to_head_idx[domain] = idx
    elif len(heads) <= 1:
        domain_to_head_idx = {"0": 0}
    elif isinstance(head, int):
        if head < 0 or head >= len(heads):
            raise ValueError(f"Head index {head} out of range for heads={heads}")
        domain_to_head_idx = {"0": int(head)}
    else:
        head_name = str(head)
        if head_name not in heads:
            raise ValueError(f"Head {head_name} not found in heads={heads}")
        domain_to_head_idx = {"0": heads.index(head_name)}

    return _build_curator_mace_model(
        mace_model,
        domain_to_head_idx,
        mode=mode,
    )


def _build_mace_from_curator_impl(curator_model):
    try:
        from mace.modules import blocks as mace_blocks
        from mace.modules import models as mace_models
    except Exception as exc:
        raise RuntimeError(
            "Failed to import mace modules; converting to original MACE requires mace "
            "and its cuequivariance dependencies."
        ) from exc

    if isinstance(curator_model, NeuralNetworkPotential):
        repr_model = curator_model.representation
        output_modules = list(curator_model.output_modules)
    else:
        repr_model = curator_model
        output_modules = []

    if not isinstance(repr_model, MACE):
        raise TypeError("Provided model is not a Curator MACE representation.")

    multi_domain_error = "curator_to_mace only supports single-domain Curator MACE models. Multi-domain export is not supported."

    def unwrap_single_domain(module, *, label: str):
        domain_modules = getattr(module, "domain_modules", None)
        if not isinstance(domain_modules, torch.nn.ModuleDict):
            return module
        if len(domain_modules) == 1:
            return next(iter(domain_modules.values()))
        raise NotImplementedError(f"{multi_domain_error} Offending {label}: {type(module).__name__}.")

    def pick_scalar(value):
        if isinstance(value, list):
            return value[0] if value else 0.0
        if torch.is_tensor(value):
            flat = value.detach().cpu().reshape(-1)
            return flat[0].item() if flat.numel() > 0 else 0.0
        return value

    def resolve_atomic_numbers() -> List[int]:
        atomic_numbers = list(range(repr_model.embeddings.onehot_embedding.num_elements))
        mapper = getattr(repr_model.embeddings.onehot_embedding, "type_mapper", None)
        if mapper is not None:
            atomic_numbers = [int(z) for z in mapper.index_to_Z.cpu().tolist()]
        return atomic_numbers

    def resolve_rescale_context(atomic_numbers: List[int]) -> tuple[float, float, torch.Tensor]:
        scale, shift = 1.0, 0.0
        atomic_energies = torch.zeros(len(atomic_numbers))
        for module in output_modules:
            if not isinstance(module, GlobalRescaleShift):
                continue
            scale = pick_scalar(module.scale_by)
            shift = pick_scalar(module.shift_by)
            atomic_energies = module.atomic_energies
            if atomic_energies is None:
                break
            if not torch.is_tensor(atomic_energies):
                atomic_energies = torch.as_tensor(atomic_energies)
            z_idx = torch.as_tensor(atomic_numbers, dtype=torch.long)
            atomic_energies = torch.index_select(atomic_energies, -1, z_idx)
            break
        return scale, shift, atomic_energies

    def resolve_source_energy_readout(readout_model):
        if readout_model is None or not hasattr(readout_model, "readouts"):
            raise TypeError("Curator MACE readout is not exportable to official MACE.")
        if getattr(readout_model, "separate_heads", False):
            hidden_readouts = list(readout_model.readouts_by_head[properties.energy])
            final_readout = readout_model.final_readouts.get(properties.energy)
            head_key = properties.energy
        else:
            hidden_readouts = list(readout_model.readouts[:-1])
            final_readout = readout_model.readouts[-1]
            head_key = properties.energy
        if len(hidden_readouts) != len(repr_model.interactions) - 1:
            raise ValueError("Curator MACE energy readout depth does not match the interaction stack.")
        if final_readout is None:
            raise ValueError(
                f"Curator MACE readout head '{head_key}' does not expose a final scalar branch for export."
            )
        return hidden_readouts, final_readout

    def build_fresh_official_model():
        atomic_numbers = resolve_atomic_numbers()
        rep_kwargs = {
            "r_max": float(rep_config["cutoff"]),
            "num_bessel": int(rep_config["num_basis"]),
            "num_polynomial_cutoff": int(rep_config["power"]),
            "max_ell": repr_model.lmax,
            "num_interactions": len(repr_model.interactions),
            "num_elements": repr_model.embeddings.onehot_embedding.num_elements,
            "hidden_irreps": rep_config["hidden_irreps"],
            "MLP_irreps": rep_config["MLP_irreps"],
            "avg_num_neighbors": float(rep_config["avg_num_neighbors"]),
            "atomic_numbers": atomic_numbers,
            "correlation": rep_config["correlation"],
            "gate": torch.nn.functional.silu,
            "heads": ["head_0"],
            "radial_MLP": rep_config.get("radial_MLP"),
        }
        rep_kwargs["distance_transform"] = {
            "none": "None",
            "agnesi": "Agnesi",
            "soft": "Soft",
        }.get(str(rep_config.get("distance_transform", "none")).lower(), "None")
        rep_kwargs["atomic_inter_scale"], rep_kwargs["atomic_inter_shift"], rep_kwargs["atomic_energies"] = (
            resolve_rescale_context(atomic_numbers)
        )
        rep_kwargs["interaction_cls_first"] = getattr(
            mace_blocks,
            rep_config["interaction_cls_first"].__name__,
            mace_blocks.RealAgnosticInteractionBlock,
        )
        rep_kwargs["interaction_cls"] = getattr(
            mace_blocks,
            rep_config["interaction_cls"].__name__,
            rep_kwargs["interaction_cls_first"],
        )
        return mace_models.ScaleShiftMACE(**rep_kwargs)

    def load_core_embeddings(mace_model):
        for target_module, source_state in (
            (mace_model.radial_embedding.bessel_fn, repr_model.embeddings.radial_basis.basis.state_dict()),
            (mace_model.radial_embedding.cutoff_fn, repr_model.embeddings.radial_basis.cutoff_fn.state_dict()),
            (mace_model.node_embedding.linear, repr_model.embeddings.chemical_embedding.linear.state_dict()),
        ):
            target_module.load_state_dict(source_state, strict=False)

    def load_readout(idx: int, mace_model, hidden_readouts, final_readout):
        if idx < len(mace_model.readouts) - 1:
            src_readout = hidden_readouts[idx]
            if hasattr(mace_model.readouts[idx], "linear") and hasattr(src_readout, "linear"):
                target = mace_model.readouts[idx].linear
                state_dict = src_readout.linear.state_dict()
                label = f"official readout {idx}.linear"
            else:
                target = mace_model.readouts[idx]
                state_dict = src_readout.state_dict()
                label = f"official readout {idx}"
        elif idx < len(mace_model.readouts):
            target = mace_model.readouts[idx]
            state_dict = {
                key.replace("0.linear.", "linear_1.").replace("1.linear.", "linear_2."): value
                for key, value in final_readout.state_dict().items()
            }
            label = "official final readout"
        else:
            return
        _load_external_mapping(target, state_dict, label=label)

    readout_model = unwrap_single_domain(getattr(repr_model, "readout", None), label="readout")
    output_modules = [unwrap_single_domain(module, label="output module") for module in output_modules]
    rep_config = repr_model.export_init_kwargs()
    source_hidden_readouts, source_final_readout = resolve_source_energy_readout(readout_model)

    template_bytes = getattr(repr_model, "_official_mace_template_bytes", None)
    if template_bytes is not None:
        mace_model = torch.load(io.BytesIO(template_bytes), map_location="cpu", weights_only=False)
    else:
        mace_model = build_fresh_official_model()

    load_core_embeddings(mace_model)
    for idx in range(len(repr_model.interactions)):
        interaction_state = dict(repr_model.interactions[idx].state_dict())
        interaction_avg_num_neighbors = interaction_state.pop("avg_num_neighbors", None)
        _load_external_mapping(
            mace_model.interactions[idx],
            interaction_state,
            label=f"official interaction {idx}",
        )
        if interaction_avg_num_neighbors is not None and hasattr(mace_model.interactions[idx], "avg_num_neighbors"):
            avg_value = interaction_avg_num_neighbors
            if torch.is_tensor(avg_value):
                avg_value = float(avg_value.detach().cpu().reshape(-1)[0].item())
            mace_model.interactions[idx].avg_num_neighbors = avg_value
        _load_external_mapping(
            mace_model.products[idx],
            repr_model.products[idx].state_dict(),
            label=f"official product {idx}",
        )
        load_readout(idx, mace_model, source_hidden_readouts, source_final_readout)
    return mace_model


def _load_official_mace_as_curator_impl(
    model_ref: Union[str, Path],
    *,
    head: Optional[Union[str, int]] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    mace_model = _load_official_mace_model(model_ref, device=device)
    curator_model = create_model_from_mace(mace_model, head=head)
    if device is not None:
        curator_model.to(torch.device(device))
    return curator_model


def _load_official_mace_model(
    model_ref: Union[str, Path],
    *,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_serialization.add_safe_globals([slice])
    try:
        from mace.modules.models import ScaleShiftMACE

        torch_serialization.add_safe_globals([ScaleShiftMACE])
    except Exception:
        pass

    obj = torch.load(Path(model_ref), map_location=device, weights_only=False)
    if isinstance(obj, torch.nn.Module):
        mace_model = obj
    elif isinstance(obj, dict) and obj.get("model") is not None:
        mace_model = obj["model"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        raise TypeError(
            "MACE checkpoint does not include an instantiated model; please provide a TorchScript or full model checkpoint."
        )
    else:
        raise TypeError(f"Unsupported MACE checkpoint format at {model_ref}")
    return mace_model


def _load_official_nequip_as_curator_impl(
    model_ref: Union[str, Path],
    *,
    device: Optional[torch.device] = None,
    compile_mode: str = "eager",
) -> torch.nn.Module:
    if device is None:
        device = torch.device("cpu")
    official_model = _load_official_nequip_saved_model(model_ref, compile_mode=compile_mode)
    curator_model = create_model_from_nequip(official_model)
    curator_model.to(torch.device(device))
    return curator_model


def _load_official_nequip_saved_model(
    model_ref: Union[str, Path],
    *,
    compile_mode: str = "eager",
):
    torch_serialization.add_safe_globals([slice])
    try:
        load_utils = importlib.import_module("nequip.model.saved_models.load_utils")
    except ModuleNotFoundError:
        local_src = Path.home() / "local" / "src" / "nequip"
        if not local_src.exists():
            raise
        sys.path.insert(0, str(local_src))
        load_utils = importlib.import_module("nequip.model.saved_models.load_utils")
    return load_utils.load_saved_model(str(model_ref), compile_mode=compile_mode)


def _unwrap_official_nequip_model(nequip_model):
    graph_model = nequip_model
    func = getattr(getattr(graph_model, "model", None), "func", None) or getattr(graph_model, "func", None)
    if func is None:
        raise TypeError(
            "Unsupported NequIP model object. Expected a GraphModel loaded from a saved model/package."
        )
    return graph_model, func


def _create_model_from_nequip_impl(nequip_model) -> torch.nn.Module:
    from e3nn import o3

    def require_graph_model_and_func(model):
        graph_model = model
        func = getattr(getattr(graph_model, "model", None), "func", None) or getattr(graph_model, "func", None)
        if func is None:
            raise TypeError(
                "Unsupported NequIP model object. Expected a GraphModel loaded from a saved model/package."
            )
        return graph_model, func

    def collect_conv_layers(func) -> List[Any]:
        layers: List[Any] = []
        for name, module in func._modules.items():
            match = re.fullmatch(r"layer(\d+)_convnet", name)
            if match is not None:
                layers.append((int(match.group(1)), module))
        ordered = [module for _, module in sorted(layers, key=lambda item: item[0])]
        if not ordered:
            raise ValueError("Could not find NequIP convnet layers in the official model.")
        return ordered

    def infer_conv_spec(conv_layers):
        lmax = max(conv_layers[0].feature_irreps_hidden.ls)
        hidden_irreps = o3.Irreps(conv_layers[0].feature_irreps_hidden)
        parity = any(
            len({ir.p for _, ir in hidden_irreps if ir.l == l}) > 1
            for l in sorted(set(hidden_irreps.ls))
        )
        num_features = [max(int(mul) for mul, ir in hidden_irreps if ir.l == l) for l in range(lmax + 1)]
        radial_depths = []
        radial_widths = []
        avg_num_neighbors = None
        for layer in conv_layers:
            dims = list(layer.conv.edge_mlp.dims)
            radial_depths.append(max(0, len(dims) - 2))
            radial_widths.append(int(dims[1]) if len(dims) > 2 else 0)
            if avg_num_neighbors is None:
                norm_const = None
                avg_num_neighbors_norm = getattr(layer.conv, "avg_num_neighbors_norm", None)
                if avg_num_neighbors_norm is not None:
                    norm_const = getattr(avg_num_neighbors_norm, "norm_const", None)
                if norm_const is not None and norm_const.numel() == 1:
                    avg_num_neighbors = float(norm_const.reshape(-1)[0].item() ** -2)
                else:
                    scatter_norm_factor = getattr(layer.conv, "scatter_norm_factor", None)
                    if scatter_norm_factor is not None:
                        avg_num_neighbors = float(float(scatter_norm_factor) ** -2)
        if len(set(radial_depths)) != 1 or len(set(radial_widths)) != 1:
            raise ValueError(
                "Curator NequIP converter currently supports official NequIP packages with uniform "
                "radial MLP depth/width across layers."
            )
        return {
            "lmax": lmax,
            "parity": parity,
            "num_features": num_features,
            "radial_depth": radial_depths[0],
            "radial_width": radial_widths[0],
            "avg_num_neighbors": avg_num_neighbors,
        }

    def infer_readout_spec(readout_module):
        dims = list(readout_module.mlp_module.dims)
        nonlinearity_by_class = {
            "SiLU": "silu",
            "Mish": "mish",
            "GELU": "gelu",
            "Tanh": "tanh",
            "ShiftedSoftplus": "ssp",
        }
        mlp = getattr(getattr(readout_module, "mlp_module", None), "mlp", None)
        nonlinearity = None
        if mlp is not None:
            for module in mlp:
                nonlinearity = nonlinearity_by_class.get(module.__class__.__name__)
                if nonlinearity is not None:
                    break
        return {
            "depth": max(0, len(dims) - 2),
            "width": int(dims[1]) if len(dims) > 2 else None,
            "nonlinearity": nonlinearity,
        }

    graph_model, func = require_graph_model_and_func(nequip_model)
    conv_layers = collect_conv_layers(func)
    conv_spec = infer_conv_spec(conv_layers)
    readout_spec = infer_readout_spec(func.per_atom_energy_readout)
    species = list(func.per_type_energy_scale_shift.type_names)

    def build_species_value_dict(values: Optional[torch.Tensor]) -> Optional[Dict[str, float]]:
        if values is None:
            return None
        flat = values.detach().cpu().reshape(-1)
        if flat.numel() != len(species):
            return None
        return {species[i]: float(flat[i].item()) for i in range(len(species))}

    def scalar_to_int(value: Any, default: int) -> int:
        if value is None:
            return int(default)
        if torch.is_tensor(value):
            return int(value.item())
        return int(value)

    readout_head = HeadConfig(
        key=properties.atomic_energy,
        is_atomwise=True,
        reduction="none",
        dim=1,
        atomwise_shift=True,
        per_species_scale=build_species_value_dict(getattr(func.per_type_energy_scale_shift, "scales", None)),
        per_species_shift=build_species_value_dict(getattr(func.per_type_energy_scale_shift, "shifts", None)),
    )
    power = scalar_to_int(getattr(func.bessel_encode.cutoff, "p", None), 6)
    representation = Nequip(
        cutoff=float(func.edge_norm.r_max),
        num_interactions=len(conv_layers),
        species=species,
        lmax=conv_spec["lmax"],
        parity=conv_spec["parity"],
        num_features=conv_spec["num_features"],
        type_embed_num_features=int(func.type_embed.embed_module.embedding_dim),
        num_basis=int(func.bessel_encode.bessel_weights.shape[-1]),
        power=power,
        radial_mlp_depth=conv_spec["radial_depth"],
        radial_mlp_width=conv_spec["radial_width"],
        readout_mlp_hidden_layers_depth=readout_spec["depth"],
        readout_mlp_hidden_layers_width=readout_spec["width"],
        readout_mlp_nonlinearity=readout_spec["nonlinearity"],
        convolution_kwargs={"avg_num_neighbors": conv_spec["avg_num_neighbors"]} if conv_spec["avg_num_neighbors"] is not None else None,
        readout=partial(AtomwiseNN, heads=[readout_head]),
        heads=[readout_head],
    )

    output_modules: List[torch.nn.Module] = [
        GlobalRescaleShift(heads=[readout_head]),
        AtomwiseReduce(output_key=properties.energy, aggregation_mode="sum"),
    ]

    pair_potential = getattr(func, "pair_potential", None)
    if pair_potential is not None:
        output_modules.append(
            PairRepulsionEnergy(
                pair_fn=ZBLBasis(
                    p=scalar_to_int(getattr(getattr(pair_potential, "cutoff", None), "p", None), power),
                    screening_exponent=0.23,
                    screening_length=0.46850,
                    phi_coefficients=(0.18175, 0.50986, 0.28022, 0.02817),
                    phi_exponents=(3.19980, 0.94229, 0.40290, 0.20162),
                    energy_prefactor=float(pair_potential._qqr2exesquare.item()),
                    cutoff=float(func.edge_norm.r_max),
                    cutoff_by_species=False,
                    scatter_to="center",
                ),
                atomic_numbers=pair_potential.atomic_numbers.detach().cpu(),
            )
        )

    output_modules.append(GradientOutput(model_outputs=[properties.forces]))

    curator_model = NeuralNetworkPotential(
        representation=representation,
        input_modules=[PairwiseDistance()],
        output_modules=output_modules,
        model_outputs=[properties.energy, properties.forces],
    )
    readout_model = getattr(getattr(curator_model, "representation", None), "readout", None)
    official_sd = {
        key.replace("model.func.", "", 1).replace("func.", "", 1): value.detach().cpu()
        for key, value in graph_model.state_dict().items()
    }
    def dense_idx_or_none(mlp_idx: int) -> Optional[int]:
        return None if mlp_idx % 2 == 1 else mlp_idx // 2

    def readout_key_for_dense(dense_idx: int, attr: str) -> str:
        if readout_model is None:
            raise TypeError("Curator NequIP model does not expose a readout module.")
        if getattr(readout_model, "separate_heads", False):
            shared_mlp = getattr(readout_model, "shared_mlp", None)
            shared_depth = len(shared_mlp) if isinstance(shared_mlp, torch.nn.Sequential) else 0
            if dense_idx < shared_depth:
                return f"representation.readout.shared_mlp.{dense_idx}.linear.{attr}"
            return f"representation.readout.head_modules.{properties.energy}.linear.{attr}"
        return f"representation.readout.readout_mlp.{dense_idx}.linear.{attr}"

    pair_module_index = next(
        (
            idx
            for idx, module in enumerate(getattr(curator_model, "output_modules", []))
            if isinstance(module, PairRepulsionEnergy)
            and isinstance(getattr(module, "pair_fn", None), ZBLBasis)
            and not bool(getattr(module.pair_fn, "cutoff_by_species", True))
            and getattr(module.pair_fn, "scatter_to", "receiver") == "center"
        ),
        None,
    )
    mapped_state: Dict[str, torch.Tensor] = {}
    for key, value in official_sd.items():
        if key == "_empty":
            continue
        if key == "bessel_encode.bessel_weights":
            mapped_state["representation.embeddings.radial_basis.basis.bessel_weights"] = (
                value.reshape(-1) * (torch.pi / float(func.edge_norm.r_max))
            )
            continue
        if key == "type_embed.embed_module.weight":
            mapped_state["representation.embeddings.chemical_embedding.linear.weight"] = (
                value.reshape(-1) * (float(value.shape[0]) ** 0.5)
            )
            continue
        if key.startswith("layer") and "_convnet." in key:
            match = re.match(r"layer(\d+)_convnet\.(.+)", key)
            if match is None:
                continue
            layer_idx = int(match.group(1))
            rest = match.group(2)
            if rest.startswith("conv.edge_mlp.mlp."):
                parts = rest.split(".")
                dense_idx = dense_idx_or_none(int(parts[3]))
                if dense_idx is None:
                    continue
                rest = f"conv.fc.layer{dense_idx}.{parts[4]}"
            else:
                rest = rest.replace("conv.tp_scatter.", "conv.")
            mapped_state[f"representation.interactions.{layer_idx}.{rest}"] = value
            continue
        if key.startswith("per_atom_energy_readout.mlp_module.mlp."):
            parts = key.split(".")
            dense_idx = dense_idx_or_none(int(parts[3]))
            if dense_idx is None:
                continue
            attr = parts[4]
            mapped_state[readout_key_for_dense(dense_idx, attr)] = value.reshape(-1) if attr == "weight" else value
            continue
        if pair_module_index is not None and key.startswith("pair_potential."):
            suffix = key.split(".", 1)[1]
            if suffix == "atomic_numbers":
                mapped_state[f"output_modules.{pair_module_index}.atomic_numbers"] = value
            elif suffix == "_qqr2exesquare":
                mapped_state[f"output_modules.{pair_module_index}.pair_fn.energy_prefactor"] = value
    required_prefixes = [
        "representation.embeddings.radial_basis.basis.bessel_weights",
        "representation.embeddings.chemical_embedding.linear.weight",
        "representation.readout.",
    ]
    required_prefixes.extend(
        f"representation.interactions.{idx}."
        for idx in range(len(conv_layers))
    )
    _load_external_mapping(
        curator_model,
        mapped_state,
        label="official NequIP",
        required_prefixes=required_prefixes,
    )
    curator_model.model_outputs = [properties.energy, properties.forces]
    curator_model.eval()
    return curator_model


def _convert_model_wrapper_impl(
    model: torch.nn.Module,
    wrapper_config,
    *,
    target_dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    from curator.layer.wrappers import WrapperConfig, apply_wrappers, get_model_wrapper_config, resolve_wrapper_config
    from curator.utils import load_cueq_weights, load_e3nn_weights

    if not isinstance(wrapper_config, WrapperConfig):
        raise TypeError(
            f"convert_model_wrapper expects WrapperConfig, got {type(wrapper_config)!r}."
        )

    source_cfg = get_model_wrapper_config(model)
    source_model = model
    if source_cfg.adapter == "lora" and source_cfg.backend != wrapper_config.backend:
        from curator.layer.wrappers import merge_model_wrappers

        source_model = copy.deepcopy(model)
        merge_model_wrappers(source_model)
        source_model.train(model.training)
        source_model._initialized = getattr(model, "_initialized", False)
        source_model._wrapper_config = resolve_wrapper_config(backend=source_cfg.backend, adapter="none")
        wrapper_config = resolve_wrapper_config(backend=wrapper_config.backend, adapter="none")
        source_cfg = get_model_wrapper_config(source_model)
    if source_cfg.adapter == "lora" and wrapper_config.adapter != "lora":
        from curator.layer.wrappers import merge_model_wrappers

        source_model = copy.deepcopy(model)
        merge_model_wrappers(source_model)
        source_model.train(model.training)
        source_model._initialized = getattr(model, "_initialized", False)
    patched_model = apply_wrappers(source_model, wrapper_config, target_dtype=target_dtype)
    if patched_model is source_model:
        return model if source_model is not model else source_model

    if source_cfg.backend != wrapper_config.backend:
        if wrapper_config.backend == "cueq":
            load_e3nn_weights(source_model, patched_model)
        elif source_cfg.backend == "cueq":
            load_cueq_weights(source_model, patched_model)

    target_state = patched_model.state_dict()
    source_state = source_model.state_dict()
    source_aliases: Dict[str, torch.Tensor] = {}
    for source_key, source_tensor in source_state.items():
        source_aliases.setdefault(source_key, source_tensor)
        source_aliases.setdefault(source_key.replace(".base.", "."), source_tensor)
    matched: Dict[str, torch.Tensor] = {}
    for key, target_tensor in target_state.items():
        canonical_key = key.replace(".base.", ".")
        for source_tensor in (
            source_state.get(key),
            source_state.get(canonical_key),
            source_aliases.get(canonical_key),
        ):
            if source_tensor is None:
                continue
            compatible = _coerce_state_tensor_shape(source_tensor, target_tensor.shape)
            if compatible is None:
                continue
            matched[key] = compatible
            break
    if matched:
        patched_model.load_state_dict(matched, strict=False)
    return patched_model


def _convert_e3nn_to_cueq_impl(model):
    from curator.layer.wrappers import get_model_wrapper_config, resolve_wrapper_config

    wrapper_cfg = get_model_wrapper_config(model).to_dict()
    wrapper_cfg["backend"] = "cueq"
    return convert_model_wrapper(model, resolve_wrapper_config(**wrapper_cfg))


def _convert_cueq_to_e3nn_impl(model):
    from curator.layer.wrappers import get_model_wrapper_config, resolve_wrapper_config

    wrapper_cfg = get_model_wrapper_config(model).to_dict()
    wrapper_cfg["backend"] = "e3nn"
    return convert_model_wrapper(model, resolve_wrapper_config(**wrapper_cfg))


def _transform_model_to_direct_force_impl(
    model: torch.nn.Module,
    *,
    heads: Optional[List[Union[str, HeadConfig, Dict[str, Any]]]] = None,
) -> NeuralNetworkPotential:
    if not isinstance(model, NeuralNetworkPotential):
        raise TypeError(f"Expected NeuralNetworkPotential, got {type(model)}")

    representation = getattr(model, "representation", None)
    if representation is None:
        raise TypeError("Expected model to expose a representation.")

    export_fn = getattr(representation, "export_init_kwargs", None)
    if not callable(export_fn):
        raise TypeError(
            f"{representation.__class__.__name__} does not implement export_init_kwargs()."
        )

    direct_heads = heads or [properties.energy, properties.forces]
    readout_cls = MACEAtomwiseNN if isinstance(getattr(representation, "readout", None), MACEAtomwiseNN) else AtomwiseNN

    rep_kwargs = dict(export_fn())
    rep_kwargs["heads"] = direct_heads
    rep_kwargs["readout"] = partial(
        readout_cls,
        heads=direct_heads,
        separate_heads=True,
    )
    direct_representation = representation.__class__(**rep_kwargs)

    input_modules: List[torch.nn.Module] = []
    for module in getattr(model, "input_modules", []):
        copied = copy.deepcopy(module)
        if isinstance(copied, PairwiseDistance):
            copied.compute_forces = False
        input_modules.append(copied)

    output_modules: List[torch.nn.Module] = []
    removed_output_modules: List[str] = []
    for module in getattr(model, "output_modules", []):
        if isinstance(module, GradientOutput):
            removed_output_modules.append(type(module).__name__)
            continue
        if isinstance(module, AtomwiseReduce):
            removed_output_modules.append(type(module).__name__)
            continue
        if isinstance(module, GlobalRescaleShift):
            copied = GlobalRescaleShift(heads=direct_heads)
            _load_state_dict_by_shape(copied, module.state_dict())
            output_modules.append(copied)
            continue
        output_modules.append(copy.deepcopy(module))

    direct_model = NeuralNetworkPotential(
        representation=direct_representation,
        input_modules=input_modules,
        output_modules=output_modules,
        model_outputs=[properties.energy, properties.forces],
        heads=direct_heads,
    )
    loaded_tensors = load_pretrained_weights_from_model(direct_model, model)
    direct_model._initialized = getattr(model, "_initialized", False)
    direct_model.train(model.training)
    logger.info(
        "Applied model transform 'direct_force': readout=%s heads=%s removed_outputs=%s "
        "pairwise_compute_forces=%s loaded_tensors=%d",
        type(direct_representation.readout).__name__,
        [head.key for head in direct_representation.readout.heads],
        removed_output_modules or ["none"],
        [
            module.compute_forces
            for module in input_modules
            if isinstance(module, PairwiseDistance)
        ],
        loaded_tensors,
    )
    return direct_model


# Public transform and conversion entry points

# Generic weight transplant
def load_pretrained_weights_from_model(
    target_model: torch.nn.Module,
    source_model: torch.nn.Module,
) -> int:
    return _load_pretrained_weights_from_model_impl(target_model, source_model)


# Domain transforms
def convert_single_to_multi_domain(
    model: torch.nn.Module,
    *,
    template_domain: str = "0",
    domains: Optional[List[Union[str, int]]] = None,
    heads_by_domain: Optional[Dict[str, List[Any]]] = None,
) -> MultiDomainPotential:
    return _convert_single_to_multi_domain_impl(
        model,
        template_domain=template_domain,
        domains=domains,
        heads_by_domain=heads_by_domain,
    )


def convert_multi_to_single_domain(
    model: torch.nn.Module,
    *,
    domain: Optional[Union[str, int]] = "last",
) -> NeuralNetworkPotential:
    return _convert_multi_to_single_domain_impl(model, domain=domain)


def convert_multi_to_selected_domains(
    model: torch.nn.Module,
    *,
    domains: Optional[Union[str, int, List[Union[str, int]]]] = None,
) -> Union[NeuralNetworkPotential, MultiDomainPotential]:
    return _convert_multi_to_selected_domains_impl(model, domains=domains)


# MACE import/export transforms
def create_model_from_mace(
    mace_model,
    head: Optional[Union[str, int]] = None,
    *,
    mode: Literal["auto", "single", "multi"] = "auto",
):
    return _create_model_from_mace_impl(mace_model, head=head, mode=mode)


def build_mace_from_curator(curator_model):
    return _build_mace_from_curator_impl(curator_model)


def load_official_mace_as_curator(
    model_ref: Union[str, Path],
    *,
    head: Optional[Union[str, int]] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    return _load_official_mace_as_curator_impl(model_ref, head=head, device=device)


def convert_mace_to_curator(
    mace_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    device: Optional[torch.device] = None,
    head: Optional[Union[str, int]] = None,
) -> Union[str, Path]:
    model = load_official_mace_as_curator(mace_path, head=head, device=device)
    torch.save(model, output_path)
    return output_path


def load_official_nequip_as_curator(
    model_ref: Union[str, Path],
    *,
    device: Optional[torch.device] = None,
    compile_mode: str = "eager",
) -> torch.nn.Module:
    return _load_official_nequip_as_curator_impl(
        model_ref,
        device=device,
        compile_mode=compile_mode,
    )


# NequIP import transform
def create_model_from_nequip(nequip_model) -> torch.nn.Module:
    return _create_model_from_nequip_impl(nequip_model)


# Direct-force transform
def transform_model_to_direct_force(
    model: torch.nn.Module,
    *,
    heads: Optional[List[Union[str, HeadConfig, Dict[str, Any]]]] = None,
) -> NeuralNetworkPotential:
    return _transform_model_to_direct_force_impl(model, heads=heads)


# Wrapper/backend transforms
def convert_model_wrapper(
    model: torch.nn.Module,
    wrapper_config,
    *,
    target_dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    return _convert_model_wrapper_impl(
        model,
        wrapper_config,
        target_dtype=target_dtype,
    )


def convert_e3nn_to_cueq(model):
    return _convert_e3nn_to_cueq_impl(model)


def convert_cueq_to_e3nn(model):
    return _convert_cueq_to_e3nn_impl(model)


MODEL_TRANSFORM_REGISTRY = {
    "single_to_multi_domain": convert_single_to_multi_domain,
    "multi_to_single_domain": convert_multi_to_single_domain,
    "multi_to_selected_domains": convert_multi_to_selected_domains,
    "direct_force": transform_model_to_direct_force,
    "model_wrapper": convert_model_wrapper,
    "wrapper": convert_model_wrapper,
    "e3nn_to_cueq": convert_e3nn_to_cueq,
    "cueq_to_e3nn": convert_cueq_to_e3nn,
}


def get_model_transform_registry():
    return dict(MODEL_TRANSFORM_REGISTRY)


def apply_model_transform(
    model: torch.nn.Module,
    transform: Union[str, Dict[str, Any]],
    *,
    target_dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    if isinstance(transform, str):
        transform_type = transform
        transform_kwargs: Dict[str, Any] = {}
    elif isinstance(transform, dict):
        transform_type = str(transform.get("type", "")).strip()
        if not transform_type:
            raise ValueError("Transform dict must contain a non-empty 'type' field.")
        transform_kwargs = {k: v for k, v in transform.items() if k != "type"}
    else:
        raise TypeError(f"Unsupported transform spec type: {type(transform)!r}")

    if transform_type not in MODEL_TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown model transform {transform_type!r}. "
            f"Available transforms: {sorted(MODEL_TRANSFORM_REGISTRY.keys())}."
        )

    transform_fn = MODEL_TRANSFORM_REGISTRY[transform_type]
    logger.debug(
        "Applying model transform %r with kwargs=%s",
        transform_type,
        transform_kwargs,
    )
    if transform_fn is convert_model_wrapper:
        from curator.layer.wrappers import WrapperConfig, resolve_wrapper_config

        wrapper_config = transform_kwargs.pop("wrapper_config", None)
        if isinstance(wrapper_config, WrapperConfig):
            resolved_wrapper = wrapper_config
        elif isinstance(wrapper_config, dict):
            resolved_wrapper = resolve_wrapper_config(**wrapper_config)
        else:
            resolved_wrapper = resolve_wrapper_config(**transform_kwargs)
            transform_kwargs = {}
        logger.info(
            "Applying model transform 'wrapper': backend=%s adapter=%s lora_target_groups=%s",
            resolved_wrapper.backend,
            resolved_wrapper.adapter,
            resolved_wrapper.lora_target_groups,
        )
        return transform_fn(model, resolved_wrapper, target_dtype=target_dtype, **transform_kwargs)

    if target_dtype is not None and "target_dtype" in transform_fn.__code__.co_varnames:
        transform_kwargs.setdefault("target_dtype", target_dtype)
    return transform_fn(model, **transform_kwargs)


def apply_model_transforms(
    model: torch.nn.Module,
    transforms: Optional[List[Union[str, Dict[str, Any]]]],
    *,
    target_dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    if transforms:
        transform_names = [
            transform
            if isinstance(transform, str)
            else str(transform.get("type", "")).strip()
            for transform in transforms
        ]
        logger.info("Applying model transforms in order: %s", " -> ".join(transform_names))
    transformed = model
    for transform in transforms or []:
        transformed = apply_model_transform(
            transformed,
            transform,
            target_dtype=target_dtype,
        )
    return transformed

__all__ = [
    'MODEL_TRANSFORM_REGISTRY',
    'apply_model_transform',
    'apply_model_transforms',
    'build_mace_from_curator',
    'convert_cueq_to_e3nn',
    'convert_e3nn_to_cueq',
    'convert_mace_to_curator',
    'convert_model_wrapper',
    'convert_multi_to_selected_domains',
    'convert_multi_to_single_domain',
    'convert_single_to_multi_domain',
    'create_model_from_mace',
    'create_model_from_nequip',
    'get_model_transform_registry',
    '_load_official_mace_model',
    '_load_official_nequip_saved_model',
    '_unwrap_official_nequip_model',
    'load_official_mace_as_curator',
    'load_official_nequip_as_curator',
    'load_pretrained_weights_from_model',
    'transform_model_to_direct_force',
]
