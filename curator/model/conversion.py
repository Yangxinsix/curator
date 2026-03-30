from __future__ import annotations

import copy
from functools import partial
import importlib
import logging
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
    NequIPZBLPairEnergy,
    PairRepulsionEnergy,
    PairwiseDistance,
    PerSpeciesRescaleShift,
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


def _load_state_dict_by_shape(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    current = module.state_dict()
    filtered = {}
    for name, param in state_dict.items():
        if name in current and current[name].shape == param.shape:
            filtered[name] = param
    module.load_state_dict(filtered, strict=False)


def _match_tensor_shape_for_load(
    src: torch.Tensor,
    target_shape: torch.Size,
    *,
    expand: bool,
) -> torch.Tensor:
    if expand and src.shape != target_shape and src.dim() + 1 == len(target_shape) and target_shape[0] == 1:
        return src.unsqueeze(0)
    if (not expand) and src.shape != target_shape and src.dim() == len(target_shape) + 1 and src.shape[0] == 1:
        return src.squeeze(0)
    return src


def _get_domain_modules(module) -> Optional[torch.nn.ModuleDict]:
    domain_modules = getattr(module, "domain_modules", None)
    if isinstance(domain_modules, torch.nn.ModuleDict) and len(domain_modules) > 0:
        return domain_modules
    return None


def _available_model_domains(model: torch.nn.Module) -> List[str]:
    readout = getattr(getattr(model, "representation", None), "readout", None)
    domain_modules = _get_domain_modules(readout)
    if domain_modules is not None:
        return [str(domain) for domain in domain_modules.keys()]

    for module in getattr(model, "output_modules", []):
        domain_modules = _get_domain_modules(module)
        if domain_modules is not None:
            return [str(domain) for domain in domain_modules.keys()]
    return []


def _resolve_model_domain(model: torch.nn.Module, domain: Optional[Union[str, int]] = "last") -> str:
    domains = _available_model_domains(model)
    if not domains:
        raise TypeError(f"Expected a multi-domain model, got {type(model)}.")
    if domain is None or str(domain) == "last":
        return domains[-1]

    selected = str(domain)
    if selected not in domains:
        raise ValueError(f"Domain {selected!r} not found in available domains {domains}.")
    return selected


def _wrap_multi_domain_model(model: torch.nn.Module) -> MultiDomainPotential:
    representation = getattr(model, "representation", None)
    if representation is None:
        raise TypeError("Expected a model with a `representation` attribute.")

    readout = getattr(representation, "readout", None)
    domain_modules = _get_domain_modules(readout)
    if domain_modules is None:
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
        heads = list(getattr(readout, "heads", None) or [])
        readout_kwargs: Dict[str, Any] = {"domains": ["0"]}
        if heads:
            readout_kwargs["heads_by_domain"] = {"0": heads}
        rep_kwargs["readout"] = partial(readout_cls, **readout_kwargs)

        new_representation = representation.__class__(**rep_kwargs)
        rep_state_dict = {}
        for key, value in representation.state_dict().items():
            if key.startswith("readout.domain_modules."):
                rep_state_dict[key] = value
            elif key.startswith("readout."):
                rep_state_dict[f"readout.domain_modules.0.{key[len('readout.') :]}"] = value
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
            multi_domain_module.domain_modules["0"] = module
            multi_domain_module.train(module.training)
            output_modules.append(multi_domain_module)
        else:
            output_modules.append(module)

    wrapped = MultiDomainPotential(
        representation=base_model.representation,
        input_modules=list(base_model.input_modules),
        output_modules=output_modules,
        model_outputs=list(base_model.model_outputs),
        heads=getattr(base_model, "heads", None),
    )
    wrapped._initialized = getattr(model, "_initialized", False)
    wrapped.train(model.training)
    return wrapped


def convert_single_to_multi_domain(
    model: torch.nn.Module,
    *,
    template_domain: str = "0",
) -> MultiDomainPotential:
    if isinstance(model, MultiDomainPotential):
        return model
    converted = _wrap_multi_domain_model(model)
    if not isinstance(converted, MultiDomainPotential):
        raise TypeError(f"Expected MultiDomainPotential after promotion, got {type(converted)}")
    if str(template_domain) != "0":
        apply_domain_set(
            converted,
            [str(template_domain)],
            mode="replace",
            template_domain="0",
            init_strategy="copy",
        )
    return converted


def convert_multi_to_single_domain(
    model: torch.nn.Module,
    *,
    domain: Optional[Union[str, int]] = "last",
) -> NeuralNetworkPotential:
    selected_domain = _resolve_model_domain(model, domain=domain)

    representation = copy.deepcopy(model.representation)
    readout_domains = _get_domain_modules(getattr(representation, "readout", None))
    if readout_domains is None or selected_domain not in readout_domains:
        raise TypeError("Expected multi-domain representation readout for single-domain conversion.")
    representation.readout = copy.deepcopy(readout_domains[selected_domain])

    output_modules: List[torch.nn.Module] = []
    for module in getattr(model, "output_modules", []):
        domain_modules = _get_domain_modules(module)
        if domain_modules is None:
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


def _resolve_mace_head_mapping(
    mace_model,
    head: Optional[Union[str, int]] = None,
) -> Dict[str, int]:
    heads = list(getattr(mace_model, "heads", []) or [])
    if head is None:
        return {str(idx): idx for idx in range(len(heads))} if len(heads) > 1 else {"0": 0}
    if len(heads) <= 1:
        return {"0": 0}
    if isinstance(head, int):
        if head < 0 or head >= len(heads):
            raise ValueError(f"Head index {head} out of range for heads={heads}")
        return {"0": int(head)}

    head_name = str(head)
    if head_name not in heads:
        raise ValueError(f"Head {head_name} not found in heads={heads}")
    return {"0": heads.index(head_name)}


def _create_curator_mace_representation(
    mace_model,
) -> MACE:
    interaction_map = {
        "RealAgnosticInteractionBlock": RealAgnosticInteractionBlock,
        "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
    }
    num_mace_heads = max(len(list(getattr(mace_model, "heads", []) or [])), 1)
    interaction_cls_first = interaction_map.get(
        mace_model.interactions[0].__class__.__name__,
        RealAgnosticInteractionBlock,
    )
    if len(mace_model.interactions) > 1:
        interaction_cls = interaction_map.get(
            mace_model.interactions[1].__class__.__name__,
            RealAgnosticResidualInteractionBlock,
        )
    else:
        interaction_cls = interaction_cls_first

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

    def load_head(target_readout, head_idx: int) -> None:
        for idx, src_readout in enumerate(mace_model.readouts):
            dst_readout = target_readout.readouts[idx]
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
                state_dict = {
                    key.replace("linear_1.", "0.linear.").replace("linear_2.", "1.linear."): value
                    for key, value in state_dict.items()
                }
            module_state = dst_readout.state_dict()
            _load_state_dict_by_shape(
                dst_readout,
                {
                    key: _match_tensor_shape_for_load(value, module_state[key].shape, expand=False)
                    if key in module_state
                    else value
                    for key, value in state_dict.items()
                },
            )

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
    load_head(curator_mace.readout, template_head_idx)

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
        load_head(readout_domains[str(domain)], head_idx)

    for module in model.output_modules:
        if isinstance(module, MultiDomainRescaleShift):
            for domain, head_idx in domain_to_head_idx.items():
                module.domain_modules[str(domain)] = GlobalRescaleShift(
                    heads=[build_energy_head(head_idx)]
                )
    return model

def create_model_from_mace(
    mace_model,
    head: Optional[Union[str, int]] = None,
):
    heads = list(getattr(mace_model, "heads", []) or [])
    if head is None and len(heads) > 1:
        return create_multi_domain_model_from_mace(mace_model)
    return _build_curator_mace_model(
        mace_model,
        _resolve_mace_head_mapping(mace_model, head=head),
        mode="single",
    )


def create_multi_domain_model_from_mace(
    mace_model,
    head: Optional[Union[str, int]] = None,
):
    return _build_curator_mace_model(
        mace_model,
        _resolve_mace_head_mapping(mace_model, head=head),
        mode="multi",
    )


def build_mace_from_curator(curator_model):
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

    rep_config = repr_model.export_init_kwargs()
    multi_domain_error = "curator_to_mace only supports single-domain Curator MACE models. Multi-domain export is not supported."
    readout_model = getattr(repr_model, "readout", None)
    readout_domains = getattr(readout_model, "domain_modules", None)
    if isinstance(readout_domains, torch.nn.ModuleDict) and len(readout_domains) == 1:
        readout_model = next(iter(readout_domains.values()))
    elif isinstance(readout_domains, torch.nn.ModuleDict):
        raise NotImplementedError(multi_domain_error)

    normalized_output_modules = []
    for module in output_modules:
        domain_modules = getattr(module, "domain_modules", None)
        if isinstance(domain_modules, torch.nn.ModuleDict) and len(domain_modules) == 1:
            module = next(iter(domain_modules.values()))
        elif isinstance(domain_modules, torch.nn.ModuleDict):
            raise NotImplementedError(multi_domain_error)
        normalized_output_modules.append(module)
    output_modules = normalized_output_modules

    if readout_model is None or not hasattr(readout_model, "readouts"):
        raise TypeError("Curator MACE readout is not exportable to official MACE.")

    atomic_numbers = list(range(repr_model.embeddings.onehot_embedding.num_elements))
    mapper = getattr(repr_model.embeddings.onehot_embedding, "type_mapper", None)
    if mapper is not None:
        atomic_numbers = [int(z) for z in mapper.index_to_Z.cpu().tolist()]

    correlation = [
        contraction.correlation
        for contraction in repr_model.products[0].symmetric_contractions.contractions
    ]
    avg_num_neighbors = float(rep_config["avg_num_neighbors"])
    num_basis = int(rep_config["num_basis"])
    num_polynomial_cutoff = int(rep_config["power"])
    distance_transform = {
        "none": "None",
        "agnesi": "Agnesi",
        "soft": "Soft",
    }.get(str(rep_config.get("distance_transform", "none")).lower(), "None")

    scale, shift = 1.0, 0.0
    atomic_energies = torch.zeros(len(atomic_numbers))

    def pick_scalar(value):
        if isinstance(value, list):
            return value[0] if value else 0.0
        if torch.is_tensor(value):
            flat = value.detach().cpu().reshape(-1)
            return flat[0].item() if flat.numel() > 0 else 0.0
        return value

    for module in output_modules:
        if isinstance(module, GlobalRescaleShift):
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

    interaction_cls_first = getattr(
        mace_blocks,
        rep_config["interaction_cls_first"].__name__,
        mace_blocks.RealAgnosticInteractionBlock,
    )
    interaction_cls = getattr(
        mace_blocks,
        rep_config["interaction_cls"].__name__,
        interaction_cls_first,
    )

    mace_model = mace_models.ScaleShiftMACE(
        atomic_inter_scale=scale,
        atomic_inter_shift=shift,
        r_max=float(rep_config["cutoff"]),
        num_bessel=num_basis,
        num_polynomial_cutoff=num_polynomial_cutoff,
        max_ell=repr_model.lmax,
        interaction_cls=interaction_cls,
        interaction_cls_first=interaction_cls_first,
        num_interactions=len(repr_model.interactions),
        num_elements=repr_model.embeddings.onehot_embedding.num_elements,
        hidden_irreps=rep_config["hidden_irreps"],
        MLP_irreps=rep_config["MLP_irreps"],
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=atomic_numbers,
        correlation=correlation,
        gate=torch.nn.functional.silu,
        heads=["head_0"],
        distance_transform=distance_transform,
    )

    mace_model.radial_embedding.bessel_fn.load_state_dict(
        repr_model.embeddings.radial_basis.basis.state_dict(),
        strict=False,
    )
    mace_model.radial_embedding.cutoff_fn.load_state_dict(
        repr_model.embeddings.radial_basis.cutoff_fn.state_dict(),
        strict=False,
    )
    mace_model.node_embedding.linear.load_state_dict(
        repr_model.embeddings.chemical_embedding.linear.state_dict(),
        strict=False,
    )
    for idx in range(len(repr_model.interactions)):
        _load_state_dict_by_shape(mace_model.interactions[idx], repr_model.interactions[idx].state_dict())
        _load_state_dict_by_shape(mace_model.products[idx], repr_model.products[idx].state_dict())
        if idx < len(mace_model.readouts) - 1 and hasattr(mace_model.readouts[idx], "linear"):
            _load_state_dict_by_shape(
                mace_model.readouts[idx].linear,
                readout_model.readouts[idx].state_dict(),
            )
        elif idx < len(mace_model.readouts):
            _load_state_dict_by_shape(
                mace_model.readouts[idx],
                readout_model.readouts[idx].state_dict(),
            )
    return mace_model


def convert_mace_to_curator(
    mace_path: Union[str, Path],
    output_path: Union[str, Path],
    head: Optional[Union[str, int]] = None,
    device: Optional[torch.device] = None,
) -> Path:
    torch_serialization.add_safe_globals([slice])
    try:
        from mace.modules.models import ScaleShiftMACE

        torch_serialization.add_safe_globals([ScaleShiftMACE])
    except Exception:
        pass

    mace_path = Path(mace_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(mace_path, map_location=device, weights_only=False)
    mace_model = None
    if isinstance(obj, torch.nn.Module):
        mace_model = obj
    elif isinstance(obj, dict):
        mace_model = obj.get("model")
        if mace_model is None and "state_dict" in obj:
            raise TypeError(
                "MACE checkpoint does not include an instantiated model; please provide a TorchScript or full model checkpoint."
            )
    if mace_model is None:
        raise TypeError(f"Unsupported MACE checkpoint format at {mace_path}")
    curator_model = create_model_from_mace(mace_model, head=head)
    output_path = Path(output_path)
    torch.save(curator_model, output_path)
    return output_path


def convert_curator_to_mace(
    curator_path: Union[str, Path],
    output_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Path:
    from curator.utils import load_model

    curator_path = Path(curator_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(curator_path, device=device, load_compiled=False, load_weights_only=False)
    mace_model = build_mace_from_curator(model)
    output_path = Path(output_path)
    torch.save(mace_model, output_path)
    return output_path


def _load_official_nequip_saved_model(
    model_ref: Union[str, Path],
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
    func = getattr(getattr(graph_model, "model", None), "func", None)
    if func is None:
        func = getattr(graph_model, "func", None)
    if func is None:
        raise TypeError(
            "Unsupported NequIP model object. Expected a GraphModel loaded from a saved model/package."
        )
    return graph_model, func


def _get_official_nequip_conv_layers(func) -> List[Any]:
    layers: List[Tuple[int, Any]] = []
    for name, module in func._modules.items():
        match = re.fullmatch(r"layer(\d+)_convnet", name)
        if match is not None:
            layers.append((int(match.group(1)), module))
    layers.sort(key=lambda item: item[0])
    return [module for _, module in layers]


def _infer_nequip_parity(hidden_irreps) -> bool:
    from e3nn import o3

    irreps = o3.Irreps(hidden_irreps)
    for l in sorted(set(irreps.ls)):
        parities = {ir.p for _, ir in irreps if ir.l == l}
        if len(parities) > 1:
            return True
    return False


def _infer_nequip_num_features(hidden_irreps, lmax: int) -> List[int]:
    from e3nn import o3

    irreps = o3.Irreps(hidden_irreps)
    return [max(int(mul) for mul, ir in irreps if ir.l == l) for l in range(lmax + 1)]


def _infer_nequip_readout_nonlinearity(readout_module) -> Optional[str]:
    mlp = getattr(getattr(readout_module, "mlp_module", None), "mlp", None)
    if mlp is None:
        return None
    for module in mlp:
        cls_name = module.__class__.__name__
        if cls_name == "SiLU":
            return "silu"
        if cls_name == "Mish":
            return "mish"
        if cls_name == "GELU":
            return "gelu"
        if cls_name == "Tanh":
            return "tanh"
        if cls_name in {"Identity", "ScalarLinearLayer"}:
            continue
        if cls_name == "ShiftedSoftplus":
            return "ssp"
    return None


def _build_species_value_dict(
    species: List[str],
    values: Optional[torch.Tensor],
) -> Optional[Dict[str, float]]:
    if values is None:
        return None
    flat = values.detach().cpu().reshape(-1)
    if flat.numel() != len(species):
        return None
    return {species[i]: float(flat[i].item()) for i in range(len(species))}


def _scalar_to_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _map_official_nequip_state_dict(
    nequip_model,
    curator_model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    graph_model, func = _unwrap_official_nequip_model(nequip_model)
    official_sd = {
        key.replace("model.func.", "", 1).replace("func.", "", 1): value.detach().cpu()
        for key, value in graph_model.state_dict().items()
    }

    mapped: Dict[str, torch.Tensor] = {}
    pair_module_index = next(
        (
            idx
            for idx, module in enumerate(getattr(curator_model, "output_modules", []))
            if isinstance(module, NequIPZBLPairEnergy)
        ),
        None,
    )

    species = list(func.per_type_energy_scale_shift.type_names)
    output_modules = list(getattr(curator_model, "output_modules", []))
    if output_modules and isinstance(output_modules[0], PerSpeciesRescaleShift):
        per_species_module = output_modules[0]
        scales = torch.ones_like(per_species_module.scales)
        shifts = torch.zeros_like(per_species_module.shifts)
        official_scales = getattr(func.per_type_energy_scale_shift, "scales", None)
        official_shifts = getattr(func.per_type_energy_scale_shift, "shifts", None)
        if official_scales is not None:
            official_scales = official_scales.detach().cpu().reshape(-1)
            for idx, symbol in enumerate(species):
                scales[chemical_symbols.index(symbol)] = official_scales[idx]
        if official_shifts is not None:
            official_shifts = official_shifts.detach().cpu().reshape(-1)
            for idx, symbol in enumerate(species):
                shifts[chemical_symbols.index(symbol)] = official_shifts[idx]
        mapped["output_modules.0.scales"] = scales
        mapped["output_modules.0.shifts"] = shifts

    for key, value in official_sd.items():
        if key == "_empty":
            continue
        if key == "bessel_encode.bessel_weights":
            mapped["representation.embeddings.radial_basis.basis.bessel_weights"] = (
                value.reshape(-1) * (torch.pi / float(func.edge_norm.r_max))
            )
            continue
        if key == "type_embed.embed_module.weight":
            mapped["representation.embeddings.chemical_embedding.linear.weight"] = (
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
                mlp_idx = int(parts[3])
                if mlp_idx % 2 == 1:
                    continue
                dense_idx = mlp_idx // 2
                rest = f"conv.fc.layer{dense_idx}.{parts[4]}"
            else:
                rest = rest.replace("conv.tp_scatter.", "conv.")
            mapped[f"representation.interactions.{layer_idx}.{rest}"] = value
            continue
        if key.startswith("per_atom_energy_readout.mlp_module.mlp."):
            parts = key.split(".")
            mlp_idx = int(parts[3])
            if mlp_idx % 2 == 1:
                continue
            dense_idx = mlp_idx // 2
            attr = parts[4]
            mapped_key = f"representation.readout.readout_mlp.{dense_idx}.linear.{attr}"
            mapped[mapped_key] = value.reshape(-1) if attr == "weight" else value
            continue
        if pair_module_index is not None and key.startswith("pair_potential."):
            suffix = key.split(".", 1)[1]
            if suffix in {"atomic_numbers", "_qqr2exesquare"}:
                mapped[f"output_modules.{pair_module_index}.{suffix.lstrip('_')}"] = value

    return mapped


def create_model_from_nequip(nequip_model) -> torch.nn.Module:
    graph_model, func = _unwrap_official_nequip_model(nequip_model)
    conv_layers = _get_official_nequip_conv_layers(func)
    if not conv_layers:
        raise ValueError("Could not find NequIP convnet layers in the official model.")

    lmax = max(conv_layers[0].feature_irreps_hidden.ls)
    parity = _infer_nequip_parity(conv_layers[0].feature_irreps_hidden)
    num_features = _infer_nequip_num_features(conv_layers[0].feature_irreps_hidden, lmax=lmax)

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

    readout_dims = list(func.per_atom_energy_readout.mlp_module.dims)
    readout_depth = max(0, len(readout_dims) - 2)
    readout_width = int(readout_dims[1]) if readout_depth > 0 else None
    readout_nonlinearity = _infer_nequip_readout_nonlinearity(func.per_atom_energy_readout)

    readout_head = HeadConfig(
        key=properties.atomic_energy,
        is_atomwise=True,
        reduction="none",
        dim=1,
    )
    power = _scalar_to_int(getattr(func.bessel_encode.cutoff, "p", None), 6)
    representation = Nequip(
        cutoff=float(func.edge_norm.r_max),
        num_interactions=len(conv_layers),
        species=list(func.per_type_energy_scale_shift.type_names),
        lmax=lmax,
        parity=parity,
        num_features=num_features,
        type_embed_num_features=int(func.type_embed.embed_module.embedding_dim),
        num_basis=int(func.bessel_encode.bessel_weights.shape[-1]),
        power=power,
        radial_mlp_depth=radial_depths[0],
        radial_mlp_width=radial_widths[0],
        readout_mlp_hidden_layers_depth=readout_depth,
        readout_mlp_hidden_layers_width=readout_width,
        readout_mlp_nonlinearity=readout_nonlinearity,
        convolution_kwargs={"avg_num_neighbors": avg_num_neighbors} if avg_num_neighbors is not None else None,
        readout=partial(AtomwiseNN, heads=[readout_head]),
        heads=[readout_head],
    )

    output_modules: List[torch.nn.Module] = [
        PerSpeciesRescaleShift(
            scales=_build_species_value_dict(
                list(func.per_type_energy_scale_shift.type_names),
                getattr(func.per_type_energy_scale_shift, "scales", None),
            ),
            shifts=_build_species_value_dict(
                list(func.per_type_energy_scale_shift.type_names),
                getattr(func.per_type_energy_scale_shift, "shifts", None),
            ),
            scales_keys=[properties.atomic_energy],
            shifts_keys=[properties.atomic_energy],
        ),
        AtomwiseReduce(output_key=properties.energy, aggregation_mode="sum"),
    ]

    pair_potential = getattr(func, "pair_potential", None)
    if pair_potential is not None:
        output_modules.append(
            NequIPZBLPairEnergy(
                atomic_numbers=pair_potential.atomic_numbers.detach().cpu(),
                cutoff=float(func.edge_norm.r_max),
                power=_scalar_to_int(getattr(getattr(pair_potential, "cutoff", None), "p", None), power),
                qqr2exesquare=float(pair_potential._qqr2exesquare.item()),
            )
        )

    output_modules.append(GradientOutput(model_outputs=[properties.forces]))

    curator_model = NeuralNetworkPotential(
        representation=representation,
        input_modules=[PairwiseDistance()],
        output_modules=output_modules,
        model_outputs=[properties.energy, properties.forces],
    )
    _load_state_dict_by_shape(
        curator_model,
        _map_official_nequip_state_dict(graph_model, curator_model),
    )
    curator_model.model_outputs = [properties.energy, properties.forces]
    curator_model.eval()
    return curator_model


def convert_nequip_to_curator(
    nequip_path: Union[str, Path],
    output_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Path:
    if device is None:
        device = torch.device("cpu")
    official_model = _load_official_nequip_saved_model(nequip_path, compile_mode="eager")
    curator_model = create_model_from_nequip(official_model)
    curator_model.to(torch.device(device))
    output_path = Path(output_path)
    torch.save(curator_model.cpu(), output_path)
    return output_path


def _convert_backend(model, *, use_cueq: bool):
    from curator.layer.wrappers import apply_wrappers, export_wrapper_config

    wrapper_cfg = export_wrapper_config(model)
    wrapper_cfg["use_cueq"] = bool(use_cueq)
    if use_cueq:
        wrapper_cfg["wrapper_stack"] = "cueq+elora" if wrapper_cfg.get("use_elora") else "cueq"
    else:
        wrapper_cfg["wrapper_stack"] = "elora" if wrapper_cfg.get("use_elora") else "e3nn"
    return apply_wrappers(model, wrapper_cfg)


def convert_e3nn_to_cueq(model):
    return _convert_backend(model, use_cueq=True)


def convert_cueq_to_e3nn(model):
    return _convert_backend(model, use_cueq=False)

__all__ = [
    "_load_state_dict_by_shape",
    "build_mace_from_curator",
    "convert_multi_to_single_domain",
    "convert_single_to_multi_domain",
    "convert_cueq_to_e3nn",
    "convert_curator_to_mace",
    "convert_e3nn_to_cueq",
    "convert_mace_to_curator",
    "convert_nequip_to_curator",
    "create_model_from_mace",
    "create_model_from_nequip",
    "create_multi_domain_model_from_mace",
]
