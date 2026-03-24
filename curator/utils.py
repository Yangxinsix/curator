import torch
import json
from e3nn.util.jit import script
from omegaconf import open_dict, OmegaConf, DictConfig, ListConfig
from hydra import compose, initialize, initialize_config_dir
import hydra
from hydra.utils import instantiate, get_class
import inspect
from collections import abc
import logging
from ase import units
from pathlib import Path, PosixPath
from typing import Any, List, Optional, Tuple, Union, Dict, Literal
import numpy as np
import torch.serialization as torch_serialization
import copy

from ase.data import chemical_symbols
import torch, re
from curator.data import properties


def write_json(path: Union[str, Path], payload: Any, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=indent)
    return path


def save_npz(path: Union[str, Path], **payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path

def register_resolvers():
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y, replace=True)
    OmegaConf.register_new_resolver("multiply_fs", lambda x: x * units.fs, replace=True)
    OmegaConf.register_new_resolver("divide_by_fs", lambda x: x / units.fs, replace=True)

def create_model_from_mace(
    mace_model,
    head: Optional[Union[str, int]] = None,
):
    from curator.layer import (
        GlobalRescaleShift,
        PairwiseDistance,
        GradientOutput,
        RealAgnosticInteractionBlock, 
        RealAgnosticResidualInteractionBlock, 
        AgnesiTransform, 
        SoftTransform, 
        PairRepulsionEnergy, 
        ZBLBasis, 
        MultiDomainMACEAtomwiseNN,
    )
    from curator.layer._rescale import MultiDomainRescaleShift
    from curator.model import NeuralNetworkPotential, MACE
    from curator.data.properties import HeadConfig
    from functools import partial

    heads = list(getattr(mace_model, "heads", []))
    # Curator semantics:
    # - domain == data origin
    # MACE semantics:
    # - head == data origin
    # Therefore: convert MACE heads into Curator domains.
    all_heads_to_domains = (head is None and len(heads) > 1)
    if all_heads_to_domains:
        domains = [str(i) for i in range(len(heads))]
        domain_to_head_idx = {str(i): int(i) for i in range(len(heads))}
    else:
        domains = ["0"]
        mace_head_idx: Optional[int] = None
        if len(heads) > 1:
            if isinstance(head, int):
                if head < 0 or head >= len(heads):
                    raise ValueError(f"Head index {head} out of range for heads={heads}")
                mace_head_idx = int(head)
            elif head is not None:
                head_name = str(head)
                if head_name not in heads:
                    raise ValueError(f"Head {head_name} not found in heads={heads}")
                mace_head_idx = heads.index(head_name)
            else:
                mace_head_idx = 0
        domain_to_head_idx = {"0": int(mace_head_idx or 0)}
    num_mace_heads = len(heads) if len(heads) > 0 else 1

    interaction_map = {
        'RealAgnosticInteractionBlock': RealAgnosticInteractionBlock,
        'RealAgnosticResidualInteractionBlock': RealAgnosticResidualInteractionBlock,
    }

    # input_modules = [Strain(), PairwiseDistance(compute_distance_from_R=True)]
    input_modules = [PairwiseDistance()]
    interaction_cls_first = interaction_map.get(
        mace_model.interactions[0].__class__.__name__, RealAgnosticInteractionBlock
    )
    if len(mace_model.interactions) > 1:
        interaction_cls = interaction_map.get(
            mace_model.interactions[1].__class__.__name__, RealAgnosticResidualInteractionBlock
        )
    else:
        interaction_cls = interaction_cls_first

    distance_transform = None
    if hasattr(mace_model, "radial_embedding") and hasattr(mace_model.radial_embedding, "distance_transform"):
        dt = mace_model.radial_embedding.distance_transform
        if dt is not None:
            if dt.__class__.__name__ == "AgnesiTransform":
                distance_transform = AgnesiTransform()
            elif dt.__class__.__name__ == "SoftTransform":
                distance_transform = SoftTransform()
            else:
                raise ValueError(f"Unsupported distance_transform '{dt.__class__.__name__}' in MACE model")
            distance_transform.load_state_dict(
                {k: v.detach().cpu() for k, v in dt.state_dict().items()},
                strict=False,
            )

    # Configure Curator-native multi-domain readout with a *single* domain (domain-agnostic).
    # Note: Curator heads are output specs (energy/forces/etc), not related to MACE heads.

    readout_head = HeadConfig(
        key=properties.energy,
        is_atomwise=True,
        reduction="sum",
        atomwise_key=properties.atomic_energy,
        write_atomwise=False,
        dim=1,
    )
    readout = partial(
        MultiDomainMACEAtomwiseNN,
        domains=domains,
        heads=[readout_head],
        activation=torch.nn.functional.silu,  # match MACE normalize2mom(silu) behavior
    )

    # Per-head MLP size needed for remove_pt_head-style slicing.
    try:
        from e3nn import o3 as _o3
        total_mlp_hidden = _o3.Irreps(getattr(mace_model.readouts[-1], "hidden_irreps", "16x0e"))
        total_mul = int(total_mlp_hidden.count(_o3.Irrep(0, 1)))
        per_head_mul = total_mul // num_mace_heads if (num_mace_heads > 0 and total_mul % num_mace_heads == 0) else total_mul
        mlp_count_irreps = int(per_head_mul)
        mlp_irreps = _o3.Irreps(f"{mlp_count_irreps}x0e")
    except Exception:
        mlp_count_irreps = 0
        mlp_irreps = getattr(mace_model.readouts[-1], "hidden_irreps", None)
    # Extract correlation values from each product layer
    # Each product has one or more contractions, we take the first one's correlation
    correlation_list = [
        prod.symmetric_contractions.contractions[0].correlation 
        for prod in mace_model.products
    ]
    
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
        readout=readout,
        heads=[readout_head],
    )

    # Copy MACE readout weights into Curator MACEAtomwiseNN modules.
    # Each Curator domain gets a single-head equivalent of the corresponding MACE head,
    # using the same slicing rules as mace.tools.scripts_utils.remove_pt_head.

    def _map_mace_mlp_state_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mapped: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            nk = k
            nk = nk.replace("linear_1.", "0.linear.")
            nk = nk.replace("linear_2.", "1.linear.")
            mapped[nk] = v
        return mapped

    def _squeeze_if_compatible(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        if src.shape != target_shape and src.dim() == len(target_shape) + 1 and src.shape[0] == 1:
            return src.squeeze(0)
        return src

    def _prepare_state_for_loading(dst: torch.nn.Module, sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dst_sd = dst.state_dict()
        out: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if torch.is_tensor(v):
                v = v.detach().cpu()
            if k in dst_sd:
                out[k] = _squeeze_if_compatible(v, dst_sd[k].shape)
            else:
                out[k] = v
        return out

    if not hasattr(curator_mace.readout, "domain_modules"):
        raise TypeError("Curator MACE readout is not multi-domain; expected MultiDomainMACEAtomwiseNN.")
    for dom, hidx in domain_to_head_idx.items():
        dst_domain = curator_mace.readout.domain_modules[str(dom)]
        for i, src_readout in enumerate(mace_model.readouts):
            dst_readout = dst_domain.readouts[i]
            sd = {k: v.detach().cpu() for k, v in src_readout.state_dict().items()}
            if num_mace_heads > 1:
                sd = _slice_mace_readout_state(
                    sd,
                    head_idx=hidx,
                    num_heads=num_mace_heads,
                    mlp_count_irreps=mlp_count_irreps,
                )
            if i == len(mace_model.readouts) - 1:
                sd = _map_mace_mlp_state_keys(sd)
            sd = _prepare_state_for_loading(dst_readout, sd)
            _load_state_dict_by_shape(dst_readout, sd)
    output_modules = []
    if hasattr(mace_model, "pair_repulsion_fn"):
        pair_fn = ZBLBasis()
        pair_fn.load_state_dict(
            {k: v.detach().cpu() for k, v in mace_model.pair_repulsion_fn.state_dict().items()},
            strict=False,
        )
        output_modules.append(PairRepulsionEnergy(pair_fn, atomic_numbers=curator_mace.atomic_numbers))

    # Build per-domain scale/shift/E0 from MACE scale_shift and atomic energies.
    scale_all = torch.atleast_1d(getattr(mace_model.scale_shift, "scale", torch.tensor([1.0]))).detach().cpu()
    shift_all = torch.atleast_1d(getattr(mace_model.scale_shift, "shift", torch.tensor([0.0]))).detach().cpu()
    atomic_energies_all = getattr(mace_model.atomic_energies_fn, "atomic_energies", None)
    md_rescale = MultiDomainRescaleShift(heads=[{"key": properties.energy}])

    def _pick_head_value(vec: torch.Tensor, idx: int) -> float:
        v = vec.reshape(-1)
        if len(heads) > 0 and v.numel() == len(heads):
            return float(v[idx].item())
        return float(v[0].item())

    def _atomic_energies_for_head(idx: int) -> torch.Tensor:
        ae = atomic_energies_all
        if ae is None:
            return torch.zeros((len(mace_model.atomic_numbers),), dtype=torch.get_default_dtype())
        if torch.is_tensor(ae) and ae.ndim > 1 and len(heads) > 0:
            if ae.shape[0] == len(heads):
                ae = ae[idx]
            elif ae.shape[1] == len(heads):
                ae = ae[:, idx]
            else:
                ae = ae[0]
        ae_t = ae if torch.is_tensor(ae) else torch.as_tensor(ae)
        return ae_t.detach().cpu().squeeze()

    for dom, hidx in domain_to_head_idx.items():
        ae = _atomic_energies_for_head(hidx)
        ae_list = ae.reshape(-1).tolist()
        energy_head = HeadConfig(
            key=properties.energy,
            is_atomwise=True,
            reduction="sum",
            atomwise_key=properties.atomic_energy,
            write_atomwise=False,
            dim=1,
            scale_by=_pick_head_value(scale_all, hidx),
            shift_by=_pick_head_value(shift_all, hidx),
            atomwise_shift=False,
            atomwise_normalization=True,
            per_species_shift={int(z): float(e) for z, e in zip(mace_model.atomic_numbers, ae_list)},
        )
        md_rescale.domain_modules[str(dom)] = GlobalRescaleShift(heads=[energy_head])

    output_modules.extend(
        [
            md_rescale,
            GradientOutput(model_outputs=['energy', 'forces'], grad_on_edge_diff=True, grad_on_positions=False),
        ]
    )
    curator_mace.embeddings.radial_basis.basis.load_state_dict(
        {k: v.detach().cpu() for k, v in mace_model.radial_embedding.bessel_fn.state_dict().items()},
        strict=False,
    )
    curator_mace.embeddings.chemical_embedding.linear.load_state_dict(
        {k: v.detach().cpu() for k, v in mace_model.node_embedding.linear.state_dict().items()},
        strict=False,
    )
    for i in range(len(mace_model.interactions)):
        curator_mace.interactions[i].avg_num_neighbors = torch.tensor(mace_model.interactions[i].avg_num_neighbors)
        _load_state_dict_by_shape(
            curator_mace.interactions[i],
            {k: v.detach().cpu() for k, v in mace_model.interactions[i].state_dict().items()},
        )
        _load_state_dict_by_shape(
            curator_mace.products[i],
            {k: v.detach().cpu() for k, v in mace_model.products[i].state_dict().items()},
        )
    nnp = NeuralNetworkPotential(
        input_modules=input_modules,
        representation=curator_mace,
        output_modules=output_modules,
    )
    try:
        target_dtype = next(mace_model.parameters()).dtype
        nnp = nnp.to(dtype=target_dtype)
    except StopIteration:
        pass

    return nnp


def _slice_mace_readout_state(
    state_dict: Dict[str, torch.Tensor],
    head_idx: int,
    num_heads: int,
    mlp_count_irreps: int,
) -> Dict[str, torch.Tensor]:
    sliced: Dict[str, torch.Tensor] = {}
    for name, param in state_dict.items():
        # Collapse a multi-head MACE readout module down to a single head.
        # Mirrors mace.tools.scripts_utils.remove_pt_head slicing rules.
        if "linear_1.weight" in name and mlp_count_irreps:
            sliced[name] = param.reshape(-1, num_heads, mlp_count_irreps)[:, head_idx, :].flatten()
        elif "linear_2.weight" in name:
            sliced[name] = (
                param.reshape(num_heads, -1, num_heads)[head_idx, :, head_idx].flatten()
                / (num_heads ** 0.5)
            )
        elif "linear.weight" in name:
            sliced[name] = param.reshape(-1, num_heads)[:, head_idx].flatten()
        elif "output_mask" in name and param.numel() == num_heads:
            sliced[name] = param[head_idx : head_idx + 1]
        elif "bias" in name and param.numel() > 0 and param.numel() % num_heads == 0:
            sliced[name] = param.reshape(-1, num_heads)[:, head_idx].flatten()
        else:
            sliced[name] = param
    return sliced


def _load_state_dict_by_shape(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load only matching parameter shapes to tolerate version skew."""
    current = module.state_dict()
    filtered = {}
    for name, param in state_dict.items():
        if name in current and current[name].shape == param.shape:
            filtered[name] = param
    module.load_state_dict(filtered, strict=False)


def _load_official_nequip_saved_model(
    model_ref: Union[str, Path],
    compile_mode: str = "eager",
):
    import sys

    nequip_src = Path.home() / "local" / "src" / "nequip"
    if nequip_src.exists():
        nequip_src_str = str(nequip_src)
        if nequip_src_str not in sys.path:
            sys.path.insert(0, nequip_src_str)

    torch_serialization.add_safe_globals([slice])
    from nequip.model.saved_models.load_utils import load_saved_model

    return load_saved_model(str(model_ref), compile_mode=compile_mode)


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
    return [
        max(int(mul) for mul, ir in irreps if ir.l == l)
        for l in range(lmax + 1)
    ]


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


def _map_official_nequip_state_dict(nequip_model, curator_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    graph_model, func = _unwrap_official_nequip_model(nequip_model)
    official_sd = {
        k.replace("model.func.", "", 1).replace("func.", "", 1): v.detach().cpu()
        for k, v in graph_model.state_dict().items()
    }

    mapped: Dict[str, torch.Tensor] = {}
    pair_module_index = next(
        (
            idx
            for idx, module in enumerate(getattr(curator_model, "output_modules", []))
            if module.__class__.__name__ == "NequIPZBLPairEnergy"
        ),
        None,
    )

    species = list(func.per_type_energy_scale_shift.type_names)
    output_modules = list(getattr(curator_model, "output_modules", []))
    if output_modules and output_modules[0].__class__.__name__ == "PerSpeciesRescaleShift":
        per_species_module = output_modules[0]
        scales = torch.ones_like(per_species_module.scales)
        shifts = torch.zeros_like(per_species_module.shifts)
        official_scales = getattr(func.per_type_energy_scale_shift, "scales", None)
        official_shifts = getattr(func.per_type_energy_scale_shift, "shifts", None)
        if official_scales is not None:
            official_scales = official_scales.detach().cpu().reshape(-1)
            for idx, symbol in enumerate(species):
                z = chemical_symbols.index(symbol)
                scales[z] = official_scales[idx]
        if official_shifts is not None:
            official_shifts = official_shifts.detach().cpu().reshape(-1)
            for idx, symbol in enumerate(species):
                z = chemical_symbols.index(symbol)
                shifts[z] = official_shifts[idx]
        mapped["output_modules.0.scales"] = scales
        mapped["output_modules.0.shifts"] = shifts

    for key, value in official_sd.items():
        if key in {"_empty"}:
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
                suffix = suffix.lstrip("_")
                mapped[f"output_modules.{pair_module_index}.{suffix}"] = value
            continue

    return mapped


def create_model_from_nequip(nequip_model) -> torch.nn.Module:
    from functools import partial
    from curator.data.properties import HeadConfig
    from curator.layer import (
        AtomwiseNN,
        AtomwiseReduce,
        GradientOutput,
        NequIPZBLPairEnergy,
        PairwiseDistance,
        PerSpeciesRescaleShift,
    )
    from curator.model import NeuralNetworkPotential, Nequip

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
            "Curator NequIP converter currently supports official NequIP packages with uniform radial MLP depth/width across layers."
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
    readout = partial(AtomwiseNN, heads=[readout_head])

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
        readout=readout,
        heads=[readout_head],
    )

    output_modules: List[torch.nn.Module] = []
    per_species_scales = _build_species_value_dict(
        list(func.per_type_energy_scale_shift.type_names),
        getattr(func.per_type_energy_scale_shift, "scales", None),
    )
    per_species_shifts = _build_species_value_dict(
        list(func.per_type_energy_scale_shift.type_names),
        getattr(func.per_type_energy_scale_shift, "shifts", None),
    )
    output_modules.append(
        PerSpeciesRescaleShift(
            scales=per_species_scales,
            shifts=per_species_shifts,
            scales_keys=[properties.atomic_energy],
            shifts_keys=[properties.atomic_energy],
        )
    )
    output_modules.append(AtomwiseReduce(output_key=properties.energy, aggregation_mode="sum"))

    pair_potential = getattr(func, "pair_potential", None)
    if pair_potential is not None:
        pair_cutoff = getattr(pair_potential, "cutoff", None)
        pair_power = _scalar_to_int(getattr(pair_cutoff, "p", None), power)
        output_modules.append(
            NequIPZBLPairEnergy(
                atomic_numbers=pair_potential.atomic_numbers.detach().cpu(),
                cutoff=float(func.edge_norm.r_max),
                power=pair_power,
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


def convert_mace_to_curator(
    mace_path: Union[str, Path],
    output_path: Union[str, Path],
    head: Optional[Union[str, int]] = None,
    device: Optional[torch.device] = None,
) -> Path:
    """Load a mace model checkpoint and save a curator-style model."""
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
            raise TypeError("MACE checkpoint does not include an instantiated model; please provide a TorchScript or full model checkpoint.")
    if mace_model is None:
        raise TypeError(f"Unsupported MACE checkpoint format at {mace_path}")
    curator_model = create_model_from_mace(mace_model, head=head)
    output_path = Path(output_path)
    torch.save(curator_model, output_path)
    return output_path

def _build_mace_from_curator(curator_model):
    from curator.layer import GlobalRescaleShift, MultiDomainRescaleShift
    from curator.model import NeuralNetworkPotential, MACE
    """Best-effort recreation of a mace.modules.models.ScaleShiftMACE from a Curator MACE model."""
    try:
        from mace.modules import models as mace_models
        from mace.modules import blocks as mace_blocks
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import mace modules; converting to original MACE requires mace "
            "and its cuequivariance dependencies (typically CUDA-enabled)."
        ) from exc

    if isinstance(curator_model, NeuralNetworkPotential):
        repr_model = curator_model.representation
        output_modules = list(curator_model.output_modules)
    else:
        repr_model = curator_model
        output_modules = []

    if not isinstance(repr_model, MACE):
        raise TypeError("Provided model is not a Curator MACE representation.")

    # species / atomic numbers
    atomic_numbers = list(range(repr_model.embeddings.onehot_embedding.num_elements))
    mapper = getattr(repr_model.embeddings.onehot_embedding, "type_mapper", None)
    if mapper is not None:
        atomic_numbers = [int(z) for z in mapper.index_to_Z.cpu().tolist()]

    heads = [f"head_{i}" for i in range(repr_model.num_heads)]
    correlation = [
        contraction.correlation for contraction in repr_model.products[0].symmetric_contractions.contractions
    ]
    num_basis = len(repr_model.embeddings.radial_basis.basis.bessel_weights)
    num_polynomial_cutoff = int(repr_model.embeddings.radial_basis.cutoff_fn.power)
    avg_num_neighbors = float(repr_model.interactions[0].avg_num_neighbors.squeeze())
    distance_transform = "None"
    dt = getattr(repr_model.embeddings.radial_basis, "distance_transform", None)
    if dt is not None:
        from curator.layer import AgnesiTransform, SoftTransform
        if isinstance(dt, AgnesiTransform):
            distance_transform = "Agnesi"
        elif isinstance(dt, SoftTransform):
            distance_transform = "Soft"
        else:
            distance_transform = dt.__class__.__name__

    scale, shift = 1.0, 0.0
    atomic_energies = torch.zeros(len(atomic_numbers))
    num_heads = getattr(repr_model, "num_heads", 1)

    def _pick_scalar(val):
        if isinstance(val, list):
            return val[0] if val else 0.0
        if torch.is_tensor(val):
            v = val.detach().cpu().reshape(-1)
            return v[0].item() if v.numel() > 0 else 0.0
        return val

    for m in output_modules:
        if isinstance(m, MultiDomainRescaleShift):
            scales = []
            shifts = []
            energies = []
            for i in range(num_heads):
                dom = str(i)
                grs = m.domain_modules.get(dom)
                if grs is None:
                    grs = next(iter(m.domain_modules.values()))
                scales.append(_pick_scalar(grs.scale_by))
                shifts.append(_pick_scalar(grs.shift_by))
                ae = grs.atomic_energies
                if ae is None:
                    ae = torch.zeros(len(atomic_numbers))
                if not torch.is_tensor(ae):
                    ae = torch.as_tensor(ae)
                energies.append(ae)
            scale = scales
            shift = shifts
            atomic_energies = torch.stack(energies, dim=0)
            z_idx = torch.as_tensor(atomic_numbers, dtype=torch.long)
            atomic_energies = torch.index_select(atomic_energies, -1, z_idx)
            break
        if isinstance(m, GlobalRescaleShift):
            scale = _pick_scalar(m.scale_by)
            shift = _pick_scalar(m.shift_by)
            atomic_energies = m.atomic_energies
            if atomic_energies is None:
                break
            if not torch.is_tensor(atomic_energies):
                atomic_energies = torch.as_tensor(atomic_energies)
            z_idx = torch.as_tensor(atomic_numbers, dtype=torch.long)
            atomic_energies = torch.index_select(atomic_energies, -1, z_idx)
            break

    mace_model = mace_models.ScaleShiftMACE(
        atomic_inter_scale=scale,
        atomic_inter_shift=shift,
        r_max=float(repr_model.cutoff),
        num_bessel=num_basis,
        num_polynomial_cutoff=num_polynomial_cutoff,
        max_ell=repr_model.lmax,
        interaction_cls=mace_blocks.InteractionBlock,
        interaction_cls_first=mace_blocks.InteractionBlock,
        num_interactions=len(repr_model.interactions),
        num_elements=repr_model.embeddings.onehot_embedding.num_elements,
        hidden_irreps=repr_model.hidden_irreps,
        MLP_irreps=repr_model.MLP_irreps,
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=atomic_numbers,
        correlation=correlation,
        gate=torch.nn.functional.silu,
        heads=heads,
        distance_transform=distance_transform,
    )

    # load weights (best effort, shapes match original MACE layout)
    mace_model.radial_embedding.bessel_fn.load_state_dict(
        repr_model.embeddings.radial_basis.basis.state_dict(), strict=False
    )
    mace_model.radial_embedding.cutoff_fn.load_state_dict(
        repr_model.embeddings.radial_basis.cutoff_fn.state_dict(), strict=False
    )
    mace_model.node_embedding.linear.load_state_dict(
        repr_model.embeddings.chemical_embedding.linear.state_dict(), strict=False
    )
    for i in range(len(repr_model.interactions)):
        mace_model.interactions[i].load_state_dict(repr_model.interactions[i].state_dict(), strict=False)
        mace_model.products[i].load_state_dict(repr_model.products[i].state_dict(), strict=False)
        if i < len(mace_model.readouts) - 1 and hasattr(mace_model.readouts[i], "linear"):
            mace_model.readouts[i].linear.load_state_dict(repr_model.readout_mlp[i].state_dict(), strict=False)
        elif i < len(mace_model.readouts):
            mace_model.readouts[i].load_state_dict(repr_model.readout_mlp[i].state_dict(), strict=False)
    return mace_model


def convert_curator_to_mace(curator_path: Union[str, Path], output_path: Union[str, Path], device: Optional[torch.device] = None) -> Path:
    """Convert a saved Curator MACE model back to a mace.modules.models.ScaleShiftMACE checkpoint."""
    curator_path = Path(curator_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(curator_path, device=device, load_compiled=False, load_weights_only=False)
    mace_model = _build_mace_from_curator(model)
    output_path = Path(output_path)
    torch.save(mace_model, output_path)
    return output_path

def dummy_load(*args, **kwargs):
    original_torch_jit_load = torch.jit.load
    def torch_jit_load_cpu(*args, **kwargs):
        if not torch.cuda.is_available():
            kwargs['map_location'] = torch.device('cpu')
        return original_torch_jit_load(*args, **kwargs)
    torch.jit.load = torch_jit_load_cpu

def camel_to_snake(name: str) -> str:
    s1 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return s1.lower()

def split_list(lst, chunk_or_num, by_chunk_size=False):
    if by_chunk_size:
        num_chunks, remainder = divmod(len(lst), chunk_or_num)
    else:
        chunk_or_num, remainder = divmod(len(lst), chunk_or_num)
    if by_chunk_size:
        return [
            lst[i * chunk_or_num + min(i, remainder):(i + 1) * chunk_or_num + min(i + 1, remainder)]
            for i in range(num_chunks)
        ]
    else:
        return [
            lst[i * (chunk_or_num + (1 if i < remainder else 0)):(i + 1) * (chunk_or_num + (1 if i < remainder else 0))]
            for i in range(chunk_or_num)
        ]

def _copy_config(config_like: Optional[Union[DictConfig, dict]]) -> Optional[DictConfig]:
    """Return a mutable DictConfig copy for easier manipulation."""

    if config_like is None:
        return None
    if isinstance(config_like, DictConfig):
        return OmegaConf.create(OmegaConf.to_container(config_like, resolve=False))
    if isinstance(config_like, dict):
        return OmegaConf.create(config_like)
    return None


def _listify_config_field(config: Optional[DictConfig], field: str) -> None:
    if config is None or field not in config:
        return
    value = config[field]
    if isinstance(value, DictConfig):
        config[field] = [value[k] for k in value.keys()]
    elif isinstance(value, dict):
        config[field] = list(value.values())
    elif isinstance(value, ListConfig):
        config[field] = list(value)


def load_trained_model(
    model_file: Union[str, Path],
    device = None,
    load_compiled: bool = True,
    load_weights_only: bool = False,
    cfg: Optional[DictConfig] = None,
) -> torch.nn.Module:
    """Load a trained model or checkpoint and return a torch.nn.Module."""

    model_file = Path(model_file)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _torch_load(model_file: Path, device, load_weights_only: bool):
        try:
            return torch.load(model_file, map_location=torch.device(device), weights_only=load_weights_only)
        except TypeError:
            return torch.load(model_file, map_location=torch.device(device))

    # TorchScript
    if model_file.suffix == '.pt' and load_compiled:
        try:
            model = torch.jit.load(model_file, map_location=torch.device(device))
        except RuntimeError as exc:
            if "cuda" in str(exc).lower() and str(device).startswith("cuda"):
                device = torch.device("cpu")
                model = torch.jit.load(model_file, map_location=device)
            else:
                raise
        try:
            model.to(device)
        except Exception:
            pass
        return model

    try:
        obj = _torch_load(model_file, device, load_weights_only)
    except RuntimeError as exc:
        if "cuda" in str(exc).lower() and str(device).startswith("cuda"):
            device = torch.device("cpu")
            obj = _torch_load(model_file, device, load_weights_only)
        else:
            raise

    if isinstance(obj, torch.nn.Module):
        obj.to(device)
        return obj

    if isinstance(obj, dict):
        stored_model = obj.get('model')
        if not load_weights_only and isinstance(stored_model, torch.nn.Module):
            stored_model.to(device)
            return stored_model

        model_cfg = cfg.model if cfg is not None else obj.get('model_params')
        model_cfg = _copy_config(model_cfg)
        if model_cfg is None:
            raise ValueError("Checkpoint does not contain model parameters to instantiate.")
        _listify_config_field(model_cfg, "input_modules")
        _listify_config_field(model_cfg, "output_modules")
        model = instantiate(model_cfg, _convert_="all")

        data_cfg = cfg.data if cfg is not None else obj.get('data_params')
        data_cfg = _copy_config(data_cfg)
        if data_cfg is not None:
            datamodule = instantiate(data_cfg, _convert_="all")
            if hasattr(datamodule, 'setup'):
                datamodule.setup()
            if hasattr(model, 'initialize_modules'):
                model.initialize_modules(datamodule)

        sd = obj.get('state_dict')
        if sd is None:
            raise ValueError("Checkpoint is missing a state_dict.")
        stripped = {k.replace("model.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(stripped, strict=False)
        model.to(device)
        return model

    raise TypeError(f"Unsupported checkpoint format at {model_file}.")


def load_model(
    model_file: Union[str, Path],
    device = None,
    load_compiled: bool = True,
    load_weights_only: bool = False,
    cfg: Optional[DictConfig] = None,
) -> torch.nn.Module:
    if isinstance(model_file, str):
        from curator.model.adapters import is_external_model_spec, load_external_model

        if is_external_model_spec(model_file):
            return load_external_model(model_file, device=device)
    return load_trained_model(
        model_file,
        device=device,
        load_compiled=load_compiled,
        load_weights_only=load_weights_only,
        cfg=cfg,
    )

def load_models(
    model_like: Union[str, Path, torch.nn.Module, List[Union[str, Path, torch.nn.Module]]],
    device = None,
    load_compiled: bool = True,
    load_weights_only: bool = False,
    cfg: Optional[DictConfig] = None,
) -> List[torch.nn.Module]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize OmegaConf list/tuple containers into plain Python lists for type checks.
    if isinstance(model_like, ListConfig):
        model_like = list(model_like)

    # for single model passed with str
    if not isinstance(model_like, (list, tuple)):
        model_like = [model_like]
    
    # for list of models passed with nn.Module
    if all(isinstance(m, torch.nn.Module) for m in model_like):
        models = list(model_like)
        for m in models:
            try:
                m.to(device)
            except Exception:
                pass
        return models
    
    # paths or run dirs
    models: List[torch.nn.Module] = []
    for m in model_like:
        if isinstance(m, (str, Path)):
            if isinstance(m, str):
                from curator.model.adapters import is_external_model_spec, load_external_model

                if is_external_model_spec(m):
                    models.append(load_external_model(m, device=device))
                    continue
            p = Path(m)
            if p.is_file():
                models.append(
                    load_model(
                        p,
                        device,
                        load_compiled,
                        load_weights_only=load_weights_only,
                        cfg=cfg,
                    )
                )
                continue
            best_info = find_best_model(p)
            if best_info is None:
                raise FileNotFoundError(
                    f"Could not find a model file in '{p}'. Expected a file path or a run directory."
                )
            best, _ = best_info
            models.append(
                load_model(
                    best,
                    device,
                    load_compiled,
                    load_weights_only=load_weights_only,
                    cfg=cfg,
                )
            )
        else:
            raise TypeError("List elements must be all nn.Module or all str/Path.")
    
    return models

def update_model_domain_config(
    model_cfg: DictConfig,
    domain_mode: Literal["extend", "replace"],
    new_domains,
    logger=None,
) -> Optional[List[str]]:
    """Update model.representation.readout.domains based on domain_mode."""
    if model_cfg is None or domain_mode is None or new_domains is None:
        return None
    try:
        rep_cfg = model_cfg.representation
    except Exception:
        if logger:
            logger.warning("Model config missing representation; cannot set domains.")
        return None
    if rep_cfg is None or not hasattr(rep_cfg, "readout"):
        if logger:
            logger.warning("Model config missing representation.readout; cannot set domains.")
        return None
    domains = [str(d) for d in ensure_list(new_domains)]
    if not domains:
        return None
    readout_cfg = rep_cfg.readout
    existing = []
    if hasattr(readout_cfg, "get"):
        existing = readout_cfg.get("domains", []) or []
    elif hasattr(readout_cfg, "domains"):
        existing = readout_cfg.domains or []
    existing = [str(d) for d in ensure_list(existing)] if existing else []
    if domain_mode == "extend":
        merged = existing[:]
        for dom in domains:
            if dom not in merged:
                merged.append(dom)
        domains = merged
    elif domain_mode != "replace":
        if logger:
            logger.warning("Unknown domain_mode=%s; expected 'extend' or 'replace'.", domain_mode)
        return None
    with open_dict(rep_cfg):
        rep_cfg.readout.domains = domains
    if logger:
        logger.debug("Set readout domains to %s (mode=%s).", domains, domain_mode)
    return domains

def update_model_domains(
    model: torch.nn.Module,
    new_domains,
    mode: Literal["extend", "replace"] = "extend",
    template_domain: str = "0",
    init_strategy: Literal["random", "copy"] = "random",
    logger=None,
) -> int:
    """Extend or replace domain_modules in-place for a model."""
    domains = [str(d) for d in ensure_list(new_domains or [])]
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

    def _reset_params(mod: torch.nn.Module) -> None:
        for sub in mod.modules():
            reset = getattr(sub, "reset_parameters", None)
            if callable(reset):
                reset()

    updated = 0
    for module in model.modules():
        domain_modules = getattr(module, "domain_modules", None)
        if domain_modules is None or not hasattr(domain_modules, "items"):
            continue
        template_key = str(template_domain)
        template_mod = None
        if template_key in domain_modules:
            template_mod = domain_modules[template_key]
        elif len(domain_modules) > 0:
            template_key, template_mod = next(iter(domain_modules.items()))
        if mode == "replace":
            for k in list(domain_modules.keys()):
                if k not in domains:
                    del domain_modules[k]
        for dom in domains:
            if dom in domain_modules:
                continue
            if template_mod is None:
                if logger:
                    logger.warning("No template domain found for module %s; skipping %s.", module.__class__.__name__, dom)
                continue
            new_mod = copy.deepcopy(template_mod)
            if init_strategy == "random":
                _reset_params(new_mod)
            domain_modules[dom] = new_mod
            updated += 1
        if hasattr(module, "domains"):
            if mode == "replace":
                module.domains = domains[:]
            else:
                existing = [str(d) for d in (getattr(module, "domains") or [])]
                for dom in domains:
                    if dom not in existing:
                        existing.append(dom)
                module.domains = existing
    return updated

def ensure_list(value: Any):
    """Convert dictionary-like Hydra nodes to list values."""

    if isinstance(value, DictConfig):
        return [value[k] for k in value.keys()]
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, ListConfig):
        return list(value)
    return value

def ensure_dict(value: Any, prefix: str = "item"):
    """Convert legacy list-style Hydra nodes to dictionaries."""

    if isinstance(value, DictConfig):
        return value
    if isinstance(value, (ListConfig, list)):
        items = {}
        for idx, entry in enumerate(value):
            key = _infer_sequence_key(entry, idx, prefix)
            if key in items:
                key = f"{key}_{idx}"
            items[key] = entry
        return OmegaConf.create(items)
    return value

def _camel_to_snake(name: str) -> str:
    import re
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

def _infer_sequence_key(entry: Any, idx: int, prefix: str) -> str:
    if isinstance(entry, (DictConfig, dict)):
        name = entry.get("name")
        if isinstance(name, str) and name:
            return name
        target = entry.get("_target_")
        if isinstance(target, str) and target:
            return _camel_to_snake(target.split(".")[-1])
    return f"{prefix}_{idx}"

def find_best_model(run_path: Union[str, Path]) -> Tuple[Path, Optional[float]]:
    """Return best ckpt path under a run directory or the path itself if it is a .ckpt."""

    run_path = Path(run_path)
    if run_path.suffix == '.ckpt':
        return run_path, None

    cands = list(run_path.glob("best_model_*.ckpt"))
    if cands:
        best_p, best_v = None, float('inf')
        for p in cands:
            try:
                v = float(str(p).split('=')[-1].rstrip('.ckpt'))
            except Exception:
                continue
            if v < best_v:
                best_v, best_p = v, p
        if best_p is not None:
            return best_p, best_v
    
    # return newest .ckpt if no best_model_*.ckpt is there
    all_ckpts = sorted(run_path.glob("*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if all_ckpts:
        return all_ckpts[0], None

class CustomFormatter(logging.Formatter):
    format = "%(asctime)s: %(message)s"
    time_format = "%Y-%m-%d %H:%M:%S"
     
    FORMATS = {
        logging.DEBUG: format,
        logging.INFO: "%(message)s",
        logging.WARNING: format,
        logging.ERROR: format,
        logging.CRITICAL: format
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, self.time_format)
        return formatter.format(record)


_LOGO_LOGGED = False


def log_logo(logger: Optional[logging.Logger] = None) -> None:
    global _LOGO_LOGGED
    if _LOGO_LOGGED:
        return
    _LOGO_LOGGED = True
    log = logger or logging.getLogger("curator")
    logo = [
        """
            █████████  █████  █████ ███████████     █████████   ███████████    ███████    ███████████  
           ███░░░░░███░░███  ░░███ ░░███░░░░░███   ███░░░░░███ ░█░░░███░░░█  ███░░░░░███ ░░███░░░░░███ 
          ███     ░░░  ░███   ░███  ░███    ░███  ░███    ░███ ░   ░███  ░  ███     ░░███ ░███    ░███ 
         ░███          ░███   ░███  ░██████████   ░███████████     ░███    ░███      ░███ ░██████████  
         ░███          ░███   ░███  ░███░░░░░███  ░███░░░░░███     ░███    ░███      ░███ ░███░░░░░███ 
         ░░███     ███ ░███   ░███  ░███    ░███  ░███    ░███     ░███    ░░███     ███  ░███    ░███ 
          ░░█████████  ░░████████   █████   █████ █████   █████    █████    ░░░███████░   █████   █████
           ░░░░░░░░░    ░░░░░░░░   ░░░░░   ░░░░░ ░░░░░   ░░░░░    ░░░░░       ░░░░░░░    ░░░░░   ░░░░░

                           Active learning for machine learning interatomic potentials
        """,
    ]
    display_lines = [line.replace("\\\\", "\\") for line in logo]
    width = max(max(len(line) for line in display_lines), 80)
    for line in display_lines:
        log.info(line.center(width))

# Set up Early stopping for pytorch training 
class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, best_loss):
        if val_loss - best_loss > self.min_delta:
            self.counter +=1
            if self.counter >= self.patience:  
                self.early_stop = True
        return self.early_stop

def deploy_model(model, file_path: str):
    compiled_model = script(model)
    compiled_model.save()

# Auxiliary function for parsing config file 
def get_all_pairs(d, keys=()):
    if isinstance(d, abc.Mapping):
         for k in d:
            for rv in get_all_pairs(d[k], keys + (k, )):
                yield rv
    else:
        yield (keys, d)

def _dictify_field(
    container: Optional[DictConfig],
    key: str,
    prefix: str,
    path: str,
    converted: set,
) -> None:
    if container is None or key not in container or container[key] is None:
        return

    new_value = ensure_dict(container[key], prefix)
    if new_value is container[key]:
        return

    if isinstance(container, DictConfig):
        with open_dict(container):
            container[key] = new_value
    else:
        container[key] = new_value

    converted.add(path)


def _dictify_sequence_nodes(config: Optional[DictConfig]) -> set:
    converted = set()
    if config is None:
        return converted

    if "trainer" in config:
        _dictify_field(config.trainer, "callbacks", "callback", "trainer.callbacks", converted)

    if "model" in config:
        _dictify_field(config.model, "input_modules", "input_module", "model.input_modules", converted)
        _dictify_field(config.model, "output_modules", "output_module", "model.output_modules", converted)

    if "task" in config:
        _dictify_field(config.task, "outputs", "output", "task.outputs", converted)

    if "simulator" in config:
        _dictify_field(config.simulator, "callbacks", "callback", "simulator.callbacks", converted)

    return converted

def normalize_config_sequences(config: Optional[DictConfig]) -> None:
    """Convert configurable sequence fields to list form for easier consumption."""
    if config is None:
        return

    if "trainer" in config:
        _listify_config_field(config.trainer, "callbacks")

    if "model" in config:
        _listify_config_field(config.model, "input_modules")
        _listify_config_field(config.model, "output_modules")

    if "task" in config:
        _listify_config_field(config.task, "outputs")

    if "simulator" in config:
        _listify_config_field(config.simulator, "callbacks")


def prune_config_targets(config: Optional[DictConfig], logger: Optional[logging.Logger] = None) -> None:
    """
    Remove keys from config nodes that specify a _target_ but include arguments
    not accepted by the target's signature (unless it has **kwargs).
    Helps prevent stale parameters from other defaults (e.g., switching models/engines).
    """
    if config is None:
        return

    log = logger or logging.getLogger("curator")
    special_keys = {"_target_", "_partial_", "_recursive_", "_convert_"}

    def _prune(node: DictConfig, path: str = ""):
        if not isinstance(node, DictConfig):
            return

        target = node.get("_target_")
        if target:
            try:
                obj = get_class(str(target))
            except Exception:
                obj = None

            if obj is not None:
                sig = inspect.signature(obj.__init__ if inspect.isclass(obj) else obj)
                params = sig.parameters
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                    allowed = None
                else:
                    allowed = {name for name in params if name != "self"}

                if allowed is not None:
                    allowed.update(special_keys)
                    unknown = [k for k in node.keys() if k not in allowed]
                    if unknown:
                        with open_dict(node):
                            for k in unknown:
                                del node[k]
                        log.debug(f"Pruned keys {unknown} from config node '{path or '<root>'}' for target {target}")

        for k, v in node.items():
            if isinstance(v, DictConfig):
                _prune(v, f"{path}.{k}" if path else k)

    _prune(config)


def update_config_from_datamodule(
    config: DictConfig,
    datamodule: Any,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Update config based on datamodule contents (species and per-domain heads).
    Keeps logic centralized and minimal.
    """
    def _is_auto(value) -> bool:
        return value is None or (isinstance(value, str) and value.lower() == "auto")

    def _ensure_list(value, default=None) -> List:
        if value is None:
            return list(default or [])
        if isinstance(value, str):
            return [value]
        return list(value)

    def _update_heads_cfg(cfg: DictConfig, keypath: str, heads: List) -> None:
        heads_cfg = OmegaConf.select(cfg, keypath)
        if isinstance(heads_cfg, DictConfig) and "_target_" in heads_cfg:
            OmegaConf.update(cfg, f"{keypath}.heads", heads, force_add=True)
        else:
            OmegaConf.update(cfg, keypath, heads, force_add=True)

    def _update_rescale_heads(cfg: DictConfig, heads: List) -> None:
        output_modules = OmegaConf.select(cfg, "model.output_modules")
        if isinstance(output_modules, (DictConfig, dict)) and "global_rescale_shift" in output_modules:
            _update_heads_cfg(cfg, "model.output_modules.global_rescale_shift.heads", heads)
            return
        if isinstance(output_modules, (ListConfig, list)):
            for idx, item in enumerate(output_modules):
                if not isinstance(item, (DictConfig, dict)):
                    continue
                target = item.get("_target_")
                if target and "RescaleShift" in str(target):
                    _update_heads_cfg(cfg, f"model.output_modules.{idx}.heads", heads)
                    return

    # Update config.data.species from datamodule or contexts.
    if hasattr(datamodule, "species") and _is_auto(getattr(datamodule, "species", None)):
        inferred = datamodule._get_species()
        config.data.species = inferred
    elif not hasattr(datamodule, "species") and hasattr(datamodule, "build_contexts"):
        try:
            ctxs = datamodule.build_contexts([])
            if "global" in ctxs and ctxs["global"].species:
                inferred = ctxs["global"].species
                config.data.species = inferred
        except Exception:
            pass

    # Update heads for readout/rescale from explicit config (no inference).
    datapath = getattr(config.data, "datapath", None)
    data_heads = OmegaConf.select(config, "data.heads")
    if _is_auto(data_heads) or data_heads is None:
        data_heads = ["energy"]
    data_heads = _ensure_list(data_heads, default=["energy"])
    rescale_shift_heads = OmegaConf.select(config, "data.rescale_shift_heads")
    rescale_shift_heads = _ensure_list(rescale_shift_heads, default=[])

    readout_heads = OmegaConf.select(config, "model.representation.readout.heads")
    should_update_heads = _is_auto(readout_heads) or (
        isinstance(readout_heads, (ListConfig, list)) and list(readout_heads) == ["energy"]
    )
    if should_update_heads:
        if isinstance(datapath, (DictConfig, dict)):
            domain_heads = {
                str(name): _ensure_list(cfg["heads"], default=data_heads)
                for name, cfg in datapath.items()
                if isinstance(cfg, (DictConfig, dict)) and "heads" in cfg
            }
            if hasattr(datamodule, "domain_modules"):
                for name in datamodule.domain_modules.keys():
                    domain_heads.setdefault(str(name), list(data_heads))
            if not domain_heads:
                domain_heads = {"0": list(data_heads)}

            if hasattr(datamodule, "domain_to_id"):
                heads_by_domain = {
                    str(datamodule.domain_to_id.get(name, name)): heads
                    for name, heads in domain_heads.items()
                }
            else:
                heads_by_domain = {str(name): heads for name, heads in domain_heads.items()}

            domains = list(heads_by_domain.keys())
            OmegaConf.update(config, "model.representation.readout.heads_by_domain", heads_by_domain, force_add=True)
            OmegaConf.update(config, "model.representation.readout.domains", domains, force_add=True)

            rescale_heads = [
                {"key": key, "domains": [dom_id]}
                for dom_id, heads in heads_by_domain.items()
                for key in dict.fromkeys(list(heads) + list(rescale_shift_heads))
            ] or [{"key": "energy", "domains": domains}]
            _update_rescale_heads(config, rescale_heads)
        else:
            _update_heads_cfg(config, "model.heads", data_heads)
            _update_heads_cfg(config, "model.representation.readout.heads", data_heads)
            merged = list(dict.fromkeys(list(data_heads) + list(rescale_shift_heads)))
            rescale_heads = [{"key": key, "domains": ["0"]} for key in merged] or [{"key": "energy", "domains": ["0"]}]
            _update_rescale_heads(config, rescale_heads)

    # If we are not using multi-domain loaders, strip dataloader_idx suffixes.
    if not hasattr(datamodule, "domain_modules"):
        def _strip_idx(value: Optional[str]) -> Optional[str]:
            if isinstance(value, str) and value.endswith("/dataloader_idx_0"):
                return value.replace("/dataloader_idx_0", "")
            return value

        sched_monitor = OmegaConf.select(config, "task.scheduler_monitor")
        sched_monitor = _strip_idx(sched_monitor)
        if sched_monitor is not None:
            OmegaConf.update(config, "task.scheduler_monitor", sched_monitor, force_add=True)

        callbacks = OmegaConf.select(config, "trainer.callbacks")
        if isinstance(callbacks, (ListConfig, list)):
            for idx, cb in enumerate(callbacks):
                if not isinstance(cb, (DictConfig, dict)):
                    continue
                monitor = cb.get("monitor", None)
                monitor = _strip_idx(monitor)
                if monitor is not None:
                    OmegaConf.update(config, f"trainer.callbacks.{idx}.monitor", monitor, force_add=True)

    if hasattr(datamodule, "log_summary"):
        summary = datamodule.log_summary()
        if summary:
            (logger or logging.getLogger(__name__)).info("%s", summary)

# Ugly workaround for specifying config files outside of the package
def read_user_config(
    cfg: Union[DictConfig, PosixPath, str, None]=None,
    config_path="configs",
    config_name="train.yaml",
    overrides: Optional[Union[str, List[str]]] = None,
):
    # load cfg
    if isinstance(cfg, DictConfig):
        user_cfg = cfg.copy()
    elif isinstance(cfg, (PosixPath, str)):
        user_cfg = OmegaConf.load(cfg)
    else:
        user_cfg = OmegaConf.create()

    converted_fields = set()
    if isinstance(user_cfg, DictConfig):
        converted_fields = _dictify_sequence_nodes(user_cfg)

    config_path_obj = Path(config_path)
    use_config_dir = config_path_obj.is_absolute()
    if not use_config_dir:
        pkg_base = Path(__file__).resolve().parent
        candidate = (pkg_base / config_path_obj).resolve()
        if candidate.exists():
            config_path_obj = candidate
            use_config_dir = True
        else:
            candidate = (Path.cwd() / config_path_obj).resolve()
            if candidate.exists():
                config_path_obj = candidate
                use_config_dir = True
    config_path = str(config_path_obj)

    override_list = []
    if "defaults" in user_cfg:
        default_list = user_cfg.pop("defaults")
        for d in default_list:
            if isinstance(d, (dict, DictConfig)):
                for k, v in d.items():
                    override_list.append(f"{k}={v}")
    
    for path in sorted(converted_fields):
        override_list.append(f"~{path}")

    deferred_updates = []
    for k, v in get_all_pairs(user_cfg):
        key = ".".join(k)
        if isinstance(v, (DictConfig, ListConfig, dict, list)):
            deferred_updates.append((key, v))
            continue
        # process value
        value = str(escape_all(v)).replace("'", "")
        if value == 'None':
            value = 'null'
        override_list.append(f'++{key}={value}')
    
    # command line overrides
    try:
        cli_overrides = hydra.core.hydra_config.HydraConfig.get().overrides.task
    except:
        cli_overrides = []
    finally:
        override_list.extend(cli_overrides)

    if overrides is not None:
        if isinstance(overrides, str):
            overrides = [overrides]
        override_list.extend(overrides)

    # reload hyperparameters         
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    if use_config_dir:
        context = initialize_config_dir(version_base=None, config_dir=config_path)
    else:
        context = initialize(version_base=None, config_path=config_path)
    with context:
        composed_cfg = compose(config_name=config_name, overrides=override_list)

    # Allow write access to unknown fields
    OmegaConf.set_struct(composed_cfg, False)

    for key, value in deferred_updates:
        OmegaConf.update(composed_cfg, key, value, merge=True)

    normalize_config_sequences(composed_cfg)
    prune_config_targets(composed_cfg)
        
    return composed_cfg

def escape_special_characters(value: str) -> str:
    special_characters = r"\()[]{}:=,&"
    for char in special_characters:
        if char in value:
            value = f'"{value}"'
            break
    return value

def escape_all(data):
    if isinstance(data, str):
        return escape_special_characters(data)
    elif isinstance(data, (dict, DictConfig)):
        return {k: escape_all(v) for k, v in data.items()}
    elif isinstance(data, (list, ListConfig)):
        return [escape_all(item) for item in data]
    else:
        return data

def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Sums all values from the `src` tensor into `out` at the indices specified in the `index` tensor
    along the dimension `dim`. If `out` is not provided, it will be automatically created with the correct size.

    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices of elements to scatter.
            Must have the same size as `src` at dimension `dim` or be broadcastable to that size.
        dim (int): The axis along which to index. Negative values wrap around.
        out (torch.Tensor, optional): The destination tensor.

    Returns:
        torch.Tensor: The resulting tensor with the summed values scattered at the specified indices.
    """
    index = _broadcast(index, src, dim)
    if out is None:
        # Determine size of output tensor along dimension `dim`
        output_size = list(src.size())
        output_size[dim] = int(index.max()) + 1  # Size along dim is max index + 1
        out = torch.zeros(output_size, dtype=src.dtype, device=src.device)

    # Perform scatter add
    out.scatter_add_(dim, index, src)

    return out

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the mean of all values from the `src` tensor into `out` at the indices specified in the `index` tensor
    along the dimension `dim`. If `out` is not provided, it will be automatically created to have the correct size.

    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices of elements to scatter. Must have the same size as `src` at dimension `dim`.
        dim (int): The axis along which to index. Negative values wrap around.
        out (torch.Tensor, optional): The destination tensor.

    Returns:
        torch.Tensor: The resulting tensor with the mean values scattered at the specified indices.
    """
    index = _broadcast(index, src, dim)

    if out is None:
        # Determine size of output tensor along dimension `dim`
        output_size = list(src.size())
        output_size[dim] = int(index.max()) + 1  # Size along dim is max index + 1
        out = torch.zeros(output_size, dtype=src.dtype, device=src.device)
        out_count = torch.zeros_like(out)
    else:
        out_count = torch.zeros_like(out)

    # Compute sum of values
    out.scatter_add_(dim, index, src)

    # Count number of occurrences at each index
    ones = torch.ones_like(src, dtype=src.dtype)
    out_count.scatter_add_(dim, index, ones)

    # Avoid division by zero
    zero_mask = out_count == 0
    out_count[zero_mask] = 1

    # Compute mean
    out = out / out_count

    return out

def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the maximum of all values from the `src` tensor into `out` at the indices specified in the `index` tensor
    along the dimension `dim`.

    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices of elements to scatter.
            Must have the same size as `src` at dimension `dim` or be broadcastable to that size.
        dim (int): The axis along which to index. Negative values wrap around.
        out (torch.Tensor, optional): The destination tensor. If None, a new tensor is created.

    Returns:
        torch.Tensor: The resulting tensor with the maximum values scattered at the specified indices.
    """
    index = _broadcast(index, src, dim)

    # Determine size of output tensor along dimension `dim`
    output_size = list(src.size())
    output_size[dim] = int(index.max()) + 1  # Size along dim is max index + 1

    # Initialize out tensor with minimum possible values
    if out is None:
        out = torch.full(output_size, torch.finfo(src.dtype).min, dtype=src.dtype, device=src.device)
    else:
        out.fill_(torch.finfo(src.dtype).min)

    # Compute maximum values
    out.scatter_(dim, index, torch.max(out.gather(dim, index), src))

    return out

def scatter_min(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the minimum of all values from the `src` tensor into `out` at the indices specified in the `index` tensor
    along the dimension `dim`.

    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices of elements to scatter.
            Must have the same size as `src` at dimension `dim` or be broadcastable to that size.
        dim (int): The axis along which to index. Negative values wrap around.
        out (torch.Tensor, optional): The destination tensor. If None, a new tensor is created.

    Returns:
        torch.Tensor: The resulting tensor with the minimum values scattered at the specified indices.
    """
    index = _broadcast(index, src, dim)

    # Determine size of output tensor along dimension `dim`
    output_size = list(src.size())
    output_size[dim] = int(index.max()) + 1

    # Initialize out tensor with maximum possible values
    if out is None:
        out = torch.full(output_size, torch.finfo(src.dtype).max, dtype=src.dtype, device=src.device)
    else:
        out.fill_(torch.finfo(src.dtype).max)

    # Compute minimum values
    out.scatter_(dim, index, torch.min(out.gather(dim, index), src))

    return out

def scatter_reduce(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: torch.Tensor = None,
                   reduce: Literal["sum", "mean", "max", "min"] = 'sum',
                   include_self: bool = False) -> torch.Tensor:
    """
    Reduces all values from the `src` tensor into `out` at the indices specified in the `index` tensor
    along the dimension `dim` using the specified reduction ('sum', 'mean', 'max', 'min').

    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices of elements to scatter.
            Must have the same size as `src` at dimension `dim` or be broadcastable to that size.
        dim (int): The axis along which to index.
        out (torch.Tensor, optional): The destination tensor. If None, a new tensor is created.
        reduce (str): The reduction operation to apply ('sum', 'mean', 'max', 'min').
        include_self (bool): Whether to include existing values in `out` during reduction.

    Returns:
        torch.Tensor: The resulting tensor with the reduced values scattered at the specified indices.
    """
    # Validate reduce operation
    if reduce not in ['sum', 'mean', 'max', 'min']:
        raise ValueError(f"Invalid reduce operation '{reduce}'. Supported operations: 'sum', 'mean', 'max', 'min'.")

    # Ensure index has the same number of dimensions as src
    index = _broadcast(index, src, dim)

    # Determine size of output tensor along dimension `dim`
    output_size = list(src.size())
    output_size[dim] = int(index.max()) + 1

    # Initialize out tensor
    if out is None:
        if reduce in ['sum', 'mean']:
            out = torch.zeros(output_size, dtype=src.dtype, device=src.device)
        elif reduce == 'max':
            out = torch.full(output_size, torch.finfo(src.dtype).min, dtype=src.dtype, device=src.device)
        elif reduce == 'min':
            out = torch.full(output_size, torch.finfo(src.dtype).max, dtype=src.dtype, device=src.device)
    else:
        if not include_self:
            if reduce in ['sum', 'mean']:
                out.zero_()
            elif reduce == 'max':
                out.fill_(torch.finfo(src.dtype).min)
            elif reduce == 'min':
                out.fill_(torch.finfo(src.dtype).max)

    if reduce == 'sum':
        out.scatter_add_(dim, index, src)
    elif reduce == 'mean':
        out.scatter_add_(dim, index, src)
        # Count occurrences for mean calculation
        count = torch.zeros_like(out)
        ones = torch.ones_like(src, dtype=src.dtype)
        count.scatter_add_(dim, index, ones)
        zero_mask = count == 0
        count[zero_mask] = 1
        out = out / count
    elif reduce == 'max':
        out.scatter_(dim, index, torch.max(out.gather(dim, index), src))
    elif reduce == 'min':
        out.scatter_(dim, index, torch.min(out.gather(dim, index), src))

    return out

# Function to check if cell is upper-triangular
def is_upper_triangular(cell):
    return np.allclose(np.tril(cell, -1), 0)

# transform lower-triangular cell to upper-triangular cell
def upper_triangular_cell(atoms, verbose=False):
    if not is_upper_triangular(atoms.get_cell()):
        a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
        angles = np.radians((alpha, beta, gamma))
        sin_a, sin_b, sin_g = np.sin(angles)
        cos_a, cos_b, cos_g = np.cos(angles)
        cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
        cos_p = np.clip(cos_p, -1, 1)
        sin_p = np.sqrt(1 - cos_p**2)
        new_basis = [
            (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
            (0, b * sin_a, b * cos_a),
            (0, 0, c),
        ]
        atoms.set_cell(new_basis, scale_atoms=True)
        if verbose:
            print("Transformed to upper triangular unit cell.", flush=True)
    return atoms

def get_representation_config(model):
    """Extract configurations of a model, which can then be used to instantiate a new one."""
    rep = model.representation
    if model.representation.__class__.__name__ == 'MACE':
        species = list(rep.embeddings.onehot_embedding.type_mapper.symbol_to_type.keys())
        correlation = None
        sc = rep.products[0].symmetric_contractions
        # cueq wrapper may expose only weight tensor
        if hasattr(sc, "contractions") and len(sc.contractions) > 0 and hasattr(sc.contractions[0], "weights"):
            correlation = len(sc.contractions[0].weights) + 1
        elif hasattr(sc, "contraction_degree"):
            correlation = sc.contraction_degree
        elif hasattr(sc, "weight"):
            # weight shape: (mul, sum_{k<=lmax}(3 blocks)) -> infer kmax assuming 3 terms per k
            total = sc.weight.shape[1]
            # correlation counts per k: 3 entries per k (weights_max, weights.0, weights.1)
            # solve for kmax where 3 * (kmax+1) == total blocks
            if total % 3 == 0:
                kmax = total // 3 - 1
                correlation = kmax + 1
        elif hasattr(sc, "sc"):
            if hasattr(sc.sc, "contraction_degree"):
                correlation = sc.sc.contraction_degree
        if correlation is None:
            raise AttributeError("Unable to infer correlation from symmetric_contractions.")

        def _gate_from_activation(activation):
            if activation is None:
                return None
            acts = getattr(activation, "acts", None)
            if acts:
                activation = acts[0]
            return getattr(activation, "f", activation)

        def _gate_from_module(module):
            if module is None:
                return None
            if isinstance(module, (list, tuple, torch.nn.ModuleList, torch.nn.Sequential)):
                for item in module:
                    gate = _gate_from_module(item)
                    if gate is not None:
                        return gate
                return None
            domain_modules = getattr(module, "domain_modules", None)
            if domain_modules:
                gate = _gate_from_module(list(domain_modules.values()))
                if gate is not None:
                    return gate
            for attr in ("readout_mlp", "readouts"):
                gate = _gate_from_module(getattr(module, attr, None))
                if gate is not None:
                    return gate
            for attr in ("activation", "non_linearity"):
                gate = _gate_from_activation(getattr(module, attr, None))
                if gate is not None:
                    return gate
            return None

        gate = _gate_from_module(getattr(rep, "readout", None))
        if gate is None:
            gate = _gate_from_module(getattr(rep, "readout_mlp", None))
        if gate is None:
            gate = torch.nn.functional.silu
        rep_config = {
            "cutoff": rep.cutoff,
            "num_interactions": len(rep.interactions),
            "correlation": correlation,
            "interaction_cls": rep.interactions[-1].__class__,
            "interaction_cls_first": rep.interactions[0].__class__,
            "radial_MLP": rep.interactions[0].conv_tp_weights.hs[1:-1],
            "species": species,
            "num_elements": len(species),
            "hidden_irreps": rep.hidden_irreps,
            "edge_sh_irreps": rep.edge_sh_irreps,
            "node_irreps": rep.node_irreps,
            "MLP_irreps": rep.MLP_irreps,
            "avg_num_neighbors": float(rep.interactions[0].avg_num_neighbors),
            "num_basis": rep.embeddings.radial_basis.basis.num_basis,
            "power": rep.embeddings.radial_basis.cutoff_fn.p,
            "distance_transform": (
                rep.embeddings.radial_basis.distance_transform.__class__.__name__
                if getattr(rep.embeddings.radial_basis, "distance_transform", None) is not None
                else "None"
            ),
            "gate": gate,
        }
        readout = getattr(rep, "readout", None)
        domain_modules = getattr(readout, "domain_modules", None)
        if domain_modules:
            from functools import partial
            from curator.layer import MultiDomainMACEAtomwiseNN

            domains = getattr(readout, "domains", None) or list(domain_modules.keys())
            heads_by_domain = {
                str(dom): list(module.heads)
                for dom, module in domain_modules.items()
                if hasattr(module, "heads")
            }
            readout_kwargs = {"domains": [str(d) for d in domains]}
            if heads_by_domain:
                readout_kwargs["heads_by_domain"] = heads_by_domain
            rep_config["readout"] = partial(MultiDomainMACEAtomwiseNN, **readout_kwargs)
    elif model.representation.__class__.__name__ == 'Nequip':
        mapper = getattr(rep.embeddings.onehot_embedding, "type_mapper", None)
        if mapper is not None:
            species = list(mapper.symbol_to_type.keys())
        else:
            species = list(rep.species or [])
        num_elements = getattr(rep.embeddings.onehot_embedding, "num_elements", len(species))

        rep_config = {
            "cutoff": rep.cutoff,
            "num_interactions": len(rep.interactions),
            "species": species,
            "num_elements": num_elements,
            "num_features": rep.num_features,
            "type_embed_num_features": rep.type_embed_num_features,
            "hidden_irreps": rep.hidden_irreps,
            "edge_sh_irreps": rep.edge_sh_irreps,
            "node_irreps": rep.node_irreps,
            "num_basis": rep.embeddings.radial_basis.basis.num_basis,
            "power": rep.embeddings.radial_basis.cutoff_fn.p,
            "resnet": rep.interactions[0].resnet,
            "nonlinearity_type": rep.nonlinearity_type,
            "nonlinearity_scalars": rep.nonlinearity_scalars,
            "nonlinearity_gates": rep.nonlinearity_gates,
            "radial_mlp_depth": rep.radial_mlp_depth,
            "radial_mlp_width": rep.radial_mlp_width,
            "readout_mlp_hidden_layers_depth": rep.readout_mlp_hidden_layers_depth,
            "readout_mlp_hidden_layers_width": rep.readout_mlp_hidden_layers_width,
            "readout_mlp_nonlinearity": rep.readout_mlp_nonlinearity,
            "convolution_kwargs": rep.convolution_kwargs,
        }
        readout = getattr(rep, "readout", None)
        domain_modules = getattr(readout, "domain_modules", None)
        if domain_modules:
            from curator.layer import MultiDomainAtomwiseNN

            domains = getattr(readout, "domains", None) or list(domain_modules.keys())
            heads_by_domain = {
                str(dom): list(module.heads)
                for dom, module in domain_modules.items()
                if hasattr(module, "heads")
            }
            readout_kwargs = {"domains": [str(d) for d in domains]}
            if heads_by_domain:
                readout_kwargs["heads_by_domain"] = heads_by_domain
            rep_config["readout"] = partial(MultiDomainAtomwiseNN, **readout_kwargs)
    elif model.representation.__class__.__name__ == 'Painn':
        rep_config = {
            "cutoff": rep.cutoff,
            "num_interactions": rep.num_interactions,
            "num_features": rep.num_features,
            "num_basis": rep.num_basis,
        }
    return rep_config

def get_kmax_pairs(
    max_L: int, correlation: int, num_layers: int
) -> List[Tuple[int, int]]:
    """Determine kmax pairs based on max_L and correlation"""
    if correlation == 2:
        raise NotImplementedError("Correlation 2 not supported yet")
    if correlation == 3:
        kmax_pairs = [[i, max_L] for i in range(num_layers - 1)]
        kmax_pairs = kmax_pairs + [[num_layers - 1, 0]]
        return kmax_pairs
    raise NotImplementedError(f"Correlation {correlation} not supported")


def transfer_symmetric_contractions(
    source_dict: Dict[str, torch.Tensor],
    target_dict: Dict[str, torch.Tensor],
    max_L: int,
    correlation: int,
    num_layers: int,
):
    """Transfer symmetric contraction weights"""
    kmax_pairs = get_kmax_pairs(max_L, correlation, num_layers)

    for i, kmax in kmax_pairs:
        wm = torch.concatenate(
            [
                source_dict[
                    f"products.{i}.symmetric_contractions.contractions.{k}.weights{j}"
                ]
                for k in range(kmax + 1)
                for j in ["_max", ".0", ".1"]
            ],
            dim=1,
        )
        target_dict[f"products.{i}.symmetric_contractions.sc.weight"] = wm

def get_transfer_keys(num_layers: int) -> List[str]:
    """Get list of keys that need to be transferred"""
    return [
        "embeddings.chemical_embedding.linear.weight",
        *[f"readout.readouts.{j}.linear.weight" for j in range(num_layers - 1)],
        *[f"readout.readout_mlp.{i}.linear.weight" for i in range(2)],
        *[f"readout.readouts.{num_layers - 1}.{i}.linear.weight" for i in range(2)],
    ] + [
        s
        for j in range(num_layers)
        for s in [
            f"interactions.{j}.linear_up.weight",
            *[f"interactions.{j}.conv_tp_weights.layer{i}.weight" for i in range(4)],
            f"interactions.{j}.linear.weight",
            f"interactions.{j}.skip_tp.weight",
            f"products.{j}.linear.weight",
        ]
    ]

def _squeeze_if_compatible(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Helper to squeeze a leading singleton dim if that matches the target shape."""
    if src.shape != target_shape and src.dim() == len(target_shape) + 1 and src.shape[0] == 1:
        return src.squeeze(0)
    return src

def _expand_if_compatible(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Helper to expand a leading singleton dim if that matches the target shape."""
    if src.shape != target_shape and src.dim() + 1 == len(target_shape) and target_shape[0] == 1:
        return src.unsqueeze(0)
    return src


def transfer_symmetric_contractions_back(
    source_dict: Dict[str, torch.Tensor],
    target_dict: Dict[str, torch.Tensor],
    max_L: int,
    correlation: int,
    num_layers: int,
):
    """Transfer symmetric contraction weights from cueq back to e3nn layout."""
    kmax_pairs = get_kmax_pairs(max_L, correlation, num_layers)

    for i, kmax in kmax_pairs:
        key = f"products.{i}.symmetric_contractions.sc.weight"
        if key not in source_dict:
            continue
        weight = source_dict[key]
        offset = 0
        for k in range(kmax + 1):
            for suffix in ["_max", ".0", ".1"]:
                tgt_key = f"products.{i}.symmetric_contractions.contractions.{k}.weights{suffix}"
                if tgt_key not in target_dict:
                    continue
                width = target_dict[tgt_key].shape[1]
                target_dict[tgt_key] = weight[:, offset : offset + width]
                offset += width

def load_e3nn_weights(source_model, target_model):
    """Load weights from an e3nn model to cuequivariance model"""
    source_dict = source_model.representation.state_dict()
    target_dict = target_model.representation.state_dict()
    target_shapes = {k: v.shape for k, v in target_dict.items()}

    # Transfer main weights
    num_layers = len(source_model.representation.interactions)
    transfer_keys = get_transfer_keys(num_layers)
    for key in transfer_keys:
        target_shape = target_shapes.get(key)
        if target_shape is None:
            continue
        if key in source_dict:
            target_dict[key] = _expand_if_compatible(source_dict[key], target_shape)
        else:
            logging.warning(f"Key {key} not found in source model")

    # transfer symmetric contractions
    use_cueq = any(k.endswith("symmetric_contractions.sc.weight") for k in target_shapes)
    if use_cueq:
        lmax = source_model.representation.lmax
        try:
            correlation = (
                len(source_model.representation.products[0].symmetric_contractions.contractions[0].weights) + 1
            )
        except AttributeError:
            correlation = source_model.representation.products[0].symmetric_contractions.sc.contraction_degree
        transfer_symmetric_contractions(source_dict, target_dict, lmax, correlation, num_layers)

    transferred_keys = set(transfer_keys)
    remaining_keys = set(source_dict.keys()) & set(target_dict.keys()) - transferred_keys
    if use_cueq:
        remaining_keys = {k for k in remaining_keys if "symmetric_contraction" not in k}
    for key in remaining_keys:
        src_val = _expand_if_compatible(source_dict[key], target_shapes[key])
        if src_val.shape == target_shapes[key]:
            logging.debug(f"Transferring additional key: {key}")
            target_dict[key] = src_val
        else:
            logging.warning(
                f"Shape mismatch for key {key}: "
                f"source {source_dict[key].shape} vs target {target_shapes[key]}"
            )

    target_model.representation.load_state_dict(target_dict)

def load_cueq_weights(source_model, target_model):
    """Load weights from a cueq model to an e3nn model."""
    source_dict = source_model.representation.state_dict()
    target_dict = target_model.representation.state_dict()
    target_shapes = {k: v.shape for k, v in target_dict.items()}

    num_layers = len(target_model.representation.interactions)
    transfer_keys = get_transfer_keys(num_layers)
    for key in transfer_keys:
        target_shape = target_shapes.get(key)
        if target_shape is None:
            continue
        if key in source_dict:
            target_dict[key] = _squeeze_if_compatible(source_dict[key], target_shape)
        else:
            logging.warning(f"Key {key} not found in source cueq model")

    for key in source_dict.keys():
        if "weight" in key and any(x in key for x in ["linear", "skip_tp"]):
            target_shape = target_shapes.get(key)
            if target_shape is not None:
                target_dict[key] = _squeeze_if_compatible(source_dict[key], target_shape)

    lmax = getattr(source_model.representation, "lmax", None)
    try:
        correlation = (
            len(source_model.representation.products[0].symmetric_contractions.contractions[0].weights) + 1
        )
    except Exception:
        correlation = source_model.representation.products[0].symmetric_contractions.sc.contraction_degree
    if lmax is not None:
        transfer_symmetric_contractions_back(source_dict, target_dict, lmax, correlation, num_layers)

    transferred_keys = set(transfer_keys)
    remaining_keys = set(source_dict.keys()) & set(target_dict.keys()) - transferred_keys
    remaining_keys = {k for k in remaining_keys if "symmetric_contraction" not in k}
    for key in remaining_keys:
        src_val = _squeeze_if_compatible(source_dict[key], target_shapes[key])
        if src_val.shape == target_shapes[key]:
            target_dict[key] = src_val

    target_model.representation.load_state_dict(target_dict)


def convert_e3nn_to_cueq(model):
    dtype = next(model.parameters()).dtype
    prev_dtype = torch.get_default_dtype()
    if dtype != prev_dtype:
        torch.set_default_dtype(dtype)
    try:
        rep_config = get_representation_config(model)
        rep_config["use_cueq"] = True
        cueq_rep = model.representation.__class__(**rep_config)

        cueq_model = model.__class__(
            input_modules=list(model.input_modules),
            output_modules=list(model.output_modules),
            representation=cueq_rep,
            model_outputs=model.model_outputs,
        )
    finally:
        if dtype != prev_dtype:
            torch.set_default_dtype(prev_dtype)

    load_e3nn_weights(model, cueq_model)

    return cueq_model


def convert_cueq_to_e3nn(model):
    dtype = next(model.parameters()).dtype
    prev_dtype = torch.get_default_dtype()
    if dtype != prev_dtype:
        torch.set_default_dtype(dtype)
    try:
        rep_config = get_representation_config(model)
        rep_config["use_cueq"] = False
        e3nn_rep = model.representation.__class__(**rep_config)

        e3nn_model = model.__class__(
            input_modules=list(model.input_modules),
            output_modules=list(model.output_modules),
            representation=e3nn_rep,
            model_outputs=model.model_outputs,
        )
    finally:
        if dtype != prev_dtype:
            torch.set_default_dtype(prev_dtype)

    load_cueq_weights(model, e3nn_model)

    return e3nn_model

def update_model(model):
    import warnings
    from functools import partial
    from curator.layer import MultiDomainAtomwiseNN, MultiDomainMACEAtomwiseNN
    from curator.layer._rescale import GlobalRescaleShift, MultiDomainRescaleShift
    from curator.data.properties import HeadConfig, HEAD_PRESETS

    def _get_readout_module(rep):
        ro = getattr(rep, "readout", None)
        if ro is None:
            return None
        if hasattr(ro, "domain_modules") and len(ro.domain_modules) > 0:
            return next(iter(ro.domain_modules.values()))
        return ro

    rep_config = get_representation_config(model)
    rep_name = model.representation.__class__.__name__
    if rep_name in {"Painn", "Nequip"}:
        rep_config["readout"] = partial(MultiDomainAtomwiseNN, domains=["0"])
    elif rep_name == "MACE":
        rep_config["readout"] = partial(MultiDomainMACEAtomwiseNN, domains=["0"])
    new_rep = model.representation.__class__(**rep_config)

    old_state_dict = model.representation.state_dict()

    # replace readout weight name
    try:
        if new_rep.__class__.__name__ == 'MACE':
            target_readout = _get_readout_module(new_rep)
            if target_readout is None:
                raise AttributeError("Missing readout module for MACE.")
            for i in range(len(model.representation.readout_mlp)):
                # replace normal layers
                if i != len(model.representation.readout_mlp) - 1:
                    for name in ['weight', 'bias', 'output_mask']:
                        old_state_dict[f'readout.readouts.{i}.linear.{name}'] = old_state_dict.pop(f'readout_mlp.{i}.{name}')
                    warnings.warn("Rename weights in deprecated readouts.")
                else:
                # replace readout_mlp layer
                    for j in range(len(target_readout.readout_mlp)):
                        for name in ['weight', 'bias', 'output_mask']:
                            old_state_dict[f'readout.readouts.{i}.{j}.linear.{name}'] = old_state_dict.pop(f'readout_mlp.{i}.linear_{j+1}.{name}')
                            old_state_dict[f'readout.readout_mlp.{j}.linear.{name}'] = old_state_dict[f'readout.readouts.{i}.{j}.linear.{name}']
                    warnings.warn("Rename weights in deprecated readout_mlp.")
        elif new_rep.__class__.__name__ == 'Painn':
            target_readout = _get_readout_module(new_rep)
            if target_readout is None:
                raise AttributeError("Missing readout module for Painn.")
            for i in range(len(target_readout.readout_mlp)):
                for name in ['weight', 'bias']:
                    old_state_dict[f'readout.readout_mlp.{i}.linear.{name}'] = old_state_dict.pop(f'readout_mlp.{2*i}.{name}')
            warnings.warn("Rename weights in deprecated readout_mlp.")

    except KeyError:
        pass

    if hasattr(new_rep, "readout") and hasattr(new_rep.readout, "domain_modules"):
        remapped = {}
        for k, v in old_state_dict.items():
            if k.startswith("readout.domain_modules."):
                remapped[k] = v
            elif k.startswith("readout."):
                remapped[f"readout.domain_modules.0.{k[len('readout.') :]}"] = v
            else:
                remapped[k] = v
        old_state_dict = remapped

    try:
        new_rep.load_state_dict(old_state_dict)
    except:
        warnings.warn("Loading weights from old model failed!")

    output_modules = model.output_modules
    # fix output modules
    try:
        # modify modules in-place
        for i, m in enumerate(output_modules):
            if m.__class__.__name__ == 'GradientOutput':
                output_modules[i] = m.__class__(
                    grad_on_edge_diff = m.grad_on_edge_diff,
                    grad_on_positions = m.grad_on_positions,
                    compute_edge_forces = getattr(m, 'compute_edge_forces', False),
                    compute_edge_forces_only = getattr(m, 'compute_edge_forces_only', False),
                    model_outputs = m.model_outputs,
                )
                warnings.warn('Replace GradientOutput module.')
            if m.__class__.__name__ == 'GlobalRescaleShift':
                def _as_scalar(value):
                    if torch.is_tensor(value):
                        v = value.detach().clone().cpu().reshape(-1)
                        return v[0].item() if v.numel() > 0 else 0.0
                    if isinstance(value, list):
                        return value[0] if value else 0.0
                    return value

                scale_by = _as_scalar(getattr(m, "scale_by", None))
                shift_by = _as_scalar(getattr(m, "shift_by", None))
                scale_keys = list(getattr(m, "scale_keys", []))
                shift_keys = list(getattr(m, "shift_keys", []))
                output_keys = list(getattr(m, "output_keys", []))
                if not output_keys:
                    output_keys = list(dict.fromkeys(scale_keys + shift_keys)) or ["energy"]

                atomwise_shift = bool(getattr(m, "atomwise_shift", False))
                atomwise_norm = bool(getattr(m, "atomwise_normalization", False))
                per_species_shift = None
                per_species_all = getattr(m, "per_species_shifts", None)
                if isinstance(per_species_all, dict):
                    per_species_shift = per_species_all.get(properties.energy) or per_species_all.get("energy")

                heads = []
                for key in output_keys:
                    use_atomwise_shift = atomwise_shift if key in shift_keys else False
                    use_atomwise_norm = atomwise_norm if key in shift_keys else False
                    if key in HEAD_PRESETS:
                        base = HEAD_PRESETS[key]
                        heads.append(
                            HeadConfig(
                                key=base.key,
                                dim=base.dim,
                                is_atomwise=base.is_atomwise,
                                reduction=base.reduction,
                                atomwise_key=base.atomwise_key,
                                write_atomwise=base.write_atomwise,
                                scale_by=scale_by if key in scale_keys else None,
                                shift_by=shift_by if key in shift_keys else None,
                                atomwise_shift=use_atomwise_shift,
                                atomwise_normalization=use_atomwise_norm,
                                per_species_shift=(
                                    per_species_shift
                                    if key in {properties.energy, properties.atomic_energy, "energy", "atomic_energy"}
                                    else None
                                ),
                            )
                        )
                    else:
                        heads.append(
                            HeadConfig(
                                key=key,
                                dim=1,
                                is_atomwise=False,
                                reduction=None,
                                scale_by=scale_by if key in scale_keys else None,
                                shift_by=shift_by if key in shift_keys else None,
                            )
                        )

                def _is_scale_trainable(mod):
                    for sc in getattr(mod, "scales", []):
                        if isinstance(getattr(sc, "scale", None), torch.nn.Parameter):
                            return True
                    return False

                def _is_shift_trainable(mod):
                    for sh in getattr(mod, "shifts", []):
                        if isinstance(getattr(sh, "shift", None), torch.nn.Parameter):
                            return True
                    return False

                grs = GlobalRescaleShift(
                    heads=heads,
                    scale_trainable=_is_scale_trainable(m),
                    shift_trainable=_is_shift_trainable(m),
                )
                md = MultiDomainRescaleShift(heads=grs.heads)
                md.domain_modules["0"] = grs
                output_modules[i] = md
                warnings.warn('Replace GlobalRescaleShift module with MultiDomainRescaleShift.')
        # remove module
        for i, m in enumerate(output_modules):
            if m.__class__.__name__ == 'AtomwiseReduce':
                output_modules.pop(i)
                warnings.warn('Remove AtomwiseReduce module in output modules.')
    except:
        pass

    new_model = model.__class__(
        input_modules=list(model.input_modules),        # almost no update in input_modules and output_modules
        output_modules=list(output_modules),
        representation=new_rep,
        model_outputs=model.model_outputs,
    )

    return new_model

def _register_legacy_outputspec() -> None:
    try:
        import curator.layer._atomwise_nn as atomwise
    except Exception:
        return
    if hasattr(atomwise, "OutputSpec"):
        return

    class OutputSpec:
        def __init__(
            self,
            key: str,
            dim: int = 1,
            is_atomwise: bool = False,
            reduction: Optional[Literal["sum", "mean", "none"]] = "sum",
            atomwise_key: Optional[str] = None,
            write_atomwise: bool = False,
        ) -> None:
            self.key = key
            self.dim = dim
            self.is_atomwise = is_atomwise
            self.reduction = reduction
            self.atomwise_key = atomwise_key
            self.write_atomwise = write_atomwise

    atomwise.OutputSpec = OutputSpec

def upgrade_checkpoint(
    ckpt_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Path:
    """Upgrade an older Curator checkpoint by rebuilding its stored model.

    Loads the checkpoint on CPU by default (so conversion works without GPUs),
    rebuilds the model via ``update_model``, and writes a new checkpoint.
    """
    import curator.model.compat  # registers legacy class aliases for torch.load
    from collections import OrderedDict

    ckpt_path = Path(ckpt_path)
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if output_path is None:
        output_path = ckpt_path.with_name(f"{ckpt_path.stem}_converted{ckpt_path.suffix}")
    output_path = Path(output_path)

    _register_legacy_outputspec()
    try:
        obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(ckpt_path, map_location=device)

    if isinstance(obj, torch.nn.Module):
        upgraded_model = update_model(obj)
        torch.save(upgraded_model, output_path)
        return output_path

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    if "model" not in obj:
        raise KeyError("Checkpoint is missing 'model' entry to upgrade.")

    upgraded_model = update_model(obj["model"])
    obj["model"] = upgraded_model
    if "state_dict" in obj:
        state_dict = upgraded_model.state_dict()
        new_state_dict = OrderedDict()
        for k in state_dict.keys():
            new_state_dict['model.' + k] = state_dict[k]
        obj["state_dict"] = new_state_dict
    torch.save(obj, output_path)
    return output_path
