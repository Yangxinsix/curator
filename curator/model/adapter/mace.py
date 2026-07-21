from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse
from urllib.request import urlretrieve

import torch
from ase import Atoms
from ase.data import chemical_symbols
from torch import nn

from curator.data import properties
from curator.model.conversion import (
    _load_official_mace_model,
    load_official_mace_as_curator,
)
from curator.model.utils import (
    bind_target_layer_aliases,
    resolve_target_layer,
    split_batch_structures,
)

from .ase import ASECalculatorAdapter
from .utils import ExternalModelSpec, build_representation, parse_bool, register_adapter_loader


def _parse_head(value: Optional[str]) -> Optional[Union[str, int]]:
    if value is None:
        return None
    token = str(value).strip()
    if token == "":
        return None
    try:
        return int(token)
    except ValueError:
        return token


def _resolve_head_name(model: nn.Module, head: Optional[Union[str, int]]) -> Optional[str]:
    heads = list(getattr(model, "heads", []) or [])
    if not heads:
        return None
    if head is None:
        return str(heads[0])
    if isinstance(head, int):
        if head < 0 or head >= len(heads):
            raise ValueError(f"Head index {head} out of range for heads={heads}.")
        return str(heads[head])
    head_name = str(head)
    if head_name not in heads:
        raise ValueError(f"Head {head_name!r} not found in heads={heads}.")
    return head_name


def _resolve_model_ref(model_ref: str) -> str:
    parsed = urlparse(str(model_ref))
    if parsed.scheme not in {"http", "https"}:
        return model_ref
    suffix = Path(parsed.path).suffix or ".model"
    digest = hashlib.sha256(str(model_ref).encode("utf-8")).hexdigest()[:16]
    target = Path.home() / ".cache" / "curator" / "mace" / f"{digest}{suffix}"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(str(model_ref), target)
    return str(target)


class MACEAdapter(nn.Module):
    """Run an official MACE model on CURATOR batches for feature-based selection.

    The adapter intentionally keeps the official MACE runtime in charge of graph
    construction and forward execution. It returns the input CURATOR batch so
    active-learning feature hooks can collect intermediate activations from the
    official model without converting weights into CURATOR modules.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str = "readouts",
        head: Optional[Union[str, int]] = None,
        compute_force: bool = True,
        training: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self.head = _resolve_head_name(model, head)
        self.compute_force = bool(compute_force)
        self.training_flag = bool(training)
        self.model_outputs = [properties.energy]
        if self.compute_force:
            self.model_outputs.append(properties.forces)
        self.cutoff = float(getattr(model, "r_max"))
        self.representation = build_representation(self.cutoff)
        self.atomic_numbers = (
            torch.as_tensor(getattr(model, "atomic_numbers"))
            .detach()
            .cpu()
            .to(torch.long)
        )
        target_module = resolve_target_layer(model, target_layer, ("readouts", "readout"))
        bind_target_layer_aliases(self, target_layer, target_module)

    @property
    def heads(self) -> list[str]:
        return [str(head) for head in list(getattr(self.model, "heads", []) or [])]

    def _structure_to_atoms(self, struct) -> Atoms:
        kwargs = {
            "numbers": struct.numbers.detach().cpu().numpy(),
            "positions": struct.positions.detach().cpu().numpy(),
            "pbc": tuple(bool(v) for v in struct.pbc.detach().cpu().tolist()),
        }
        if struct.cell is not None:
            kwargs["cell"] = struct.cell.detach().cpu().numpy()
        return Atoms(**kwargs)

    def _structure_head(self, batch: properties.Type, index: int) -> Optional[str]:
        heads = self.heads
        if not heads:
            return None
        domain = batch.get(properties.domain)
        if domain is not None:
            domain_tensor = torch.as_tensor(domain).view(-1)
            if domain_tensor.numel() > index:
                domain_idx = int(domain_tensor[index].item())
                if 0 <= domain_idx < len(heads):
                    return heads[domain_idx]
        return self.head

    def _build_mace_batch(self, data: properties.Type):
        from mace.data.atomic_data import AtomicData
        from mace.data.utils import Configuration
        from mace.tools import torch_geometric
        from mace.tools.utils import get_atomic_number_table_from_zs

        z_table = get_atomic_number_table_from_zs(self.atomic_numbers.numpy())
        mace_heads = self.heads or None
        frames = []
        for idx, struct in enumerate(split_batch_structures(data)):
            atoms = self._structure_to_atoms(struct)
            head = self._structure_head(data, idx)
            config = Configuration(
                atomic_numbers=atoms.numbers,
                positions=atoms.positions,
                properties={},
                property_weights={},
                cell=atoms.cell.array,
                pbc=atoms.pbc,
                head=head,
            )
            frames.append(
                AtomicData.from_config(
                    config,
                    z_table=z_table,
                    cutoff=self.cutoff,
                    heads=mace_heads,
                )
            )
        return torch_geometric.Batch.from_data_list(frames).to_dict()

    def _device_dtype(self) -> tuple[torch.device, torch.dtype | None]:
        device = torch.device("cpu")
        dtype = None
        for parameter in self.model.parameters():
            device = parameter.device
            if parameter.is_floating_point():
                dtype = parameter.dtype
                break
        return device, dtype

    def forward(self, data: properties.Type) -> properties.Type:
        batch = self._build_mace_batch(data)
        device, dtype = self._device_dtype()
        for key, value in list(batch.items()):
            if not torch.is_tensor(value):
                continue
            value = value.to(device)
            if dtype is not None and torch.is_floating_point(value):
                value = value.to(dtype)
            batch[key] = value
        outputs = self.model(
            batch,
            compute_force=self.compute_force,
            training=self.training_flag,
        )
        if isinstance(outputs, dict):
            if "energy" in outputs:
                data[properties.energy] = outputs["energy"]
            if "forces" in outputs:
                data[properties.forces] = outputs["forces"]
        return data

    def __repr__(self) -> str:
        symbols = [chemical_symbols[int(z)] for z in self.atomic_numbers.tolist()]
        head = self.head if self.head is not None else "none"
        return (
            f"{self.__class__.__name__}(cutoff={self.cutoff}, "
            f"target_layer={self.target_layer!r}, head={head!r}, species={symbols})"
        )


def _load_mace(spec: ExternalModelSpec, device: Optional[torch.device]) -> nn.Module:
    omol_model_name = _mace_omol_model_name(spec.resource)
    if omol_model_name is not None and parse_bool(spec.params.get("as_calculator"), True):
        return _load_mace_omol_calculator(
            omol_model_name,
            device=device or torch.device("cpu"),
        )
    head = _parse_head(spec.params.get("head"))
    model_ref = _resolve_mace_resource(_resolve_model_ref(spec.resource))
    runtime = str(spec.params.get("runtime", spec.params.get("mode", ""))).strip().lower()
    use_adapter = parse_bool(
        spec.params.get("adapter"),
        False,
    ) or runtime in {"official", "mace", "adapter"}
    if use_adapter:
        model = _load_official_mace_model(
            model_ref,
            device=device or torch.device("cpu"),
        )
        adapter = MACEAdapter(
            model=model,
            target_layer=spec.params.get("target_layer", "readouts"),
            head=head,
            compute_force=parse_bool(spec.params.get("compute_force"), True),
            training=parse_bool(spec.params.get("training"), False),
        )
        if device is not None:
            adapter.to(device)
        adapter.eval()
        return adapter
    return load_official_mace_as_curator(
        model_ref,
        head=head,
        device=device or torch.device("cpu"),
    )


def _mace_omol_model_name(resource: str) -> Optional[str]:
    resource_key = str(resource).strip()
    omol_aliases = {
        "omol-extra-large": "extra_large",
        "omol-extra_large": "extra_large",
        "MACE-OMOL-0-extra-large": "extra_large",
    }
    return omol_aliases.get(resource_key)


def _resolve_mace_resource(resource: str):
    path = Path(resource).expanduser()
    if path.exists():
        return path

    resource_key = str(resource).strip()
    off_aliases = {
        "off23-small": "small",
        "off-small": "small",
        "small-off": "small",
        "MACE-OFF23-small": "small",
        "off23-medium": "medium",
        "off-medium": "medium",
        "medium-off": "medium",
        "MACE-OFF23-medium": "medium",
        "off23-large": "large",
        "off-large": "large",
        "large-off": "large",
        "MACE-OFF23-large": "large",
    }
    if resource_key in off_aliases:
        return _resolve_mace_off_resource(off_aliases[resource_key])

    omol_model_name = _mace_omol_model_name(resource_key)
    if omol_model_name is not None:
        return _resolve_mace_omol_resource(omol_model_name)

    try:
        from mace.calculators import mace_mp
    except Exception as exc:
        raise ModuleNotFoundError(
            "MACE support requires a local checkpoint path, or the mace package "
            "for official foundation model names."
        ) from exc

    known_names = set(mace_mp.__globals__.get("mace_mp_names") or ())
    model_name = None if resource in {"", "default", "None", "none"} else resource
    if model_name not in known_names:
        raise FileNotFoundError(
            f"MACE resource {resource!r} is neither a local checkpoint nor a known "
            f"MACE foundation model name. Known names: {sorted(name for name in known_names if name is not None)}"
        )
    downloader = mace_mp.__globals__.get("download_mace_mp_checkpoint")
    if not callable(downloader):
        raise RuntimeError("The installed mace package does not expose download_mace_mp_checkpoint().")
    return downloader(model_name)


def _resolve_mace_off_resource(model_name: str):
    try:
        from mace.calculators import mace_off
    except Exception as exc:
        raise ModuleNotFoundError(
            "MACE-OFF support requires the mace package."
        ) from exc

    raw_model = mace_off(model=model_name, device="cpu", return_raw_model=True)
    model_path = getattr(raw_model, "_mace_model_path", None)
    if model_path is not None:
        return model_path

    # The MACE-OFF loader does not expose its cached path, so reproduce its
    # deterministic cache naming from the loader source.
    get_cache_dir = mace_off.__globals__.get("get_cache_dir")
    if not callable(get_cache_dir):
        raise RuntimeError("The installed mace package does not expose get_cache_dir().")
    filename = {
        "small": "MACE-OFF23_small.model",
        "medium": "MACE-OFF23_medium.model",
        "large": "MACE-OFF23_large.model",
    }[model_name]
    return Path(get_cache_dir()) / filename


def _resolve_mace_omol_resource(model_name: str):
    try:
        from mace.calculators import mace_omol
    except Exception as exc:
        raise ModuleNotFoundError(
            "MACE-OMOL support requires the mace package."
        ) from exc

    raw_model = mace_omol(model=model_name, device="cpu", return_raw_model=True)
    model_path = getattr(raw_model, "_mace_model_path", None)
    if model_path is not None:
        return model_path
    get_cache_dir = mace_omol.__globals__.get("get_cache_dir")
    if not callable(get_cache_dir):
        raise RuntimeError("The installed mace package does not expose get_cache_dir().")
    return Path(get_cache_dir()) / "MACE-omol-0-extra-large-1024.model"


def _load_mace_omol_calculator(model_name: str, *, device: torch.device) -> nn.Module:
    try:
        from mace.calculators import mace_omol
    except Exception as exc:
        raise ModuleNotFoundError(
            "MACE-OMOL support requires the mace package."
        ) from exc

    calc = mace_omol(model=model_name, device=str(device))
    raw_model = mace_omol(model=model_name, device="cpu", return_raw_model=True)
    cutoff = float(getattr(raw_model, "r_max", 0.0) or 0.0)
    return ASECalculatorAdapter(calc, cutoff=cutoff).eval()


register_adapter_loader("mace", _load_mace)

__all__ = ["MACEAdapter"]
