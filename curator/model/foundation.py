from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen

from ase.data import atomic_numbers as ase_atomic_numbers
from ase.data import chemical_symbols as ase_chemical_symbols

from curator.data import properties
from curator.model.adapter import (
    ExternalModelSpec,
    format_external_model_spec,
    is_external_model_spec,
    load_external_model,
    parse_bool,
    parse_external_model_spec,
    registered_adapter_schemes,
)


_CAPABILITY_KEYS = (
    "energy",
    "forces",
    "stress",
    "periodic",
    "features",
    "finetune",
    "ase_calculator",
    "representation",
    "simulation_probe",
)


_BACKEND_GROUPS: Dict[str, set[str]] = {
    "allegro": {"allegro", "allegro_net"},
    "nequip": {"nequip", "nequip_hf", "nequip_net"},
}


_SCHEME_CAPABILITIES: Dict[str, Dict[str, Optional[bool]]] = {
    "mace": {
        "energy": True,
        "forces": True,
        "stress": None,
        "periodic": True,
        "features": True,
        "finetune": True,
        "ase_calculator": True,
        "representation": True,
        "simulation_probe": True,
    },
    "nequip": {
        "energy": True,
        "forces": True,
        "stress": None,
        "periodic": True,
        "features": True,
        "finetune": True,
        "ase_calculator": True,
        "representation": True,
        "simulation_probe": True,
    },
    "nequip_hf": {
        "energy": True,
        "forces": True,
        "stress": None,
        "periodic": True,
        "features": True,
        "finetune": True,
        "ase_calculator": True,
        "representation": True,
        "simulation_probe": True,
    },
    "nequip_net": {
        "energy": True,
        "forces": True,
        "stress": None,
        "periodic": True,
        "features": True,
        "finetune": True,
        "ase_calculator": True,
        "representation": True,
        "simulation_probe": True,
    },
    "matgl": {
        "energy": True,
        "forces": True,
        "stress": False,
        "periodic": True,
        "features": True,
        "finetune": True,
        "ase_calculator": True,
        "representation": True,
        "simulation_probe": None,
    },
    "esen": {
        "energy": True,
        "forces": True,
        "stress": None,
        "periodic": True,
        "features": True,
        "finetune": False,
        "ase_calculator": None,
        "representation": False,
        "simulation_probe": None,
    },
    "orb": {
        "energy": True,
        "forces": True,
        "stress": True,
        "periodic": True,
        "features": False,
        "finetune": False,
        "ase_calculator": True,
        "representation": False,
        "simulation_probe": None,
    },
    "mattersim": {
        "energy": True,
        "forces": True,
        "stress": True,
        "periodic": True,
        "features": False,
        "finetune": False,
        "ase_calculator": True,
        "representation": False,
        "simulation_probe": True,
    },
    "sevennet": {
        "energy": True,
        "forces": True,
        "stress": True,
        "periodic": True,
        "features": False,
        "finetune": False,
        "ase_calculator": True,
        "representation": False,
        "simulation_probe": True,
    },
    "eqv2": {
        "energy": True,
        "forces": True,
        "stress": True,
        "periodic": True,
        "features": False,
        "finetune": False,
        "ase_calculator": True,
        "representation": False,
        "simulation_probe": None,
    },
    "allegro": {
        "energy": True,
        "forces": True,
        "stress": None,
        "periodic": True,
        "features": True,
        "finetune": False,
        "ase_calculator": None,
        "representation": False,
        "simulation_probe": None,
    },
    "allegro_net": {
        "energy": True,
        "forces": True,
        "stress": None,
        "periodic": True,
        "features": True,
        "finetune": False,
        "ase_calculator": None,
        "representation": True,
        "simulation_probe": None,
    },
}


@dataclass
class ErrorInfo:
    type: str
    message: str
    recoverable: bool = True
    log_path: Optional[str] = None


@dataclass
class ResourceInfo:
    kind: str
    original: str
    resolved: str
    cache_hit: bool = False
    sha256: Optional[str] = None


@dataclass
class FoundationModelEntry:
    id: str
    adapter_spec: str
    backend: str
    aliases: list[str]
    source: str
    capabilities: Dict[str, Optional[bool]]
    description: Optional[str] = None
    elements: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    priority: int = 100


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _safe_name(value: str, limit: int = 80) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return (safe or "model")[:limit]


def _work_dir(model_id: str, adapter_hint: str, out: str) -> Path:
    digest = hashlib.sha256(adapter_hint.encode("utf-8")).hexdigest()[:12]
    return Path(out).expanduser() / f"{_safe_name(model_id)}-{digest}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{id(payload)}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_url(resource: str) -> bool:
    return urlparse(resource).scheme in {"http", "https"}


def _looks_like_path(resource: str) -> bool:
    if resource.startswith(("/", "./", "../", "~")):
        return True
    if "/" in resource or "\\" in resource:
        return True
    return Path(resource).suffix.lower() in {".pt", ".pth", ".ckpt", ".model", ".zip", ".tar", ".gz"}


def _download_to_cache(resource: str, cache_dir: Path, timeout_sec: int) -> tuple[Path, bool]:
    parsed = urlparse(resource)
    name = Path(parsed.path).name or "model"
    target_dir = cache_dir / hashlib.sha256(resource.encode("utf-8")).hexdigest()[:16]
    target = target_dir / name
    if target.exists():
        return target, True
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{id(resource)}.download")
    with urlopen(resource, timeout=timeout_sec) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(tmp, target)
    return target, False


def _entry_from_adapter_spec(
    model_id: str,
    adapter_spec: str,
    *,
    aliases: Optional[Sequence[str]] = None,
    source: str = "registry",
    description: Optional[str] = None,
    elements: Optional[Sequence[str]] = None,
    tags: Optional[Sequence[str]] = None,
    capabilities: Optional[Mapping[str, Any]] = None,
    priority: int = 100,
) -> Optional[FoundationModelEntry]:
    parsed = parse_external_model_spec(adapter_spec)
    if parsed is None:
        return None
    declared = _declared_capabilities(parsed)
    inferred = _infer_capabilities_from_model_name(model_id, adapter_spec, tags)
    declared.update(inferred)
    if capabilities:
        for key, value in capabilities.items():
            declared[str(key)] = None if value is None else bool(value)
    return FoundationModelEntry(
        id=str(model_id),
        adapter_spec=format_external_model_spec(parsed),
        backend=parsed.scheme,
        aliases=[str(alias) for alias in aliases or []],
        source=source,
        capabilities=declared,
        description=description,
        elements=[str(element) for element in elements] if elements is not None else None,
        tags=[str(tag) for tag in tags] if tags is not None else None,
        priority=int(priority),
    )


def _infer_capabilities_from_model_name(
    model_id: str,
    adapter_spec: str,
    tags: Optional[Sequence[str]] = None,
) -> Dict[str, Optional[bool]]:
    text = " ".join([model_id, adapter_spec, *(tags or [])]).lower()
    is_potential = any(
        token in text
        for token in (
            "pes",
            "potential",
            "force-field",
            "forcefield",
            "mace",
            "nequip",
            "allegro",
            "orb",
            "mattersim",
            "sevennet",
            "eqv2",
            "equiformer",
        )
    )
    property_tokens = ("eform", "bandgap", "band-gap", "formation", "property")
    is_property_only = any(token in text for token in property_tokens) and not is_potential
    if is_property_only:
        return {
            "forces": False,
            "stress": False,
            "ase_calculator": False,
            "simulation_probe": False,
        }
    return {}


def _looks_like_potential_entry(entry: FoundationModelEntry) -> bool:
    text = " ".join(
        [
            entry.id,
            entry.adapter_spec,
            entry.description or "",
            *(entry.tags or []),
        ]
    ).lower()
    if any(
        token in text
        for token in (
            "pes",
            "potential",
            "force-field",
            "forcefield",
            "mace",
            "nequip",
            "allegro",
            "orb",
            "mattersim",
            "sevennet",
            "eqv2",
            "equiformer",
        )
    ):
        return True
    if any(token in text for token in ("eform", "bandgap", "band-gap", "formation")):
        return False
    return entry.capabilities.get("forces") is True


def _mace_entry(
    model_name: str,
    *,
    model_id: Optional[str] = None,
    aliases: Sequence[str],
    description: str,
    tags: Sequence[str],
    priority: int,
    capabilities: Optional[Mapping[str, Any]] = None,
) -> Optional[FoundationModelEntry]:
    canonical_id = model_id or (model_name if model_name.startswith("mace-") else f"mace-{model_name}")
    full_aliases = list(aliases)
    if "-" in model_name and model_name not in full_aliases:
        full_aliases.append(model_name)
    return _entry_from_adapter_spec(
        canonical_id,
        f"mace:{model_name}",
        aliases=full_aliases,
        source="builtin",
        description=description,
        tags=("foundation", "mace", "pes", *tags),
        capabilities=capabilities,
        priority=priority,
    )


def _orb_entry(
    model_name: str,
    *,
    model_id: Optional[str] = None,
    aliases: Sequence[str] = (),
    description: str,
    tags: Sequence[str],
    priority: int,
) -> Optional[FoundationModelEntry]:
    canonical_id = model_id or (model_name if model_name.startswith("orb") else f"orb-{model_name}")
    return _entry_from_adapter_spec(
        canonical_id,
        f"orb:{model_name}",
        aliases=aliases,
        source="builtin",
        description=description,
        tags=("foundation", "orb", "pes", *tags),
        priority=priority,
    )


def _mattersim_entry(
    model_name: str,
    *,
    model_id: str,
    aliases: Sequence[str] = (),
    description: str,
    tags: Sequence[str],
    priority: int,
) -> Optional[FoundationModelEntry]:
    return _entry_from_adapter_spec(
        model_id,
        f"mattersim:{model_name}",
        aliases=aliases,
        source="builtin",
        description=description,
        tags=("foundation", "mattersim", "pes", *tags),
        priority=priority,
    )


def _sevennet_entry(
    model_name: str,
    *,
    model_id: Optional[str] = None,
    aliases: Sequence[str] = (),
    description: str,
    tags: Sequence[str],
    priority: int,
) -> Optional[FoundationModelEntry]:
    canonical_id = model_id or (model_name if model_name.startswith("sevennet") else f"sevennet-{model_name}")
    return _entry_from_adapter_spec(
        canonical_id,
        f"sevennet:{model_name}",
        aliases=aliases,
        source="builtin",
        description=description,
        tags=("foundation", "sevennet", "pes", *tags),
        priority=priority,
    )


def _eqv2_entry(
    model_name: str,
    filename: str,
    *,
    model_id: str,
    aliases: Sequence[str] = (),
    description: str,
    tags: Sequence[str],
    priority: int,
) -> Optional[FoundationModelEntry]:
    return _entry_from_adapter_spec(
        model_id,
        f"eqv2:facebook/OMAT24?filename={filename}",
        aliases=aliases,
        source="builtin",
        description=description,
        tags=("foundation", "eqv2", "equiformer", "omat24", "pes", *tags),
        priority=priority,
    )


def _nequip_net_entry(
    model_name: str,
    *,
    model_id: str,
    aliases: Sequence[str] = (),
    description: str,
    tags: Sequence[str],
    priority: int,
    version: str = "0.1",
) -> Optional[FoundationModelEntry]:
    return _entry_from_adapter_spec(
        model_id,
        f"nequip_net:mir-group/{model_name}?version={version}",
        aliases=aliases,
        source="builtin",
        description=description,
        tags=("foundation", "nequip", "nequip.net", "pes", *tags),
        priority=priority,
    )


def _allegro_net_entry(
    model_name: str,
    *,
    model_id: str,
    aliases: Sequence[str] = (),
    description: str,
    tags: Sequence[str],
    priority: int,
    version: str = "0.1",
) -> Optional[FoundationModelEntry]:
    return _entry_from_adapter_spec(
        model_id,
        f"allegro_net:mir-group/{model_name}?version={version}",
        aliases=aliases,
        source="builtin",
        description=description,
        tags=("foundation", "allegro", "nequip.net", "pes", *tags),
        priority=priority,
    )


def _builtin_model_entries() -> list[FoundationModelEntry]:
    entries = [
        _mace_entry(
            "small",
            aliases=("mace-small",),
            description="MACE-MP small foundation potential.",
            tags=("mp", "small"),
            priority=5,
        ),
        _mace_entry(
            "medium",
            aliases=("mace-medium",),
            description="MACE-MP medium foundation potential.",
            tags=("mp", "medium"),
            priority=6,
        ),
        _mace_entry(
            "large",
            aliases=("mace-large",),
            description="MACE-MP large foundation potential.",
            tags=("mp", "large"),
            priority=7,
        ),
        _mace_entry(
            "small-0b",
            aliases=("mace-small-0b",),
            description="MACE-MP 0b small foundation potential.",
            tags=("mp", "0b", "small"),
            priority=8,
        ),
        _mace_entry(
            "medium-0b",
            aliases=("mace-medium-0b",),
            description="MACE-MP 0b medium foundation potential.",
            tags=("mp", "0b", "medium"),
            priority=9,
        ),
        _mace_entry(
            "small-0b2",
            aliases=("mace-small-0b2",),
            description="MACE-MP 0b2 small foundation potential.",
            tags=("mp", "0b2", "small"),
            priority=10,
        ),
        _mace_entry(
            "medium-0b2",
            aliases=("mace-medium-0b2",),
            description="MACE-MP 0b2 medium foundation potential.",
            tags=("mp", "0b2", "medium"),
            priority=11,
        ),
        _mace_entry(
            "large-0b2",
            aliases=("mace-large-0b2",),
            description="MACE-MP 0b2 large foundation potential.",
            tags=("mp", "0b2", "large"),
            priority=12,
        ),
        _mace_entry(
            "medium-0b3",
            aliases=("mace-medium-0b3",),
            description="MACE-MP 0b3 medium foundation potential.",
            tags=("mp", "0b3", "medium"),
            priority=13,
        ),
        _mace_entry(
            "medium-mpa-0",
            aliases=("mace-medium-mpa-0",),
            description="MACE MPA medium foundation potential.",
            tags=("mpa", "medium"),
            priority=14,
        ),
        _mace_entry(
            "small-omat-0",
            aliases=("mace-small-omat-0",),
            description="MACE-OMAT small foundation potential.",
            tags=("omat", "small"),
            priority=15,
        ),
        _mace_entry(
            "medium-omat-0",
            aliases=("mace-medium-omat-0",),
            description="MACE-OMAT medium foundation potential.",
            tags=("omat", "medium"),
            priority=16,
        ),
        _mace_entry(
            "mace-matpes-pbe-0",
            aliases=("mace-matpes-pbe", "mace-pbe"),
            description="MACE MatPES PBE foundation potential.",
            tags=("matpes", "pbe"),
            priority=17,
        ),
        _mace_entry(
            "mace-matpes-r2scan-0",
            aliases=("mace-matpes-r2scan", "mace-r2scan"),
            description="MACE MatPES r2SCAN foundation potential.",
            tags=("matpes", "r2scan"),
            priority=18,
        ),
        _mace_entry(
            "off23-small",
            model_id="mace-off23-small",
            aliases=("mace-off-small", "small-off", "MACE-OFF23-small"),
            description="MACE-OFF23 small molecular foundation potential.",
            tags=("off23", "molecular", "small"),
            priority=22,
        ),
        _mace_entry(
            "off23-medium",
            model_id="mace-off23-medium",
            aliases=("mace-off", "mace-off-medium", "medium-off", "MACE-OFF23-medium"),
            description="MACE-OFF23 medium molecular foundation potential.",
            tags=("off23", "molecular", "medium"),
            priority=23,
        ),
        _mace_entry(
            "off23-large",
            model_id="mace-off23-large",
            aliases=("mace-off-large", "large-off", "MACE-OFF23-large"),
            description="MACE-OFF23 large molecular foundation potential.",
            tags=("off23", "molecular", "large"),
            priority=24,
        ),
        _mace_entry(
            "omol-extra-large",
            model_id="mace-omol-extra-large",
            aliases=("mace-omol", "mace-omol-0", "MACE-OMOL-0-extra-large"),
            description="MACE-OMOL extra-large molecular foundation potential.",
            tags=("omol", "molecular", "extra-large"),
            capabilities={"finetune": False, "representation": False, "ase_calculator": True},
            priority=25,
        ),
        _nequip_net_entry(
            "NequIP-MP-L",
            model_id="nequip-mp-l",
            aliases=("NequIP-MP-L", "mir-group/NequIP-MP-L", "nequip-mp"),
            description="NequIP-MP-L foundation potential from nequip.net trained on MPTrj.",
            tags=("mptrj", "matbench-compliant", "large"),
            priority=26,
        ),
        _nequip_net_entry(
            "NequIP-OAM-S",
            model_id="nequip-oam-s",
            aliases=("NequIP-OAM-S", "mir-group/NequIP-OAM-S"),
            description="Small NequIP-OAM foundation potential from nequip.net.",
            tags=("omat24", "mptrj", "salex", "small"),
            priority=27,
        ),
        _nequip_net_entry(
            "NequIP-OAM-M",
            model_id="nequip-oam-m",
            aliases=("NequIP-OAM-M", "mir-group/NequIP-OAM-M", "nequip-oam"),
            description="Medium NequIP-OAM foundation potential from nequip.net.",
            tags=("omat24", "mptrj", "salex", "medium"),
            priority=28,
        ),
        _nequip_net_entry(
            "NequIP-OAM-L",
            model_id="nequip-oam-l",
            aliases=("NequIP-OAM-L", "mir-group/NequIP-OAM-L"),
            description="Large NequIP-OAM foundation potential from nequip.net.",
            tags=("omat24", "mptrj", "salex", "large"),
            priority=29,
        ),
        _nequip_net_entry(
            "NequIP-OAM-XL",
            model_id="nequip-oam-xl",
            aliases=("NequIP-OAM-XL", "mir-group/NequIP-OAM-XL"),
            description="Extra-large NequIP-OAM foundation potential from nequip.net.",
            tags=("omat24", "mptrj", "salex", "extra-large"),
            priority=30,
        ),
        _allegro_net_entry(
            "Allegro-MP-L",
            model_id="allegro-mp-l",
            aliases=("Allegro-MP-L", "mir-group/Allegro-MP-L", "allegro-mp"),
            description="Allegro-MP-L foundation potential from nequip.net trained on MPTrj.",
            tags=("mptrj", "matbench-compliant", "large"),
            priority=31,
        ),
        _allegro_net_entry(
            "Allegro-OAM-L",
            model_id="allegro-oam-l",
            aliases=("Allegro-OAM-L", "mir-group/Allegro-OAM-L", "allegro-oam"),
            description="Allegro-OAM-L foundation potential from nequip.net.",
            tags=("omat24", "mptrj", "salex", "large"),
            priority=32,
        ),
        _orb_entry(
            "orb-v3-conservative-inf-omat",
            aliases=("orb", "orb-v3", "orb-v3-omat", "orb-v3-cons-inf-omat"),
            description="ORB v3 conservative OMat foundation potential.",
            tags=("omat", "v3", "conservative"),
            priority=26,
        ),
        _orb_entry(
            "orb-v3-direct-inf-omat",
            aliases=("orb-direct", "orb-v3-direct", "orb-v3-direct-omat"),
            description="ORB v3 direct-force OMat foundation potential.",
            tags=("omat", "v3", "direct"),
            priority=27,
        ),
        _orb_entry(
            "orb-v3-conservative-20-omat",
            aliases=("orb-v3-conservative-20", "orb-v3-conservative-120-omat"),
            description="ORB v3 conservative OMat potential with finite-neighbor inference settings.",
            tags=("omat", "v3", "conservative", "20-neighbor"),
            priority=28,
        ),
        _orb_entry(
            "orb-v3-direct-20-omat",
            aliases=("orb-v3-direct-20", "orb-v3-direct-120-omat"),
            description="ORB v3 direct-force OMat potential with finite-neighbor inference settings.",
            tags=("omat", "v3", "direct", "20-neighbor"),
            priority=29,
        ),
        _orb_entry(
            "orb-v3-conservative-inf-mpa",
            aliases=("orb-v3-mpa", "orb-v3-cons-inf-mpa"),
            description="ORB v3 conservative MPtrj + Alexandria foundation potential.",
            tags=("mpa", "v3", "conservative"),
            priority=30,
        ),
        _orb_entry(
            "orb-v3-direct-inf-mpa",
            aliases=("orb-v3-direct-mpa",),
            description="ORB v3 direct-force MPtrj + Alexandria foundation potential.",
            tags=("mpa", "v3", "direct"),
            priority=31,
        ),
        _orb_entry(
            "orb-v3-conservative-20-mpa",
            aliases=("orb-v3-conservative-20-mpa", "orb-v3-conservative-120-mpa"),
            description="ORB v3 conservative MPtrj + Alexandria potential with finite-neighbor inference settings.",
            tags=("mpa", "v3", "conservative", "20-neighbor"),
            priority=32,
        ),
        _orb_entry(
            "orb-v3-direct-20-mpa",
            aliases=("orb-v3-direct-20-mpa", "orb-v3-direct-120-mpa"),
            description="ORB v3 direct-force MPtrj + Alexandria potential with finite-neighbor inference settings.",
            tags=("mpa", "v3", "direct", "20-neighbor"),
            priority=33,
        ),
        _orb_entry(
            "orb-v2",
            aliases=("orb-v2-mptrj",),
            description="ORB v2 pretrained interatomic potential.",
            tags=("v2", "mptraj"),
            priority=34,
        ),
        _orb_entry(
            "orb-mptraj-only-v2",
            aliases=("orb-v2-mptraj-only",),
            description="ORB v2 model trained only on MPtrj data.",
            tags=("v2", "mptraj"),
            priority=35,
        ),
        _orb_entry(
            "orb-d3-v2",
            aliases=("orb-d3",),
            description="ORB v2 model trained with integrated D3-corrected targets.",
            tags=("v2", "d3", "mptraj"),
            priority=36,
        ),
        _orb_entry(
            "orbmol-v1-conservative",
            aliases=("orbmol-v1", "orb-mol-v1-conservative"),
            description="OrbMol v1 conservative molecular potential.",
            tags=("molecular", "orbmol", "conservative"),
            priority=37,
        ),
        _orb_entry(
            "orbmol-v1-direct",
            aliases=("orb-mol-v1-direct",),
            description="OrbMol v1 direct-force molecular potential.",
            tags=("molecular", "orbmol", "direct"),
            priority=38,
        ),
        _orb_entry(
            "orbmol-v2",
            aliases=("orb-mol-v2",),
            description="OrbMol v2 molecular potential with learnable electrostatics.",
            tags=("molecular", "orbmol", "electrostatics"),
            priority=39,
        ),
        _mattersim_entry(
            "mattersim-v1-1m",
            model_id="mattersim-v1-1m",
            aliases=("mattersim", "mattersim-v1", "mattersim-1m", "MatterSim-v1.0.0-1M"),
            description="MatterSim v1.0.0 1M atomistic foundation model.",
            tags=("v1", "1m"),
            priority=40,
        ),
        _mattersim_entry(
            "mattersim-v1-5m",
            model_id="mattersim-v1-5m",
            aliases=("mattersim-5m", "MatterSim-v1.0.0-5M"),
            description="MatterSim v1.0.0 5M atomistic foundation model.",
            tags=("v1", "5m"),
            priority=41,
        ),
        _sevennet_entry(
            "7net-omni",
            model_id="sevennet-omni",
            aliases=("sevennet", "7net-omni"),
            description="SevenNet-Omni multi-fidelity foundation potential.",
            tags=("omni", "multi-fidelity"),
            priority=42,
        ),
        _sevennet_entry(
            "7net-omni-i8",
            model_id="sevennet-omni-i8",
            aliases=("sevennet-omni-i8", "7net-omni-i8"),
            description="SevenNet-Omni i8 multi-fidelity foundation potential.",
            tags=("omni", "multi-fidelity", "i8"),
            priority=43,
        ),
        _sevennet_entry(
            "7net-omni-i12",
            model_id="sevennet-omni-i12",
            aliases=("sevennet-omni-i12", "7net-omni-i12"),
            description="SevenNet-Omni i12 multi-fidelity foundation potential.",
            tags=("omni", "multi-fidelity", "i12"),
            priority=44,
        ),
        _sevennet_entry(
            "7net-mf-ompa",
            model_id="sevennet-mf-ompa",
            aliases=("sevennet-mf", "7net-mf-ompa"),
            description="SevenNet multi-fidelity OMat24, MPtrj, and sAlex foundation potential.",
            tags=("omat24", "mptrj", "salex", "multi-fidelity"),
            priority=45,
        ),
        _sevennet_entry(
            "7net-omat",
            model_id="sevennet-omat",
            aliases=("sevennet-omat", "7net-omat"),
            description="SevenNet OMat24 pretrained foundation potential.",
            tags=("omat24",),
            priority=46,
        ),
        _sevennet_entry(
            "7net-l3i5",
            model_id="sevennet-l3i5",
            aliases=("sevennet-l3i5", "7net-l3i5"),
            description="SevenNet L3i5 pretrained foundation potential.",
            tags=("l3i5",),
            priority=47,
        ),
        _sevennet_entry(
            "7net-0",
            model_id="sevennet-0",
            aliases=("7net-0",),
            description="Original SevenNet-0 pretrained interatomic potential.",
            tags=("sevennet-0",),
            priority=48,
        ),
        _sevennet_entry(
            "7net-0_11Jul2024",
            model_id="sevennet-0-11jul2024",
            aliases=("7net-0_11Jul2024", "sevennet-0_11Jul2024"),
            description="SevenNet-0 checkpoint released on 11 Jul 2024.",
            tags=("sevennet-0", "2024"),
            priority=49,
        ),
        _eqv2_entry(
            "eqv2-s-oam",
            "eqV2_31M_omat_mp_salex.pt",
            model_id="eqv2-s-oam",
            aliases=("eqv2-31m-omat-mp-salex", "eqv2-31m-oam"),
            description="EquiformerV2 31M checkpoint trained on OMat24, MPtrj, and sAlex.",
            tags=("31m", "omat24", "mptrj", "salex", "oam"),
            priority=50,
        ),
        _eqv2_entry(
            "eqv2-m-oam",
            "eqV2_86M_omat_mp_salex.pt",
            model_id="eqv2-m-oam",
            aliases=("eqv2", "eqv2-m", "eqv2-86m-omat-mp-salex", "eqv2-86m-oam"),
            description="EquiformerV2 86M checkpoint trained on OMat24, MPtrj, and sAlex.",
            tags=("86m", "omat24", "mptrj", "salex", "oam"),
            priority=51,
        ),
        _eqv2_entry(
            "eqv2-l-oam",
            "eqV2_153M_omat_mp_salex.pt",
            model_id="eqv2-l-oam",
            aliases=("eqv2-l", "eqv2-153m-omat-mp-salex", "eqv2-153m-oam"),
            description="EquiformerV2 153M checkpoint trained on OMat24, MPtrj, and sAlex.",
            tags=("153m", "omat24", "mptrj", "salex", "oam"),
            priority=52,
        ),
        _eqv2_entry(
            "eqv2-s-omat",
            "eqV2_31M_omat.pt",
            model_id="eqv2-s-omat",
            aliases=("eqv2-31m-omat",),
            description="EquiformerV2 31M checkpoint trained on OMat24.",
            tags=("31m", "omat24"),
            priority=53,
        ),
        _eqv2_entry(
            "eqv2-m-omat",
            "eqV2_86M_omat.pt",
            model_id="eqv2-m-omat",
            aliases=("eqv2-86m-omat",),
            description="EquiformerV2 86M checkpoint trained on OMat24.",
            tags=("86m", "omat24"),
            priority=54,
        ),
        _eqv2_entry(
            "eqv2-l-omat",
            "eqV2_153M_omat.pt",
            model_id="eqv2-l-omat",
            aliases=("eqv2-153m-omat",),
            description="EquiformerV2 153M checkpoint trained on OMat24.",
            tags=("153m", "omat24"),
            priority=55,
        ),
        _eqv2_entry(
            "eqv2-s-mp",
            "eqV2_31M_mp.pt",
            model_id="eqv2-s-mp",
            aliases=("eqv2-31m-mp",),
            description="EquiformerV2 31M checkpoint trained on MPtrj.",
            tags=("31m", "mptrj"),
            priority=56,
        ),
        _eqv2_entry(
            "eqv2-s-dens",
            "eqV2_dens_31M_mp.pt",
            model_id="eqv2-s-dens",
            aliases=("eqv2-31m-dens",),
            description="EquiformerV2 DeNS 31M checkpoint trained on Materials Project data.",
            tags=("31m", "dens", "mp"),
            priority=57,
        ),
        _eqv2_entry(
            "eqv2-m-dens",
            "eqV2_dens_86M_mp.pt",
            model_id="eqv2-m-dens",
            aliases=("eqv2-86m-dens",),
            description="EquiformerV2 DeNS 86M checkpoint trained on Materials Project data.",
            tags=("86m", "dens", "mp"),
            priority=58,
        ),
        _eqv2_entry(
            "eqv2-l-dens",
            "eqV2_dens_153M_mp.pt",
            model_id="eqv2-l-dens",
            aliases=("eqv2-153m-dens",),
            description="EquiformerV2 DeNS 153M checkpoint trained on Materials Project data.",
            tags=("153m", "dens", "mp"),
            priority=59,
        ),
        _entry_from_adapter_spec(
            "matgl-tensornet-matpes-r2scan",
            "matgl:TensorNet-PES-MatPES-r2SCAN-2025.2",
            aliases=(
                "tensornet",
                "tensornet-matpes-r2scan",
                "TensorNet-PES-MatPES-r2SCAN-2025.2",
            ),
            source="builtin",
            description="MatGL TensorNet MatPES r2SCAN foundation potential.",
            tags=("foundation", "matgl", "tensornet", "matpes", "r2scan", "pes"),
            priority=10,
        ),
        _entry_from_adapter_spec(
            "matgl-tensornet-matpes-pbe",
            "matgl:TensorNet-PES-MatPES-PBE-2025.2",
            aliases=(
                "tensornet-matpes-pbe",
                "TensorNet-PES-MatPES-PBE-2025.2",
            ),
            source="builtin",
            description="MatGL TensorNet MatPES PBE foundation potential.",
            tags=("foundation", "matgl", "tensornet", "matpes", "pbe", "pes"),
            priority=20,
        ),
        _entry_from_adapter_spec(
            "matgl-m3gnet-matpes-pbe",
            "matgl:M3GNet-PES-MatPES-PBE-2025.2",
            aliases=(
                "m3gnet",
                "m3gnet-matpes-pbe",
                "M3GNet-PES-MatPES-PBE-2025.2",
            ),
            source="builtin",
            description="MatGL M3GNet MatPES PBE foundation potential.",
            tags=("foundation", "matgl", "m3gnet", "matpes", "pbe", "pes"),
            priority=30,
        ),
        _entry_from_adapter_spec(
            "matgl-m3gnet-matpes-r2scan",
            "matgl:M3GNet-PES-MatPES-r2SCAN-2025.2",
            aliases=(
                "m3gnet-matpes-r2scan",
                "M3GNet-PES-MatPES-r2SCAN-2025.2",
            ),
            source="builtin",
            description="MatGL M3GNet MatPES r2SCAN foundation potential.",
            tags=("foundation", "matgl", "m3gnet", "matpes", "r2scan", "pes"),
            priority=31,
        ),
        _entry_from_adapter_spec(
            "matgl-chgnet-matpes-pbe",
            "matgl:CHGNet-PES-MatPES-PBE-2025.2.10",
            aliases=(
                "chgnet",
                "chgnet-matpes-pbe",
                "CHGNet-PES-MatPES-PBE-2025.2.10",
            ),
            source="builtin",
            description="MatGL CHGNet MatPES PBE foundation potential.",
            tags=("foundation", "matgl", "chgnet", "matpes", "pbe", "pes"),
            priority=40,
        ),
        _entry_from_adapter_spec(
            "matgl-chgnet-matpes-r2scan",
            "matgl:CHGNet-PES-MatPES-r2SCAN-2025.2.10",
            aliases=(
                "chgnet-matpes-r2scan",
                "CHGNet-PES-MatPES-r2SCAN-2025.2.10",
            ),
            source="builtin",
            description="MatGL CHGNet MatPES r2SCAN foundation potential.",
            tags=("foundation", "matgl", "chgnet", "matpes", "r2scan", "pes"),
            priority=41,
        ),
        _entry_from_adapter_spec(
            "matgl-qet-matpes-pbe",
            "matgl:QET-PES-MatPES-PBE-2025.2",
            aliases=(
                "qet",
                "qet-matpes-pbe",
                "QET-PES-MatPES-PBE-2025.2",
            ),
            source="builtin",
            description="MatGL QET MatPES PBE foundation potential.",
            tags=("foundation", "matgl", "qet", "matpes", "pbe", "pes"),
            priority=50,
        ),
    ]
    return [entry for entry in entries if entry is not None]


def _load_registry_entries(registry_path: Optional[str]) -> list[FoundationModelEntry]:
    if registry_path is None:
        return []
    path = Path(registry_path).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_models = payload.get("models", payload) if isinstance(payload, dict) else {}
    entries: list[FoundationModelEntry] = []
    if not isinstance(raw_models, dict):
        return entries
    for key, value in raw_models.items():
        if isinstance(value, str):
            entry = _entry_from_adapter_spec(str(key), value, source=f"registry:{path}")
            if entry is not None:
                entries.append(entry)
        elif isinstance(value, dict) and isinstance(value.get("adapter_spec"), str):
            entry = _entry_from_adapter_spec(
                str(key),
                str(value["adapter_spec"]),
                aliases=value.get("aliases"),
                source=str(value.get("source", f"registry:{path}")),
                description=value.get("description"),
                elements=value.get("elements"),
                tags=value.get("tags"),
                capabilities=value.get("capabilities"),
                priority=int(value.get("priority", 100)),
            )
            if entry is not None:
                entries.append(entry)
    return entries


def _discover_hf_matgl_entries(limit: int = 64) -> tuple[list[FoundationModelEntry], list[str]]:
    warnings: list[str] = []
    entries: list[FoundationModelEntry] = []
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return [], [f"Hugging Face MatGL discovery is unavailable: {exc.__class__.__name__}: {exc}"]
    try:
        models = HfApi().list_models(filter="matgl", limit=limit)
        for model in models:
            model_id = str(getattr(model, "modelId", "") or "")
            if not model_id:
                continue
            bare_name = model_id.rsplit("/", 1)[-1]
            tags = [str(tag) for tag in getattr(model, "tags", []) or []]
            entry = _entry_from_adapter_spec(
                f"matgl:{model_id}",
                f"matgl:{model_id}",
                aliases=(bare_name, bare_name.lower(), model_id.lower()),
                source="huggingface_dynamic",
                description="MatGL-compatible pretrained model discovered from Hugging Face Hub.",
                tags=("matgl", "dynamic", "huggingface", *tags),
                priority=60,
            )
            if entry is not None:
                entries.append(entry)
    except Exception as exc:
        return [], [f"Hugging Face MatGL discovery failed: {exc.__class__.__name__}: {exc}"]
    return entries, warnings


def _discover_github_matgl_entries(timeout_sec: int = 10) -> tuple[list[FoundationModelEntry], list[str]]:
    url = "https://api.github.com/repos/materialsvirtuallab/matgl/contents/pretrained_models"
    warnings: list[str] = []
    entries: list[FoundationModelEntry] = []
    try:
        with urlopen(url, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return [], [f"MatGL dynamic discovery failed: {exc.__class__.__name__}: {exc}"]
    if not isinstance(payload, list):
        return [], ["MatGL dynamic discovery returned an unexpected payload."]
    for item in payload:
        if not isinstance(item, dict) or item.get("type") != "dir" or not item.get("name"):
            continue
        name = str(item["name"])
        entry = _entry_from_adapter_spec(
            f"matgl:{name}",
            f"matgl:{name}",
            aliases=(name, name.lower(), f"matgl-{name.lower()}"),
            source="matgl_dynamic",
            description="MatGL pretrained model discovered from the upstream pretrained_models registry.",
            tags=("matgl", "dynamic"),
            priority=50,
        )
        if entry is not None:
            entries.append(entry)
    return entries, warnings


def _discover_matgl_entries(timeout_sec: int = 10, limit: int = 64) -> tuple[list[FoundationModelEntry], list[str]]:
    entries, warnings = _discover_hf_matgl_entries(limit=limit)
    if entries:
        return entries, warnings
    fallback_entries, fallback_warnings = _discover_github_matgl_entries(timeout_sec=timeout_sec)
    return fallback_entries, [*warnings, *fallback_warnings]


def _discover_orb_github_entries(timeout_sec: int = 10, limit: int = 64) -> tuple[list[FoundationModelEntry], list[str]]:
    url = "https://raw.githubusercontent.com/orbital-materials/orb-models/main/MODELS.md"
    warnings: list[str] = []
    entries: list[FoundationModelEntry] = []
    try:
        with urlopen(url, timeout=timeout_sec) as response:
            text = response.read().decode("utf-8")
    except Exception as exc:
        return [], [f"ORB dynamic discovery failed: {exc.__class__.__name__}: {exc}"]

    names = []
    for match in re.finditer(r"`([^`]*orb[^`]*)`", text, flags=re.IGNORECASE):
        name = match.group(1).strip()
        if not name or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            continue
        lowered = name.lower()
        if lowered in {"orb-v3", "orb-v3-x-y-z"}:
            continue
        if lowered.startswith(("orb-", "orb_", "orbmol-", "orbmol_")) and lowered not in names:
            names.append(lowered)
        if len(names) >= limit:
            break

    for name in names:
        entry = _entry_from_adapter_spec(
            f"orb:{name}",
            f"orb:{name}",
            aliases=(name.replace("_", "-"), name.replace("-", "_")),
            source="orb_github_dynamic",
            description="ORB pretrained model discovered from the upstream model list.",
            tags=("orb", "dynamic", "pes"),
            priority=65,
        )
        if entry is not None:
            entries.append(entry)
    return entries, warnings


def _discover_hf_eqv2_entries(limit: int = 64) -> tuple[list[FoundationModelEntry], list[str]]:
    warnings: list[str] = []
    entries: list[FoundationModelEntry] = []
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return [], [f"Hugging Face eqV2 discovery is unavailable: {exc.__class__.__name__}: {exc}"]
    try:
        files = HfApi().list_repo_files("facebook/OMAT24")
    except Exception as exc:
        return [], [f"Hugging Face eqV2 discovery failed: {exc.__class__.__name__}: {exc}"]

    filenames = [
        str(path)
        for path in files
        if Path(str(path)).name.startswith("eqV2") and str(path).lower().endswith(".pt")
    ]
    filenames.sort()
    for filename in filenames[:limit]:
        stem = Path(filename).stem
        entry = _entry_from_adapter_spec(
            f"eqv2:{_safe_name(stem.lower())}",
            f"eqv2:facebook/OMAT24?filename={filename}",
            aliases=(stem, stem.lower(), stem.replace("_", "-").lower()),
            source="huggingface_dynamic",
            description="EquiformerV2 checkpoint discovered from facebook/OMAT24 on Hugging Face Hub.",
            tags=("eqv2", "equiformer", "omat24", "dynamic", "huggingface", "pes"),
            priority=66,
        )
        if entry is not None:
            entries.append(entry)
    return entries, warnings


def _discover_nequip_net_entries(timeout_sec: int = 10, limit: int = 64) -> tuple[list[FoundationModelEntry], list[str]]:
    url = "https://www.nequip.net/models"
    warnings: list[str] = []
    entries: list[FoundationModelEntry] = []
    try:
        with urlopen(url, timeout=timeout_sec) as response:
            text = response.read().decode("utf-8")
    except Exception as exc:
        return [], [f"nequip.net discovery failed: {exc.__class__.__name__}: {exc}"]

    names = []
    for match in re.finditer(r"mir-group/(NequIP-[A-Za-z0-9_-]+)", text):
        name = match.group(1)
        if name.lower().startswith("nequip-tutorial"):
            continue
        if name not in names:
            names.append(name)
        if len(names) >= limit:
            break

    for name in names:
        normalized = _safe_name(name.lower())
        entry = _entry_from_adapter_spec(
            f"nequip:{normalized}",
            f"nequip_net:mir-group/{name}?version=0.1",
            aliases=(name, name.lower(), f"mir-group/{name}"),
            source="nequip_net_dynamic",
            description="NequIP foundation potential discovered from nequip.net.",
            tags=("nequip", "nequip.net", "dynamic", "pes"),
            priority=67,
        )
        if entry is not None:
            entries.append(entry)
    return entries, warnings


def _discover_allegro_net_entries(timeout_sec: int = 10, limit: int = 64) -> tuple[list[FoundationModelEntry], list[str]]:
    url = "https://www.nequip.net/models"
    warnings: list[str] = []
    entries: list[FoundationModelEntry] = []
    try:
        with urlopen(url, timeout=timeout_sec) as response:
            text = response.read().decode("utf-8")
    except Exception as exc:
        return [], [f"nequip.net Allegro discovery failed: {exc.__class__.__name__}: {exc}"]

    names = []
    for match in re.finditer(r"mir-group/(Allegro-[A-Za-z0-9_-]+)", text):
        name = match.group(1)
        if name not in names:
            names.append(name)
        if len(names) >= limit:
            break

    for name in names:
        normalized = _safe_name(name.lower())
        entry = _entry_from_adapter_spec(
            f"allegro:{normalized}",
            f"allegro_net:mir-group/{name}?version=0.1",
            aliases=(name, name.lower(), f"mir-group/{name}"),
            source="nequip_net_dynamic",
            description="Allegro foundation potential discovered from nequip.net.",
            tags=("allegro", "nequip.net", "dynamic", "pes"),
            priority=68,
        )
        if entry is not None:
            entries.append(entry)
    return entries, warnings


def _select_nequip_hf_artifact(files: Sequence[str]) -> Optional[str]:
    preferred_suffixes = (".nequip.pth", ".pth", ".pt", ".zip")
    ignored_names = {"README.md", "LICENSE", "MODEL-LICENSE.md", ".gitattributes"}
    candidates = [
        str(path)
        for path in files
        if Path(str(path)).name not in ignored_names
        and str(path).lower().endswith(preferred_suffixes)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (not path.lower().endswith(".zip"), path))
    return candidates[0]


def _discover_hf_nequip_entries(limit: int = 32) -> tuple[list[FoundationModelEntry], list[str]]:
    warnings: list[str] = []
    entries: list[FoundationModelEntry] = []
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return [], [f"Hugging Face NequIP discovery is unavailable: {exc.__class__.__name__}: {exc}"]
    api = HfApi()
    try:
        models = api.list_models(search="nequip", limit=limit)
        for model in models:
            repo_id = str(getattr(model, "modelId", "") or "")
            if not repo_id:
                continue
            try:
                files = api.list_repo_files(repo_id)
            except Exception as exc:
                warnings.append(f"Skipping NequIP repo {repo_id!r}: cannot list files ({exc.__class__.__name__}: {exc}).")
                continue
            filename = _select_nequip_hf_artifact(files)
            if filename is None:
                warnings.append(f"Skipping NequIP repo {repo_id!r}: no packaged model artifact found.")
                continue
            bare_name = repo_id.rsplit("/", 1)[-1]
            tags = [str(tag) for tag in getattr(model, "tags", []) or []]
            entry = _entry_from_adapter_spec(
                f"nequip:{repo_id}",
                f"nequip_hf:{repo_id}?filename={filename}",
                aliases=(bare_name, bare_name.lower(), repo_id),
                source="huggingface_dynamic",
                description="NequIP-compatible packaged model discovered from Hugging Face Hub.",
                tags=("nequip", "dynamic", "huggingface", "pes", *tags),
                priority=70,
            )
            if entry is not None:
                entries.append(entry)
    except Exception as exc:
        return [], [f"Hugging Face NequIP discovery failed: {exc.__class__.__name__}: {exc}"]
    return entries, warnings


def _model_entries(
    registry_path: Optional[str],
    *,
    include_builtin: bool = True,
    include_dynamic: bool = False,
    discovery_timeout_sec: int = 10,
    discovery_limit: int = 64,
) -> tuple[list[FoundationModelEntry], list[str]]:
    warnings: list[str] = []
    entries: list[FoundationModelEntry] = []
    if include_builtin:
        entries.extend(_builtin_model_entries())
    entries.extend(_load_registry_entries(registry_path))
    if include_dynamic:
        dynamic_entries, dynamic_warnings = _discover_matgl_entries(discovery_timeout_sec, discovery_limit)
        entries.extend(dynamic_entries)
        warnings.extend(dynamic_warnings)
        orb_entries, orb_warnings = _discover_orb_github_entries(discovery_timeout_sec, discovery_limit)
        entries.extend(orb_entries)
        warnings.extend(orb_warnings)
        eqv2_entries, eqv2_warnings = _discover_hf_eqv2_entries(discovery_limit)
        entries.extend(eqv2_entries)
        warnings.extend(eqv2_warnings)
        nequip_net_entries, nequip_net_warnings = _discover_nequip_net_entries(discovery_timeout_sec, discovery_limit)
        entries.extend(nequip_net_entries)
        warnings.extend(nequip_net_warnings)
        allegro_net_entries, allegro_net_warnings = _discover_allegro_net_entries(discovery_timeout_sec, discovery_limit)
        entries.extend(allegro_net_entries)
        warnings.extend(allegro_net_warnings)
        nequip_entries, nequip_warnings = _discover_hf_nequip_entries(discovery_limit)
        entries.extend(nequip_entries)
        warnings.extend(nequip_warnings)

    deduped: dict[str, FoundationModelEntry] = {}
    specs_seen: set[str] = set()
    for entry in entries:
        if entry.backend not in registered_adapter_schemes():
            warnings.append(f"Skipping {entry.id!r}: backend {entry.backend!r} is not registered.")
            continue
        if entry.adapter_spec in specs_seen:
            continue
        specs_seen.add(entry.adapter_spec)
        deduped[entry.id] = entry
    return sorted(deduped.values(), key=lambda item: (item.priority, item.id)), warnings


def _alias_map(
    registry_path: Optional[str],
    *,
    include_builtin: bool = True,
) -> Dict[str, str]:
    entries, _ = _model_entries(
        registry_path,
        include_builtin=include_builtin,
        include_dynamic=False,
    )
    aliases: Dict[str, str] = {}
    for entry in entries:
        for alias in (entry.id, *entry.aliases):
            aliases[str(alias)] = entry.adapter_spec
    return aliases


def _normalize_params(params: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    if params is None:
        return {}
    return {str(key): str(value) for key, value in params.items() if value is not None}


def _resolve_adapter_spec(
    model_id: str,
    *,
    backend: Optional[str],
    resource: Optional[str],
    params: Optional[Mapping[str, Any]],
    registry_path: Optional[str],
) -> tuple[str, ExternalModelSpec, Optional[str]]:
    aliases = _alias_map(registry_path)
    params_dict = _normalize_params(params)
    alias_used: Optional[str] = None

    if model_id in aliases:
        alias_used = model_id
        raw_spec = aliases[model_id]
        if params_dict:
            parsed = parse_external_model_spec(raw_spec)
            if parsed is None:
                raise ValueError(f"Registry alias {model_id!r} does not contain a valid adapter spec.")
            parsed.params.update(params_dict)
            raw_spec = format_external_model_spec(parsed)
    elif backend is not None:
        raw_spec = format_external_model_spec(
            ExternalModelSpec(
                scheme=str(backend).strip().lower(),
                resource=str(resource if resource is not None else model_id),
                params=params_dict,
            )
        )
    else:
        raw_spec = model_id

    parsed = parse_external_model_spec(raw_spec)
    if parsed is None:
        known = ", ".join(registered_adapter_schemes())
        raise ValueError(
            "model_id must be an external adapter spec like 'mace:/path/model.pt', "
            "or pass backend/resource explicitly. Known backends: "
            + known
        )
    if not is_external_model_spec(raw_spec):
        known = ", ".join(registered_adapter_schemes())
        raise ValueError(f"Unsupported model backend {parsed.scheme!r}. Known backends: {known}")
    return raw_spec, parsed, alias_used


def _prepare_resource(
    spec: ExternalModelSpec,
    *,
    cache_dir: Optional[str],
    download: bool,
    expected_sha256: Optional[str],
    hash_resource: bool,
    timeout_sec: int,
) -> tuple[ExternalModelSpec, ResourceInfo, list[str]]:
    warnings: list[str] = []
    resource = spec.resource
    resolved_resource = resource
    cache_hit = False
    checksum: Optional[str] = None
    local_file_schemes = {"mace", "nequip", "allegro"}

    if _is_url(resource):
        if not download:
            warnings.append("Remote resource was left unresolved because download=False.")
            return spec, ResourceInfo("remote", resource, resolved_resource), warnings
        root = Path(cache_dir or os.environ.get("CURATOR_MODEL_CACHE", "~/.cache/curator/models")).expanduser()
        target, cache_hit = _download_to_cache(resource, root / spec.scheme, timeout_sec)
        resolved_resource = str(target.resolve())
        checksum = _sha256_file(target)
    else:
        candidate = Path(resource).expanduser()
        if candidate.exists():
            resolved_resource = str(candidate.resolve())
            if expected_sha256 is not None or hash_resource:
                checksum = _sha256_file(candidate)
        elif spec.scheme in local_file_schemes and _looks_like_path(resource):
            raise FileNotFoundError(f"Model resource path does not exist: {resource}")

    if expected_sha256 is not None:
        if checksum is None:
            checksum = _sha256_file(Path(resolved_resource))
        if checksum.lower() != expected_sha256.lower():
            raise ValueError(
                f"sha256 mismatch for {resource}: expected {expected_sha256}, got {checksum}"
            )

    prepared = replace(spec, resource=resolved_resource)
    kind = "url" if _is_url(resource) else "local" if Path(resolved_resource).exists() else "named"
    return prepared, ResourceInfo(kind, resource, resolved_resource, cache_hit, checksum), warnings


def _declared_capabilities(spec: ExternalModelSpec) -> Dict[str, Optional[bool]]:
    capabilities = {key: None for key in _CAPABILITY_KEYS}
    capabilities.update(_SCHEME_CAPABILITIES.get(spec.scheme, {}))
    if spec.scheme == "matgl":
        capabilities["stress"] = parse_bool(spec.params.get("calc_stresses"), False)
    return capabilities


def _normalize_elements(elements: Optional[Sequence[str]]) -> list[str]:
    if elements is None:
        return []
    normalized = []
    for element in elements:
        token = str(element).strip()
        if token:
            normalized.append(token)
    return normalized


def _normalize_requirements(require: Optional[Mapping[str, Any]]) -> Dict[str, bool]:
    if require is None:
        return {}
    return {str(key): bool(value) for key, value in require.items()}


def _check_requirements(
    capabilities: Mapping[str, Optional[bool]],
    require: Mapping[str, bool],
) -> tuple[list[str], list[str]]:
    failures = []
    warnings = []
    for key, required in require.items():
        if not required:
            continue
        actual = capabilities.get(key)
        if actual is False:
            failures.append(key)
        elif actual is None:
            warnings.append(f"Capability {key!r} is not declared; run with probe=True or audit before relying on it.")
    return failures, warnings


def _model_class_name(model: Any) -> str:
    return f"{model.__class__.__module__}.{model.__class__.__name__}"


def _infer_cutoff(model: Any) -> Optional[float]:
    candidates = (
        getattr(model, "cutoff", None),
        getattr(getattr(model, "representation", None), "cutoff", None),
        getattr(getattr(model, "model", None), "cutoff", None),
        getattr(getattr(getattr(model, "model", None), "representation", None), "cutoff", None),
    )
    for value in candidates:
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _infer_supported_elements(model: Any) -> Optional[list[str]]:
    candidates = (
        getattr(model, "species", None),
        getattr(model, "type_names", None),
        getattr(getattr(model, "representation", None), "species", None),
        getattr(getattr(model, "representation", None), "type_names", None),
        getattr(getattr(model, "core_model", None), "element_types", None),
        getattr(getattr(model, "model", None), "element_types", None),
    )
    for value in candidates:
        if value is not None:
            return [str(item) for item in value]
    return None


def _elements_supported(requested: Sequence[str], supported: Optional[Sequence[str]]) -> tuple[bool, list[str]]:
    if not requested or supported is None:
        return True, []
    supported_set = {str(item) for item in supported}
    missing = []
    for item in requested:
        if item in supported_set:
            continue
        if item.isdigit():
            z = int(item)
            if z < len(ase_chemical_symbols) and ase_chemical_symbols[z] in supported_set:
                continue
        if item in ase_atomic_numbers and str(ase_atomic_numbers[item]) in supported_set:
            continue
        missing.append(item)
    return not missing, missing


def _probe_model(adapter_spec: str, device: str, capabilities: Dict[str, Optional[bool]]) -> Dict[str, Any]:
    model = load_external_model(adapter_spec, device=device)
    outputs = [str(item) for item in getattr(model, "model_outputs", [])]
    if outputs:
        capabilities["energy"] = capabilities["energy"] or properties.energy in outputs
        capabilities["forces"] = capabilities["forces"] or properties.forces in outputs
        capabilities["stress"] = properties.stress in outputs
    if hasattr(model, "readout_mlp") or hasattr(model, "final_layer"):
        capabilities["features"] = True

    try:
        num_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    except Exception:
        num_parameters = None

    return {
        "load_ok": True,
        "model_class": _model_class_name(model),
        "num_parameters": num_parameters,
        "cutoff": _infer_cutoff(model),
        "model_outputs": outputs,
        "supported_elements": _infer_supported_elements(model),
    }


def _base_result(
    *,
    ok: bool,
    status: str,
    model_id: str,
    adapter_spec: Optional[str],
    manifest_path: Path,
    work_dir: Path,
    error: Optional[ErrorInfo] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ok": ok,
        "status": status,
        "model_id": model_id,
        "adapter_spec": adapter_spec,
        "artifacts": {
            "work_dir": str(work_dir),
            "manifest": str(manifest_path),
        },
    }
    if error is not None:
        result["error"] = asdict(error)
    return result


def _entry_to_dict(entry: FoundationModelEntry) -> Dict[str, Any]:
    return {
        "id": entry.id,
        "adapter_spec": entry.adapter_spec,
        "backend": entry.backend,
        "aliases": list(entry.aliases),
        "source": entry.source,
        "capabilities": dict(entry.capabilities),
        "description": entry.description,
        "elements": None if entry.elements is None else list(entry.elements),
        "tags": [] if entry.tags is None else list(entry.tags),
        "priority": entry.priority,
    }


def _candidate_rejection_reasons(
    entry: FoundationModelEntry,
    *,
    elements: Sequence[str],
    require: Mapping[str, bool],
    backend: Optional[str],
    potential_only: bool,
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    allowed_backends = _BACKEND_GROUPS.get(backend, {backend}) if backend is not None else None
    if allowed_backends is not None and entry.backend not in allowed_backends:
        reasons.append(f"backend {entry.backend!r} does not match requested backend {backend!r}")
    if potential_only and not _looks_like_potential_entry(entry):
        reasons.append("not identified as a PES / interatomic potential model")

    requirement_failures, requirement_warnings = _check_requirements(entry.capabilities, require)
    reasons.extend(f"requires unsupported capability {item!r}" for item in requirement_failures)
    warnings.extend(requirement_warnings)

    if elements and entry.elements is not None:
        supported, missing = _elements_supported(elements, entry.elements)
        if not supported:
            reasons.append("missing requested elements: " + ", ".join(missing))
    return reasons, warnings


def list_foundation_models(
    elements: Optional[Sequence[str]] = None,
    require: Optional[Mapping[str, Any]] = None,
    backend: Optional[str] = None,
    registry_path: Optional[str] = None,
    potential_only: bool = True,
    include_builtin: bool = True,
    include_dynamic: bool = False,
    discovery_timeout_sec: int = 10,
    discovery_limit: int = 64,
    out: Optional[str] = None,
) -> Dict[str, Any]:
    requested_elements = _normalize_elements(elements)
    requirements = _normalize_requirements(require)
    normalized_backend = str(backend).strip().lower() if backend else None
    entries, warnings = _model_entries(
        registry_path,
        include_builtin=include_builtin,
        include_dynamic=include_dynamic,
        discovery_timeout_sec=discovery_timeout_sec,
        discovery_limit=discovery_limit,
    )

    candidates: list[Dict[str, Any]] = []
    rejected: list[Dict[str, Any]] = []
    for entry in entries:
        reasons, candidate_warnings = _candidate_rejection_reasons(
            entry,
            elements=requested_elements,
            require=requirements,
            backend=normalized_backend,
            potential_only=potential_only,
        )
        payload = _entry_to_dict(entry)
        if candidate_warnings:
            payload["warnings"] = candidate_warnings
        if reasons:
            payload["reasons"] = reasons
            rejected.append(payload)
        else:
            candidates.append(payload)

    result: Dict[str, Any] = {
        "ok": True,
        "status": "completed",
        "requested_at": _utc_now(),
        "filters": {
            "elements": requested_elements,
            "require": requirements,
            "backend": normalized_backend,
            "potential_only": potential_only,
            "include_builtin": include_builtin,
            "include_dynamic": include_dynamic,
            "discovery_limit": discovery_limit,
        },
        "known_backends": registered_adapter_schemes(),
        "num_candidates": len(candidates),
        "num_rejected": len(rejected),
        "candidates": candidates,
        "rejected": rejected,
        "warnings": warnings,
    }
    if out is not None:
        out_path = Path(out).expanduser() / "foundation_model_candidates.json"
        _write_json(out_path, result)
        result["artifacts"] = {"candidates": str(out_path)}
    return result


def fetch_model(
    model_id: str,
    *,
    backend: Optional[str] = None,
    resource: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
    elements: Optional[Sequence[str]] = None,
    require: Optional[Mapping[str, Any]] = None,
    out: str = "model_fetch",
    cache_dir: Optional[str] = None,
    registry_path: Optional[str] = None,
    download: bool = True,
    expected_sha256: Optional[str] = None,
    hash_resource: bool = False,
    probe: bool = False,
    device: str = "cpu",
    timeout_sec: int = 300,
) -> Dict[str, Any]:
    """Resolve and optionally verify an external pretrained MLIP for Curator.

    The returned ``adapter_spec`` is intentionally compatible with Curator's
    existing ``load_models`` path, while the manifest carries the richer
    provenance and capability data needed by agent workflows.
    """
    started_at = _utc_now()
    adapter_hint = f"{backend or ''}:{resource or model_id}"
    work_dir = _work_dir(model_id, adapter_hint, out)
    manifest_path = work_dir / "model_manifest.json"

    try:
        raw_spec, parsed, alias_used = _resolve_adapter_spec(
            model_id,
            backend=backend,
            resource=resource,
            params=params,
            registry_path=registry_path,
        )
        work_dir = _work_dir(model_id, raw_spec, out)
        manifest_path = work_dir / "model_manifest.json"
        prepared, resource_info, resource_warnings = _prepare_resource(
            parsed,
            cache_dir=cache_dir,
            download=download,
            expected_sha256=expected_sha256,
            hash_resource=hash_resource,
            timeout_sec=timeout_sec,
        )
        adapter_spec = format_external_model_spec(prepared)
        capabilities = _declared_capabilities(prepared)
        requested_elements = _normalize_elements(elements)
        requirements = _normalize_requirements(require)
        failures, requirement_warnings = _check_requirements(capabilities, requirements)
        warnings = [*resource_warnings, *requirement_warnings]
        probe_result: Optional[Dict[str, Any]] = None

        if probe:
            try:
                probe_result = _probe_model(adapter_spec, device, capabilities)
            except Exception as exc:
                result = _base_result(
                    ok=False,
                    status="probe_failed",
                    model_id=model_id,
                    adapter_spec=adapter_spec,
                    manifest_path=manifest_path,
                    work_dir=work_dir,
                    error=ErrorInfo(type=exc.__class__.__name__, message=str(exc), recoverable=True),
                )
                result.update(
                    {
                        "requested_at": started_at,
                        "completed_at": _utc_now(),
                        "raw_adapter_spec": raw_spec,
                        "parsed_spec": asdict(prepared),
                        "resource": asdict(resource_info),
                        "capabilities": capabilities,
                        "warnings": warnings,
                    }
                )
                _write_json(manifest_path, result)
                return result

        if probe_result is not None:
            supported_ok, missing_elements = _elements_supported(
                requested_elements,
                probe_result.get("supported_elements"),
            )
            if not supported_ok:
                failures.append("elements")
                warnings.append(f"Requested elements are not supported by this model: {missing_elements}")

        requirement_failures, late_warnings = _check_requirements(capabilities, requirements)
        failures = list(dict.fromkeys([*failures, *requirement_failures]))
        warnings.extend(late_warnings)
        status = "ready" if probe else "resolved"
        ok = not failures
        if failures:
            status = "unsupported_requirement"

        result = _base_result(
            ok=ok,
            status=status,
            model_id=model_id,
            adapter_spec=adapter_spec,
            manifest_path=manifest_path,
            work_dir=work_dir,
            error=None
            if ok
            else ErrorInfo(
                type="UnsupportedRequirement",
                message="Unsupported requirements: " + ", ".join(sorted(set(failures))),
                recoverable=True,
            ),
        )
        result.update(
            {
                "requested_at": started_at,
                "completed_at": _utc_now(),
                "backend": prepared.scheme,
                "raw_adapter_spec": raw_spec,
                "parsed_spec": asdict(prepared),
                "alias_used": alias_used,
                "resource": asdict(resource_info),
                "capabilities": capabilities,
                "requirements": requirements,
                "elements": requested_elements,
                "probe": probe_result,
                "warnings": warnings,
                "environment": {
                    "python": sys.version.split()[0],
                },
            }
        )
        _write_json(manifest_path, result)
        return result
    except Exception as exc:
        result = _base_result(
            ok=False,
            status="failed",
            model_id=model_id,
            adapter_spec=None,
            manifest_path=manifest_path,
            work_dir=work_dir,
            error=ErrorInfo(type=exc.__class__.__name__, message=str(exc), recoverable=True),
        )
        result.update(
            {
                "requested_at": started_at,
                "completed_at": _utc_now(),
                "known_backends": registered_adapter_schemes(),
            }
        )
        _write_json(manifest_path, result)
        return result


__all__ = [
    "fetch_model",
    "list_foundation_models",
    "ErrorInfo",
    "ResourceInfo",
    "FoundationModelEntry",
]
