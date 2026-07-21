from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import urlopen


_NEQUIP_NET_ARTIFACTS = {
    ("mir-group/NequIP-MP-L", "0.1"): ("18775904", "NequIP-MP-L-0.1.nequip.zip"),
    ("mir-group/NequIP-OAM-S", "0.1"): ("18775904", "NequIP-OAM-S-0.1.nequip.zip"),
    ("mir-group/NequIP-OAM-M", "0.1"): ("18775904", "NequIP-OAM-M-0.1.nequip.zip"),
    ("mir-group/NequIP-OAM-L", "0.1"): ("18775904", "NequIP-OAM-L-0.1.nequip.zip"),
    ("mir-group/NequIP-OAM-XL", "0.1"): ("18775904", "NequIP-OAM-XL-0.1.nequip.zip"),
    ("mir-group/Allegro-MP-L", "0.1"): ("16980200", "Allegro-MP-L-0.1.nequip.zip"),
    ("mir-group/Allegro-OAM-L", "0.1"): ("16980200", "Allegro-OAM-L-0.1.nequip.zip"),
}


def _is_url(resource: str) -> bool:
    return urlparse(resource).scheme in {"http", "https"}


def _cache_target(url: str, *, cache_dir: Optional[str]) -> Path:
    root = Path(
        cache_dir or os.environ.get("CURATOR_MODEL_CACHE", "~/.cache/curator/models")
    ).expanduser()
    name = Path(urlparse(url).path).name
    if not name or name == "content":
        name = "model.nequip.zip"
    target_dir = root / "nequip.net" / hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return target_dir / name


def _download(url: str, *, cache_dir: Optional[str], timeout_sec: int, download: bool) -> str:
    target = _cache_target(url, cache_dir=cache_dir)
    if target.exists():
        return str(target)
    if not download:
        raise FileNotFoundError(
            "nequip.net artifact is not cached and download=False. "
            f"URL: {url}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.download")
    with urlopen(url, timeout=timeout_sec) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(tmp, target)
    return str(target)


def parse_nequip_net_ref(resource: str, version: str = "0.1") -> tuple[str, str]:
    token = resource.strip()
    if token.startswith("nequip.net:"):
        token = token[len("nequip.net:") :]
    if ":" in token:
        token, parsed_version = token.rsplit(":", 1)
        version = parsed_version or version
    if "/" not in token:
        token = f"mir-group/{token}"
    return token, version


def nequip_net_url(resource: str, version: str = "0.1") -> Optional[str]:
    model_id, resolved_version = parse_nequip_net_ref(resource, version=version)
    artifact = _NEQUIP_NET_ARTIFACTS.get((model_id, resolved_version))
    if artifact is None:
        return None
    record_id, filename = artifact
    return f"https://zenodo.org/api/records/{record_id}/files/{filename}/content"


def resolve_nequip_net_artifact(
    resource: str,
    *,
    version: str = "0.1",
    cache_dir: Optional[str] = None,
    download: bool = True,
    timeout_sec: int = 300,
) -> str:
    path = Path(resource).expanduser()
    if path.exists():
        return str(path)
    if _is_url(resource):
        return _download(resource, cache_dir=cache_dir, timeout_sec=timeout_sec, download=download)
    url = nequip_net_url(resource, version=version)
    if url is None:
        model_id, resolved_version = parse_nequip_net_ref(resource, version=version)
        return f"nequip.net:{model_id}:{resolved_version}"
    return _download(url, cache_dir=cache_dir, timeout_sec=timeout_sec, download=download)


__all__ = [
    "nequip_net_url",
    "parse_nequip_net_ref",
    "resolve_nequip_net_artifact",
]
