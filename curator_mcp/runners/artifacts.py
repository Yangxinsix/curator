from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from curator.utils import (
    ensure_dir,
    error_result,
    path_status,
    read_json,
    to_jsonable,
    utc_now,
    write_json as _write_json,
    write_text as _write_text,
)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    return str(_write_json(path, payload, sort_keys=True, atomic=True))


def write_text(path: str | Path, text: str) -> str:
    return str(_write_text(path, text))


__all__ = [
    "ensure_dir",
    "error_result",
    "path_status",
    "read_json",
    "to_jsonable",
    "utc_now",
    "write_json",
    "write_text",
]
