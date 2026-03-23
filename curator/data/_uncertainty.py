from __future__ import annotations

from typing import List, Optional, Tuple

from torch import nn


class UncertaintyModule(nn.Module):
    """Thin nn.Module base for modules that explicitly produce uncertainty outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.uncertainty_keys: Tuple[str, ...] = ()
        self.per_atom_uncertainty_keys: Tuple[str, ...] = ()

    def set_uncertainty_outputs(
        self,
        scalar_keys: Optional[List[str]] = None,
        per_atom_keys: Optional[List[str]] = None,
    ) -> None:
        scalar = list(dict.fromkeys(str(key) for key in (scalar_keys or [])))
        per_atom = list(dict.fromkeys(str(key) for key in (per_atom_keys or [])))
        self.uncertainty_keys = tuple(scalar)
        self.per_atom_uncertainty_keys = tuple(per_atom)


def collect_uncertainty_outputs(module: nn.Module) -> Tuple[List[str], List[str]]:
    """Return declared scalar and per-atom uncertainty keys for a module tree."""

    scalar_keys: List[str] = []
    per_atom_keys: List[str] = []

    def add_from(candidate: nn.Module) -> None:
        for key in getattr(candidate, "uncertainty_keys", ()):
            if key not in scalar_keys:
                scalar_keys.append(key)
        for key in getattr(candidate, "per_atom_uncertainty_keys", ()):
            if key not in per_atom_keys:
                per_atom_keys.append(key)

    add_from(module)
    for child in getattr(module, "output_modules", []):
        add_from(child)

    return scalar_keys, per_atom_keys
