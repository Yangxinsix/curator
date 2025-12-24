from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from . import properties

_TARGET_KEY_ALIASES: Dict[str, str] = {
    "energy": properties.energy,
    "forces": properties.forces,
    "stress": properties.stress,
    "virial": properties.virial,
    "charge": properties.total_charge,
    "charges": properties.atomic_charge,
    "total_charge": properties.total_charge,
    "atomic_charge": properties.atomic_charge,
    "dipole": properties.dipole,
    "total_magmom": properties.total_magmom,
    "atomic_energy": properties.atomic_energy,
}

_ATOMWISE_TARGETS = {
    properties.forces,
    properties.atomic_charge,
    properties.atomic_energy,
    properties.ewald_forces,
    properties.residual_forces,
}


def normalize_target_key(key: str) -> str:
    return _TARGET_KEY_ALIASES.get(key, key)


def is_atomwise_target(key: str) -> bool:
    return key in _ATOMWISE_TARGETS


@dataclass
class AtomsData:
    atoms: Any
    targets: Dict[str, torch.Tensor]
    task: str
    weight: float = 1.0
    meta: Optional[Dict[str, Any]] = None

    def get_target(self, key: str, default=None):
        key = normalize_target_key(key)
        return self.targets.get(key, default)

    def normalized_targets(self) -> Dict[str, torch.Tensor]:
        normalized: Dict[str, torch.Tensor] = {}
        for k, v in self.targets.items():
            nk = normalize_target_key(k)
            if nk not in normalized:
                normalized[nk] = v
        return normalized

    def validate(self) -> None:
        for k, v in self.targets.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"Target '{k}' must be a torch.Tensor.")


def split_atoms_targets(atoms_dict: Dict[str, torch.Tensor]) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    targets: Dict[str, torch.Tensor] = {}
    atoms = dict(atoms_dict)
    for key in list(atoms.keys()):
        if normalize_target_key(key) in _TARGET_KEY_ALIASES.values():
            targets[normalize_target_key(key)] = atoms.pop(key)
    return atoms, targets


def atoms_data_from_dict(
    atoms_dict: Dict[str, torch.Tensor],
    task: str,
    weight: float = 1.0,
    meta: Optional[Dict[str, Any]] = None,
) -> AtomsData:
    atoms, targets = split_atoms_targets(atoms_dict)
    return AtomsData(atoms=atoms, targets=targets, task=task, weight=weight, meta=meta)


def get_sample_atoms(sample: Any) -> Dict[str, torch.Tensor]:
    if isinstance(sample, AtomsData):
        return sample.atoms
    return sample


def get_sample_target(sample: Any, key: str):
    if isinstance(sample, AtomsData):
        return sample.get_target(key)
    return sample.get(key)
