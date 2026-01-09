from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, Iterator
from collections.abc import MutableMapping

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
class AtomsData(MutableMapping[str, Any]):
    atoms: Dict[str, torch.Tensor]
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

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ) -> "AtomsData":
        return AtomsData(
            atoms=self._move(self.atoms, device=device, dtype=dtype, non_blocking=non_blocking),
            targets=self._move(self.targets, device=device, dtype=dtype, non_blocking=non_blocking),
            task=self.task,
            weight=self.weight,
            meta=self.meta,
        )

    @staticmethod
    def _move(
        values: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ) -> Dict[str, torch.Tensor]:
        moved: Dict[str, torch.Tensor] = {}
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(device=device, dtype=dtype, non_blocking=non_blocking)
            else:
                moved[k] = v
        return moved

    def to_dict(
        self,
        include_meta: bool = False,
        include_task: bool = False,
        include_weight: bool = False,
    ) -> Dict[str, Any]:
        data = dict(self.atoms)
        data.update(self.targets)
        if include_task:
            data["task"] = self.task
        if include_weight:
            data["weight"] = self.weight
        if include_meta and self.meta is not None:
            data["meta"] = self.meta
        return data

    def __getitem__(self, key: str) -> Any:
        if key == "task":
            return self.task
        if key == "weight":
            return self.weight
        if key == "meta":
            return self.meta
        if key == "atoms":
            return self.atoms
        if key == "targets":
            return self.targets
        if key in self.atoms:
            return self.atoms[key]
        norm_key = normalize_target_key(key)
        if norm_key in self.targets:
            return self.targets[norm_key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "task":
            self.task = str(value)
            return
        if key == "weight":
            self.weight = float(value)
            return
        if key == "meta":
            self.meta = value
            return
        if key == "atoms":
            self.atoms = dict(value)
            return
        if key == "targets":
            self.targets = dict(value)
            return
        norm_key = normalize_target_key(key)
        if norm_key in _TARGET_KEY_ALIASES.values() or norm_key in self.targets:
            self.targets[norm_key] = value
        else:
            self.atoms[key] = value

    def __delitem__(self, key: str) -> None:
        if key in {"task", "weight", "meta"}:
            raise KeyError(f"Cannot delete '{key}' from AtomsData.")
        if key in self.atoms:
            del self.atoms[key]
            return
        norm_key = normalize_target_key(key)
        if norm_key in self.targets:
            del self.targets[norm_key]
            return
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yielded = set()
        for key in self.atoms:
            if key not in yielded:
                yielded.add(key)
                yield key
        for key in self.targets:
            if key not in yielded:
                yielded.add(key)
                yield key

    def __len__(self) -> int:
        return len(set(self.atoms.keys()) | set(self.targets.keys()))


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
