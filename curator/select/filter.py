from typing import Dict, List, Optional, Sequence, Union, Literal

import torch
from ase import Atoms
from ase.data import atomic_numbers
from torch.utils.data import DataLoader

from curator.data import properties
import logging

logger = logging.getLogger(__name__)


class Filter:
    def filter_dataset(self, dataset, label: Optional[str] = None):
        return dataset


FilteringType = Literal["none", "subset", "superset", "exact", "overlap"]


class ElementFilter(Filter):
    def __init__(
        self,
        required_elements: Optional[Sequence[Union[str, int]]] = None,
        filtering_type: FilteringType = "superset",
    ) -> None:
        self.required_elements = required_elements
        self.filtering_type = filtering_type
        self.required_numbers = self._resolve_required_numbers(required_elements)

    @staticmethod
    def _resolve_required_numbers(
        required_elements: Optional[Sequence[Union[str, int]]],
    ) -> Optional[set]:
        if not required_elements:
            return None
        if isinstance(required_elements, str) and required_elements.lower() == "all":
            return None
        numbers = set()
        for elem in required_elements:
            if isinstance(elem, str):
                if elem.isdigit():
                    numbers.add(int(elem))
                else:
                    numbers.add(int(atomic_numbers[elem]))
            else:
                numbers.add(int(elem))
        return numbers

    def _matches(self, present: set) -> bool:
        filtering = str(self.filtering_type).lower() if self.filtering_type is not None else "superset"
        if filtering == "none" or self.required_numbers is None:
            return True
        if filtering == "subset":
            return present.issubset(self.required_numbers)
        if filtering == "exact":
            return present == self.required_numbers
        if filtering == "superset":
            return self.required_numbers.issubset(present)
        if filtering == "overlap":
            return bool(present.intersection(self.required_numbers))
        raise ValueError(
            "Filtering type is not recognised. Must be one of: "
            "none, subset, superset, exact, overlap."
        )

    @staticmethod
    def _numbers_from_item(item) -> Optional[set]:
        if isinstance(item, Atoms):
            return set(item.get_atomic_numbers().tolist())
        if isinstance(item, dict) and properties.Z in item:
            numbers = item[properties.Z]
        else:
            numbers = getattr(item, "atoms", {}).get(properties.Z) if hasattr(item, "atoms") else None
        if isinstance(numbers, torch.Tensor):
            return set(numbers.tolist())
        return None

    def filter_atoms(self, atoms_list: Sequence[Atoms]) -> List[Atoms]:
        if self.required_numbers is None or str(self.filtering_type).lower() == "none":
            return list(atoms_list)
        filtered: List[Atoms] = []
        for atoms in atoms_list:
            if self._matches(set(atoms.get_atomic_numbers().tolist())):
                filtered.append(atoms)
        return filtered

    def filter_dataset(self, dataset, label: Optional[str] = None):
        if self.required_numbers is None or str(self.filtering_type).lower() == "none":
            return dataset
        base = dataset.dataset if isinstance(dataset, DataLoader) else dataset
        db = getattr(base, "db", None)
        source = db if db is not None else base
        if not hasattr(source, "__len__"):
            return dataset
        indices: List[int] = []
        for i in range(len(source)):
            item = source[i] if db is not None else base[i]
            present = self._numbers_from_item(item)
            if present is None:
                continue
            if self._matches(present):
                indices.append(i)
        if label is not None and hasattr(source, "__len__"):
            logger.info("Filtered %s dataset: %d -> %d", label, len(source), len(indices))
        return torch.utils.data.Subset(base, indices)


class ForceFilter(Filter):
    def __init__(
        self,
        min_force: float = 0.0,
        max_force: float = 20.0,
    ) -> None:
        self.min_force = min_force
        self.max_force = max_force

    @staticmethod
    def _forces_from_item(item) -> Optional[torch.Tensor]:
        if isinstance(item, Atoms):
            try:
                return torch.as_tensor(item.get_forces())
            except Exception:
                return None
        if isinstance(item, dict):
            return item.get(properties.forces) or item.get("forces")
        try:
            return item[properties.forces]
        except Exception:
            return None

    def _in_range(self, max_force: float) -> bool:
        if self.min_force is not None and max_force < self.min_force:
            return False
        if self.max_force is not None and max_force > self.max_force:
            return False
        return True

    def filter_dataset(self, dataset, label: Optional[str] = None):
        base = dataset.dataset if isinstance(dataset, DataLoader) else dataset
        db = getattr(base, "db", None)
        source = db if db is not None else base
        if not hasattr(source, "__len__"):
            return dataset
        indices: List[int] = []
        for i in range(len(source)):
            item = source[i] if db is not None else base[i]
            forces = self._forces_from_item(item)
            if forces is None:
                indices.append(i)
                continue
            if not isinstance(forces, torch.Tensor):
                forces = torch.as_tensor(forces)
            max_force = torch.linalg.norm(forces, dim=-1).max().item()
            if self._in_range(max_force):
                indices.append(i)
        if label is not None and hasattr(source, "__len__"):
            logger.info("Filtered %s dataset: %d -> %d", label, len(source), len(indices))
        return torch.utils.data.Subset(base, indices)


class UncertaintyFilter(Filter):
    pass
