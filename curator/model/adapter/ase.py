from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn

from curator.data import properties
from curator.model.utils import batch_to_atoms

from .utils import build_representation


class ASECalculatorAdapter(nn.Module):
    def __init__(
        self,
        calculator,
        *,
        cutoff: Optional[float] = None,
        model_outputs: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        self.calculator = calculator
        self.model_outputs = list(model_outputs or (properties.energy, properties.forces, properties.stress))
        self.representation = build_representation(float(cutoff or 0.0))

    def forward(self, data: properties.Type) -> properties.Type:
        atoms_list = batch_to_atoms(data)
        device = data[properties.positions].device
        dtype = data[properties.positions].dtype
        energies = []
        forces = []
        stresses = []
        for atoms in atoms_list:
            atoms.calc = self.calculator
            if properties.energy in self.model_outputs:
                energies.append(float(atoms.get_potential_energy()))
            if properties.forces in self.model_outputs:
                forces.append(torch.as_tensor(atoms.get_forces(), dtype=dtype, device=device))
            if properties.stress in self.model_outputs:
                try:
                    stresses.append(torch.as_tensor(atoms.get_stress(), dtype=dtype, device=device))
                except Exception:
                    pass
        if energies:
            data[properties.energy] = torch.as_tensor(energies, dtype=dtype, device=device)
        if forces:
            data[properties.forces] = torch.cat(forces, dim=0)
        if stresses and len(stresses) == len(atoms_list):
            data[properties.stress] = torch.stack(stresses, dim=0)
        return data


__all__ = ["ASECalculatorAdapter"]
