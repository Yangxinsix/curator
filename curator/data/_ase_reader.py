import torch
import numpy as np
from typing import Dict, Optional, List
from ase import Atoms
from . import properties
from ._neighborlist import NeighborListTransform, Asap3NeighborList
from ._transform import Transform
import warnings
import sys
try:
    import asap3
except ModuleNotFoundError:
    warnings.warn("Failed to import asap3 module for fast neighborlist")

class AseDataReader:
    """ASE data reader"""
    def __init__(
        self,
        cutoff: Optional[float] = None,
        compute_neighborlist: bool = True,
        transforms: List[Transform] = [],
    )   -> None:
        """ASE data reader

        Args:
            cutoff (Optional[float], optional): Cutoff radius. Defaults to None.
            compute_neighborlist (bool, optional): Compute neighborlist. Defaults to True.
            transforms (List[Transform], optional): Transforms. Defaults to [].
        """ 
        self.cutoff = cutoff
        self.compute_neighborlist = compute_neighborlist
        self.transforms = transforms
        if self.compute_neighborlist:
            assert isinstance(self.cutoff, float), "Cutoff radius must be given when compute the neighbor list"
            if not any([isinstance(t, NeighborListTransform) for t in self.transforms]):
                self.transforms.append(Asap3NeighborList(cutoff=self.cutoff))
        
    def __call__(self, atoms: Atoms) -> Dict[str, torch.tensor]:
        # basic properties
        n_atoms = atoms.get_global_number_of_atoms()
        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()
        
        atoms_data = {
            properties.n_atoms: torch.tensor([n_atoms]),
            properties.Z: torch.tensor(atomic_numbers),
            properties.positions: torch.tensor(positions, dtype=torch.float),
            properties.image_idx: torch.zeros((n_atoms,), dtype=torch.long),                 # used for scatter add
        }
        if atoms.pbc.any():
            cell = atoms.cell[:]
            atoms_data[properties.cell] = torch.tensor(cell, dtype=torch.float)
        
        # transform
        for t in self.transforms:
            atoms_data = t(atoms_data)
        
        try:
            energy = torch.tensor([atoms.get_potential_energy()], dtype=torch.float)
            atoms_data[properties.energy] = energy
        except (AttributeError, RuntimeError):
            pass
        
        try: 
            forces = torch.tensor(atoms.get_forces(apply_constraint=False), dtype=torch.float)
            atoms_data[properties.forces] = forces
        except (AttributeError, RuntimeError):
            pass
        
        try: 
            stress = torch.tensor(atoms.get_stress(apply_constraint=False), dtype=torch.float)
            atoms_data[properties.stress] = stress
        except (AttributeError, RuntimeError):
            pass
        
        return atoms_data