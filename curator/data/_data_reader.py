import torch
import numpy as np
from typing import Dict, Optional, List
from ase import Atoms
from ase.parallel import world
from ase.io.trajectory import (
    TrajectoryReader,
    TrajectoryWriter,
    SlicedTrajectory,
)
from . import properties
from ._neighborlist import NeighborListTransform, Asap3NeighborList
from ._transform import Transform
import warnings
import sys
import abc

try:
    import asap3
except ModuleNotFoundError:
    warnings.warn("Failed to import asap3 module for fast neighborlist")

def Trajectory(filenames, mode='r', atoms=None, properties=None, master=None, comm=world):
    if mode == 'r':
        return CombinedTrajectoryReader(filenames)
    else:
        return CombinedTrajectoryWriter(filenames[0], mode, atoms, properties, master=master, comm=comm)

class CombinedTrajectoryReader(TrajectoryReader):
    def __init__(self, filenames):
        if isinstance(filenames, str):
            filenames = [filenames]
        self.readers = [TrajectoryReader(filename) for filename in filenames]

    def __len__(self):
        return sum(len(reader) for reader in self.readers)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return SlicedTrajectory(self, i)
        
        for reader in self.readers:
            if i < len(reader):
                return reader[i]
            i -= len(reader)
        raise IndexError("Index out of range")

    def __iter__(self):
        for reader in self.readers:
            for item in reader:
                yield item

    def __add__(self, other):
        if not isinstance(other, CombinedTrajectoryReader):
            raise TypeError("Operands must be of type CombinedTrajectoryReader")
        combined_filenames = [reader.filename for reader in self.readers] + [reader.filename for reader in other.readers]
        return CombinedTrajectoryReader(combined_filenames)

    def close(self):
        for reader in self.readers:
            reader.close()

class CombinedTrajectoryWriter(TrajectoryWriter):
    def __init__(self, filename, mode='w'):
        super().__init__(filename, mode)

    def __add__(self, other):
        if not isinstance(other, CombinedTrajectoryWriter):
            raise TypeError("Operands must be of type CombinedTrajectoryWriter")
        combined_filename = "combined_output.traj"
        combined_writer = CombinedTrajectoryWriter(combined_filename, 'w')

        # Assuming self.writer and other.writer support reading their contents
        for writer in [self.writer, other.writer]:
            combined_writer.write(writer.read())

        return combined_writer

    def close(self):
        super().close()

class DataReader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self):
        pass     

class AseDataReader(DataReader):
    """ASE data reader"""
    def __init__(
        self,
        cutoff: Optional[float] = None,
        compute_neighbor_list: bool = True,
        transforms: List[Transform] = [],
        default_dtype: torch.dtype = torch.get_default_dtype(),
    )   -> None:
        """ASE data reader

        Args:
            cutoff (Optional[float], optional): Cutoff radius. Defaults to None.
            compute_neighbor_list (bool, optional): Compute neighborlist. Defaults to True.
            transforms (List[Transform], optional): Transforms. Defaults to [].
        """ 
        self.cutoff = cutoff
        self.compute_neighbor_list = compute_neighbor_list
        self.transforms = transforms
        self.default_dtype = default_dtype
        if self.compute_neighbor_list:
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
            properties.positions: torch.tensor(positions, dtype=self.default_dtype),
            properties.image_idx: torch.zeros((n_atoms,), dtype=torch.long),                 # used for scatter add
        }
        if atoms.pbc.any():
            cell = atoms.cell[:]
            atoms_data[properties.cell] = torch.tensor(cell, dtype=self.default_dtype)
        
        # transform
        for t in self.transforms:
            atoms_data = t(atoms_data)
        
        try:
            energy = torch.tensor([atoms.get_potential_energy()], dtype=self.default_dtype)
            atoms_data[properties.energy] = energy
        except (AttributeError, RuntimeError):
            pass
        
        try: 
            forces = torch.tensor(atoms.get_forces(apply_constraint=False), dtype=self.default_dtype)
            atoms_data[properties.forces] = forces
        except (AttributeError, RuntimeError):
            pass
        
        try: 
            stress = torch.tensor(atoms.get_stress(apply_constraint=False), dtype=self.default_dtype)
            atoms_data[properties.stress] = stress
        except (AttributeError, RuntimeError):
            pass
        
        return atoms_data