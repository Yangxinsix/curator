import torch
from ._data_reader import AseDataReader, Trajectory
from ._neighborlist import NeighborListTransform, Asap3NeighborList
from typing import List, Union, Dict, Optional
from ase.io.trajectory import TrajectoryReader
from ase.io import read
from ase import Atoms
from . import properties
from .atoms_data import AtomsData, atoms_data_from_dict
from .collate_atoms_data import collate_atoms_data
from ._transform import Transform
from .utils import read_trajectory
import numpy as np

class AseDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        ase_db: Union[List[Atoms], TrajectoryReader, str, List[str]], 
        cutoff: float=5.0, 
        compute_neighbor_list: bool=True, 
        transforms: Optional[List[Transform]] = None,
        default_dtype: torch.dtype = torch.get_default_dtype(),
        task: str = "ase",
        weight: float = 1.0,
        meta: Dict = None,
        return_atoms_data: bool = True,
    ) -> None:
        super().__init__()
        
        self.db = read_trajectory(ase_db)
        transforms = [] if transforms is None else transforms

        self.cutoff = cutoff
        self.default_dtype = default_dtype
        self.atoms_reader = AseDataReader(
            cutoff,
            compute_neighbor_list,
            transforms,
            default_dtype=self.default_dtype,
        )
        self.task = task
        self.weight = weight
        self.meta = meta
        self.return_atoms_data = return_atoms_data
        
    def __len__(self) -> int:
        return len(self.db)

    def get_n_atoms(self, idx: int) -> int:
        return len(self.db[idx])
    
    def __getitem__(self, idx):
        atoms = self.db[idx]
        atoms_data = self.atoms_reader(atoms)
        if not self.return_atoms_data:
            return atoms_data
        meta = None if self.meta is None else dict(self.meta)
        if meta is not None:
            meta.setdefault("index", idx)
        return atoms_data_from_dict(atoms_data, task=self.task, weight=self.weight, meta=meta)

    def __del__(self) -> None:
        try:
            if hasattr(self, "db") and hasattr(self.db, "close"):
                self.db.close()
        except Exception:
            pass

class BambooDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, task: str = "bamboo", weight: float = 1.0, meta: Dict = None, return_atoms_data: bool = True):
        # cutoff is 5.0 A
        self.data = torch.load(datapath, map_location='cpu')
        self.task = task
        self.weight = weight
        self.meta = meta
        self.return_atoms_data = return_atoms_data

    def __getitem__(self, index):
        left_a, right_a = self.data['cumsum_atom'][index], self.data['cumsum_atom'][index+1]
        left_e, right_e = self.data['cumsum_edge'][index], self.data['cumsum_edge'][index+1]
        atoms_dict = {
            properties.Z: self.data['atom_types'][left_a:right_a],
            properties.R: self.data['pos'][left_a:right_a],
            properties.n_atoms: (right_a - left_a).unsqueeze(0),
            properties.n_pairs: (right_e - left_e).unsqueeze(0),
            properties.edge_idx: self.data['edge_index'][left_e:right_e] - self.data['cumsum_atom'][index],
            properties.edge_diff: self.data['pos'][self.data['edge_index'][left_e:right_e, 1]] - \
                self.data['pos'][self.data['edge_index'][left_e:right_e, 0]],
            properties.energy: self.data['energy'][index].unsqueeze(0),
            properties.forces: self.data['forces'][left_a:right_a],
            properties.virial: self.data['virial'][index].flatten()[[0, 4, 8, 5, 2, 1]].unsqueeze(0),
        }

        if not self.return_atoms_data:
            return atoms_dict
        meta = None if self.meta is None else dict(self.meta)
        if meta is not None:
            meta.setdefault("index", index)
        return atoms_data_from_dict(atoms_dict, task=self.task, weight=self.weight, meta=meta)

    def get_n_atoms(self, index: int) -> int:
        left_a, right_a = self.data['cumsum_atom'][index], self.data['cumsum_atom'][index+1]
        return int((right_a - left_a).item())

    def to_ase_atoms(self, index):
        sample = self.__getitem__(index)
        if isinstance(sample, AtomsData):
            atoms_dict = sample.to_dict()
        else:
            atoms_dict = sample
        atoms = Atoms(
            symbols=atoms_dict[properties.Z].numpy(),
            positions=atoms_dict[properties.R].numpy(),
        )
        atoms.info['energy'] = float(atoms_dict[properties.energy].numpy())
        atoms.info['forces'] = atoms_dict[properties.forces].numpy()
        atoms.info['virial'] = atoms_dict[properties.virial].numpy()
        atoms.info[properties.edge_idx] = atoms_dict[properties.edge_idx].numpy()
        atoms.info[properties.edge_diff] = atoms_dict[properties.edge_diff].numpy()

        return atoms

    def __len__(self):
        return len(self.data['energy'])

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        datapath, 
        cutoff: float=5.0, 
        compute_neighbor_list: bool=True, 
        transforms: Optional[List[Transform]] = None,
        default_dtype: torch.dtype = torch.get_default_dtype(),
        task: str = "numpy",
        weight: float = 1.0,
        meta: Dict = None,
        return_atoms_data: bool = True,
    ) -> None:
        super().__init__()
        
        self.npdata = np.load(datapath)
        transforms = [] if transforms is None else transforms
        self.cutoff = cutoff
        self.default_dtype = default_dtype
        self.compute_neighbor_list = compute_neighbor_list
        self.transforms = transforms
        self.task = task
        self.weight = weight
        self.meta = meta
        self.return_atoms_data = return_atoms_data
        if self.compute_neighbor_list:
            assert isinstance(self.cutoff, float), "Cutoff radius must be given when compute the neighbor list"
            if not any([isinstance(t, NeighborListTransform) for t in self.transforms]):
                self.transforms.append(Asap3NeighborList(cutoff=self.cutoff))
        
    def __len__(self) -> int:
        return len(self.npdata['E'])

    def get_n_atoms(self, idx: int) -> int:
        return int(len(self.npdata["z"]))
    
    def __getitem__(self, idx):
        atoms_dict = {
            properties.Z: torch.from_numpy(self.npdata["z"]).type(torch.long), 
            properties.R: torch.from_numpy(self.npdata["R"][idx]).type(self.default_dtype),
        }
        n_atoms = len(self.npdata["z"])
        atoms_dict[properties.n_atoms] = torch.tensor([n_atoms], dtype=torch.long)
        atoms_dict[properties.image_idx] = torch.zeros((n_atoms,), dtype=self.default_dtype)

        if "cell" in self.npdata:
            cell = torch.from_numpy(self.npdata["cell"]).type(self.default_dtype)
        
        # transform
        for t in self.transforms:
            atoms_dict = t(atoms_dict)
        
        try:
            atoms_dict[properties.energy] = torch.from_numpy(self.npdata["E"][idx]).type(self.default_dtype)
        except (AttributeError, RuntimeError, KeyError):
            pass
        
        try: 
            atoms_dict[properties.forces] = torch.from_numpy(self.npdata["F"][idx]).type(self.default_dtype)
        except (AttributeError, RuntimeError, KeyError):
            pass
        
        try: 
            atoms_dict[properties.stress] = torch.from_numpy(self.npdata["stress"][idx]).type(self.default_dtype)
        except (AttributeError, RuntimeError, KeyError):
            pass
        
        if not self.return_atoms_data:
            return atoms_dict
        meta = None if self.meta is None else dict(self.meta)
        if meta is not None:
            meta.setdefault("index", idx)
        return atoms_data_from_dict(atoms_dict, task=self.task, weight=self.weight, meta=meta)
        
def cat_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    if tensors[0].shape:
        return torch.cat(tensors)
    return torch.stack(tensors)

def collate_atomsdata(atoms_data: List[dict], pin_memory=False) -> Dict:
    return collate_atoms_data(atoms_data, pin_memory=pin_memory)
