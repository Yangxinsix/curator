import torch
from ._data_reader import AseDataReader, Trajectory
from ._neighborlist import NeighborListTransform, Asap3NeighborList
from typing import List, Union, Dict
from ase.io.trajectory import TrajectoryReader
from ase.io import read
from ase import Atoms
from . import properties
from ._transform import Transform
from .utils import read_trajectory
import numpy as np

class AseDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        ase_db: Union[List[Atoms], TrajectoryReader, str, List[str]], 
        cutoff: float=5.0, 
        compute_neighbor_list: bool=True, 
        transforms: List[Transform] = [],
        default_dtype: torch.dtype = torch.get_default_dtype(),
    ) -> None:
        super().__init__()
        
        self.db = read_trajectory(ase_db)

        self.cutoff = cutoff
        self.default_dtype = default_dtype
        self.atoms_reader = AseDataReader(cutoff, compute_neighbor_list, transforms)
        
    def __len__(self) -> int:
        return len(self.db)
    
    def __getitem__(self, idx):
        atoms = self.db[idx]
        atoms_data = self.atoms_reader(atoms)
        return atoms_data

class BambooDataset(torch.utils.data.Dataset):
    def __init__(self, datapath):
        # cutoff is 5.0 A
        self.data = torch.load(datapath, map_location='cpu')

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

        return atoms_dict

    def to_ase_atoms(self, index):
        atoms_dict = self.__getitem__(index)
        atoms = Atoms(
            symbols=atoms_dict[properties.Z].numpy(),
            positions=atoms_dict[properties.R].numpy(),
        )

        return atoms

    def __len__(self):
        return len(self.data['energy'])

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        datapath, 
        cutoff: float=5.0, 
        compute_neighbor_list: bool=True, 
        transforms: List[Transform] = [],
        default_dtype: torch.dtype = torch.get_default_dtype(),
    ) -> None:
        super().__init__()
        
        self.npdata = np.load(datapath)
        self.cutoff = cutoff
        self.default_dtype = default_dtype
        self.compute_neighbor_list = compute_neighbor_list
        self.transforms = transforms
        if self.compute_neighbor_list:
            assert isinstance(self.cutoff, float), "Cutoff radius must be given when compute the neighbor list"
            if not any([isinstance(t, NeighborListTransform) for t in self.transforms]):
                self.transforms.append(Asap3NeighborList(cutoff=self.cutoff))
        
    def __len__(self) -> int:
        return len(self.npdata['E'])
    
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
        
        return atoms_dict
        
def cat_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    if tensors[0].shape:
        return torch.cat(tensors)
    return torch.stack(tensors)

def collate_atomsdata(atoms_data: List[dict], pin_memory=True) -> Dict:
    # convert from list of dicts to dict of lists
    dict_of_lists = {k: [dic[k] for dic in atoms_data] for k in atoms_data[0]}
    if pin_memory:
        pin = lambda x: x.pin_memory()
    else:
        pin = lambda x: x
    
    # concatenate tensors
    collated = {k: cat_tensors(v) for k, v in dict_of_lists.items()}
    
    # create image index for each atom
    image_idx = torch.repeat_interleave(
        torch.arange(len(atoms_data)), collated[properties.n_atoms], dim=0
    )
    collated[properties.image_idx] = image_idx
    
    # shift index of edges (because of batching)
    if properties.edge_idx in collated:
        edge_offset = torch.zeros_like(collated[properties.n_atoms])
        edge_offset[1:] = collated[properties.n_atoms][:-1]
        edge_offset = torch.cumsum(edge_offset, dim=0)
        edge_offset = torch.repeat_interleave(edge_offset, collated[properties.n_pairs])
        edge_idx = collated[properties.edge_idx] + edge_offset.unsqueeze(-1)
        collated[properties.edge_idx] = edge_idx
    
    return collated