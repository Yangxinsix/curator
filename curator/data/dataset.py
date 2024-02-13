import torch
from ._ase_reader import AseDataReader
from typing import List, Union, Dict
from ase.io import Trajectory
from ase import Atoms
from . import properties
from ._transform import Transform

class AseDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        ase_db, 
        cutoff: float=5.0, 
        compute_neighborlist: bool=True, 
        transforms: List[Transform] = [],
    ) -> None:
        super().__init__()
        
        if isinstance(ase_db, str):
            self.db = Trajectory(ase_db)
        else:
            self.db = ase_db
        
        self.cutoff = cutoff
        self.atoms_reader = AseDataReader(cutoff, compute_neighborlist, transforms)
        
    def __len__(self) -> int:
        return len(self.db)
    
    def __getitem__(self, idx):
        atoms = self.db[idx]
        atoms_data = self.atoms_reader(atoms)
        return atoms_data

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
    edge_offset = torch.zeros_like(collated[properties.n_atoms])
    edge_offset[1:] = collated[properties.n_atoms][:-1]
    edge_offset = torch.cumsum(edge_offset, dim=0)
    edge_offset = torch.repeat_interleave(edge_offset, collated[properties.n_pairs])
    edge_idx = collated[properties.edge_idx] + edge_offset.unsqueeze(-1)
    collated[properties.edge_idx] = edge_idx
    
    return collated