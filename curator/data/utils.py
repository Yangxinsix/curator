from . import properties
from typing import List, Dict, Tuple, Union, Optional
from ase.data import atomic_names, atomic_numbers
import torch
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
from ase.io import read
from ._data_reader import Trajectory
from ase import Atoms
from pathlib import PosixPath
from ase.io.trajectory import TrajectoryReader
from omegaconf import ListConfig
import os

def read_trajectory(ase_db, *args, **kwargs):
    if isinstance(ase_db, (str, PosixPath)):
        ase_db = str(ase_db)  # Convert PosixPath to string if necessary
        if ase_db.endswith('.traj'):
            db = Trajectory(ase_db)
        else:
            db = read(ase_db, ':')
    elif isinstance(ase_db, (list, ListConfig)):
        if all(isinstance(item, Atoms) for item in ase_db):
            db = ase_db
        elif all(isinstance(item, (str, PosixPath)) and str(item).endswith('.traj') for item in ase_db):
            db = Trajectory([str(item) for item in ase_db if os.path.getsize(item)])
        else:
            db = []
            for item in ase_db:
                if isinstance(item, (str, PosixPath)) and os.path.getsize(item):
                    item = str(item)  # Convert PosixPath to string if necessary
                    db += read(item, index=':', *args, **kwargs)
    elif isinstance(ase_db, TrajectoryReader):
        db = ase_db

    return db

def compute_average_E0(
    dataset,
    symbols: Optional[List[str]]=None,
):
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s
    """
    numbers = [atomic_numbers[s] for s in symbols]
    len_train = len(dataset)
    num_elements = len(numbers)
    
    A = torch.zeros((len_train, num_elements))
    B = torch.zeros((len_train,))
    
    for i in range(len_train):
        B[i] = dataset[i][properties.energy]
        for j, z in enumerate(numbers):
            A[i, j] = torch.count_nonzero(dataset[i][properties.Z] == z)
    atomic_energies_dict = {z: 0.0 for z in numbers}
    try:
        E0s = torch.linalg.lstsq(A, B, rcond=None)[0]
        for i, z in enumerate(numbers):
            atomic_energies_dict[z] = E0s[i].item()
            
    except torch.linalg.LinAlgError:
        print(
            "Failed to compute E0s using least squares regression, using the same for all atoms"
        )
    return atomic_energies_dict

def compute_scale_shift(
    dataloader: Union[DataLoader, Dataset], 
    per_atom=True,
    atomic_energies: Optional[Dict[int, float]]=None,
) -> Tuple[float, float]:
    try:
        from torch_scatter import scatter_add
    except ImportError:
        from curator.utils import scatter_add
    reference_energies = torch.zeros((119,), dtype=torch.float)
    if atomic_energies is not None:
        for k, v in atomic_energies.items():
            reference_energies[k] = v

    energies = []
    for batch in enumerate(dataloader):
        node_e0 = reference_energies[batch[properties.Z]]
        e0 = scatter_add(node_e0, batch[properties.image_idx])
        e = batch[properties.energy] - e0
        if per_atom:
            e /= batch[properties.n_atoms]
        energies.append(e)
    energies = torch.cat(energies)
    mean = torch.mean(energies).item()
    std = torch.std(energies).item()
    
    return mean, std

def compute_avg_num_neighbors(dataloader: Union[DataLoader, Dataset]) -> float:
    n_atoms = 0
    n_neighbors = 0
    for batch in dataloader:
        n_atoms += batch[properties.n_atoms].sum()
        # TODO: add compute_neighbor_list here if neighbors are not computed
        n_neighbors += batch[properties.edge_idx].shape[0]
        
    return n_neighbors / n_atoms.item()
    
def split_data(dataset: Dataset, val_ratio: float):
        # Load or generate splits
    datalen = len(dataset)
    num_validation = int(math.ceil(datalen * val_ratio))
    indices = np.random.permutation(len(dataset))
    splits = {
        "train": indices[num_validation:].tolist(),
        "validation": indices[:num_validation].tolist(),
    }

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = torch.utils.data.Subset(dataset, indices)
    return datasplits