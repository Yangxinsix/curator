import pytorch_lightning as pl
import torch
from typing import Union, Optional, List, Dict, Callable, Tuple
from ._transform import Transform
from ._neighborlist import NeighborListTransform, TorchNeighborList
from .dataset import collate_atomsdata, AseDataset
import math
from . import properties
from ase.data import chemical_symbols, atomic_numbers
import json
import logging


logger = logging.getLogger(__name__)

class AtomsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_type: str = 'AseDataset',         # select from ['AseDataset', 'NumpyDataset', 'BambooDataset']
        datapath: Union[List[str], str, None] = None,
        train_path: Union[List[str], str, None] = None,
        val_path: Union[List[str], str, None] = None,
        test_path: Union[List[str], str, None] = None,
        cutoff: Optional[float] = None,
        compute_neighbor_list: bool = True,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: List[Transform] = [],
        collate_fn: Callable = collate_atomsdata,
        split_file: Optional[str] = None,
        num_train: Union[int, float, None] = None,
        num_val: Union[int, float, None] = 0.1,
        num_test: Union[int, float, None] = None,
        val_only: bool = True,
        train_val_split: str = "random",  # could be random or sequential
        shuffle: bool = True,
        num_workers: int = 1,
        pin_memory: bool = True,
        species: Union[List[str], str, None] = "auto",
        avg_num_neighbors: Union[float, str, None] = "auto",
        atomic_energies: Union[Dict[int, float], Dict[str, float], None, str] = "auto",
        normalization: bool = True,
        atomwise_normalization: bool = True,
        scale_forces: bool = False,             # scale forces by forces_rms
        default_dtype: torch.dtype = torch.get_default_dtype(),
    ) -> None:
        super().__init__()
        
        self.datapath = datapath
        self.data_type = data_type
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.transforms = transforms
        self.cutoff = cutoff
        self.compute_neighbor_list = compute_neighbor_list
        # batch size parameters
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or test_batch_size or batch_size // 2
        self.test_batch_size = test_batch_size or val_batch_size or batch_size // 2
        self.default_dtype = default_dtype
        
        # splitting parameters
        self.split_file = split_file
        self.num_train = num_train
        if num_val < 1.0 and val_only:
            self.num_train = 1 - num_val
        self.num_val = num_val
        self.num_test = num_test
        self.datalen = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.train_val_split = train_val_split
        self.shuffle = shuffle
        
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._collate_fn = collate_fn
        
        # dataset and dataloaders
        self.dataset = None
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None
        
        # data used for constructing model
        self.normalization = normalization
        self.atomwise_normalization = atomwise_normalization
        self.scale_forces = scale_forces
        self.species = species
        self.avg_num_neighbors = avg_num_neighbors
        self.atomic_energies = atomic_energies
        self.mean = None
        self.std = None
        
    def setup(self, stage: Optional[str] = None) -> None:
        if self._train_dataset is None:
            # differentiate datasets
            # if separate data files are provided
            if self.train_path is not None:
                assert self.datapath is None, "Datapath should be None if train_path is provided."
                self._train_dataset = self.setup_dataset(self.data_type, self.train_path)
                self.num_train = len(self._train_dataset)
            if self.val_path is not None:
                self._val_dataset = self.setup_dataset(self.data_type, self.val_path)
                self.num_val = len(self._val_dataset)
            if self.test_path is not None:
                self._test_dataset = self.setup_dataset(self.data_type, self.test_path)
                self.num_test = len(self._test_dataset)
            else:
                self.num_test = 0
                
            if self.datapath is not None:
                self.dataset = self.setup_dataset(self.data_type, self.datapath)
                self.datalen = len(self.dataset)
                
                if self.train_idx is None:
                    # get number of train, validation, and test points
                    if self.num_train is not None and self.num_train < 1.0:
                        self.num_train = int(math.floor(self.num_train * self.datalen))
                    if self.num_val is not None and self.num_val < 1.0:
                        self.num_val = int(math.floor(self.num_val * self.datalen))
                    if self.num_test is not None:
                        if self.num_test < 1.0:
                            self.num_test = int(math.floor(self.num_test * self.datalen))
                    else:
                        self.num_test = 0
                        
                    assert self.num_train + self.num_val + self.num_test <= self.datalen, f"Number of train, validation, and test points exceed the total number of dataset."
                    self._split_data()
                    
                if self.train_idx is not None and self._train_dataset is None:
                    self._train_dataset = torch.utils.data.Subset(self.dataset, self.train_idx)
                if self.val_idx is not None and self._val_dataset is None:
                    self._val_dataset = torch.utils.data.Subset(self.dataset, self.val_idx)
                if self.test_idx is not None and self._test_dataset is None:
                    self._test_dataset = torch.utils.data.Subset(self.dataset, self.test_idx)
            
            logger.info(self)
    
    def setup_dataset(self, data_type: str, datapath: str) -> None:
        if data_type == 'AseDataset':
            dataset = AseDataset(
                datapath,
                cutoff=self.cutoff,
                compute_neighbor_list=self.compute_neighbor_list,
                transforms=self.transforms,
            )
        elif data_type == 'NumpyDataset':
            from .dataset import NumpyDataset
            dataset = NumpyDataset(
                datapath,
                cutoff=self.cutoff,
                compute_neighbor_list=self.compute_neighbor_list,
                transforms=self.transforms,
            )
        elif data_type == 'BambooDataset':
            from .dataset import BambooDataset
            dataset = BambooDataset(datapath)
        
        return dataset

    @property
    def train_dataset(self) -> torch.utils.data.Dataset:
        return self._train_dataset

    @property
    def val_dataset(self) -> torch.utils.data.Dataset:
        return self._val_dataset

    @property
    def test_dataset(self) -> torch.utils.data.Dataset:
        return self._test_dataset
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self._train_dataloader is None and self._train_dataset is not None:
            self._train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                collate_fn=self._collate_fn,
                num_workers=self._num_workers,
                shuffle=self.shuffle,
                pin_memory=self._pin_memory,
            )
        return self._train_dataloader
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if self._val_dataloader is None and self._val_dataset is not None:
            self._val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                collate_fn=self._collate_fn,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
            )
        return self._val_dataloader
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        if self._test_dataloader is None and self._test_dataset is not None:
            self._test_dataloader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.test_batch_size,
                collate_fn=self._collate_fn,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
            )
        return self._test_dataloader
    
    def _split_data(self) -> None:
        if self.split_file is None:
            if self.train_val_split == "random":
                indices = torch.randperm(self.datalen)
            elif self.train_val_split == "sequential":
                indices = torch.arange(self.datalen)
            else:
                raise NotImplementedError(
                    f"splitting mode {self.train_val_split} not implemented"
                )
            self.train_idx = indices[:self.num_train]
            self.val_idx = indices[self.num_train:self.num_train+self.num_val]
            if self.num_test != 0:
                self.test_idx = indices[self.num_train+self.num_val:self.num_train+self.num_val+self.num_test]
        else:
            with open(self.split_file, "r") as fp:
                splits = json.load(fp)
            for k, v in splits.items():
                if k == "train":
                    self.train_idx = v
                elif k == "validation":
                    self.val_idx = v
                elif k == "test":
                    self.test_idx = v
            self.num_train = len(self.train_idx) if self.train_idx is not None else self.num_train
            self.num_val = len(self.val_idx) if self.val_idx is not None else self.num_val
            self.num_test = len(self.test_idx) if self.test_idx is not None else self.num_test
            
        self._train_dataset = self._train_dataset or torch.utils.data.Subset(self.dataset, self.train_idx)
        self._val_dataset = self._val_dataset or torch.utils.data.Subset(self.dataset, self.val_idx)
        self._test_dataset = self._test_dataset or torch.utils.data.Subset(self.dataset, self.test_idx) if self.num_test != 0 else None

    def _get_species(self) -> Optional[List[str]]:
        if self.species == "auto":
            numbers = []
            for sample in self.train_dataset:
                numbers.append(torch.unique(sample[properties.Z]))
            numbers = torch.unique(torch.cat(numbers))
            self.species = [chemical_symbols[int(n)] for n in numbers]
        logger.debug(f"Training model for elements: {self.species}.")
        return self.species
            
    def _get_avg_num_neighbors(self) -> Optional[float]:
        if self.avg_num_neighbors == "auto":
            n_atoms = 0
            n_neighbors = 0
            for sample in self.train_dataset:
                n_atoms += sample[properties.n_atoms].sum()
                # TODO: add compute_neighbor_list here if neighbors are not computed
                if not self.compute_neighbor_list and not any(isinstance(t, NeighborListTransform) for t in self.transforms):
                    torch_nl = TorchNeighborList(cutoff=self.cutoff, wrap_atoms=True, requires_grad=False, return_distance=False)
                    sample = torch_nl(sample)
                n_neighbors += sample[properties.n_pairs].sum()
            self.avg_num_neighbors = n_neighbors.sum() / n_atoms.item()
            logger.debug(f"The average number of neighbors is calculated to be: {self.avg_num_neighbors:.3f}")
        return self.avg_num_neighbors
    
    def _get_average_E0(self) -> Optional[Dict[int, float]]:
        """
        Function to compute the average interaction energy of each chemical element
        returns dictionary of E0s
        """
        if self.atomic_energies == "auto":
            numbers = [atomic_numbers[s] for s in self._get_species()]
            num_elements = len(numbers)
            len_train = len(self.train_dataset)
            A = torch.zeros((len_train, num_elements))
            B = torch.zeros((len_train,))
            
            for i, sample in enumerate(self.train_dataset):
                B[i] = sample[properties.energy]
                for j, z in enumerate(numbers):
                    A[i, j] = torch.count_nonzero(sample[properties.Z] == z)
            atomic_energies_dict = {z: 0.0 for z in numbers}
            
            try:
                E0s = torch.linalg.lstsq(A, B, rcond=None)[0]
                for i, z in enumerate(numbers):
                    atomic_energies_dict[z] = E0s[i].item()
                    
            except torch.linalg.LinAlgError:
                print(
                    "Failed to compute E0s using least squares regression, using the same for all atoms"
                )
            self.atomic_energies = atomic_energies_dict
            
        if isinstance(self.atomic_energies, Dict):
            for k in list(self.atomic_energies.keys()):
                if isinstance(k, int):
                    self.atomic_energies[chemical_symbols[k]] = self.atomic_energies.pop(k)

        logger.debug(f"Using reference energies for elements: {self.atomic_energies}.")
        return self.atomic_energies
    
    def _get_scale_shift(
        self,
    ) -> Tuple[float, float]:
        if self.mean is None:
            if self.normalization:
                atomic_energies = self._get_average_E0()
                if atomic_energies is not None:
                    reference_energies = torch.zeros((119,), dtype=self.default_dtype)
                    for k, v in atomic_energies.items():
                        reference_energies[atomic_numbers[k] if isinstance(k, str) else k] = v

                energies = []
                if self.scale_forces:
                    forces = []
                for sample in self.train_dataset:
                    e = sample[properties.energy]
                    if atomic_energies is not None:
                        node_e0 = reference_energies[sample[properties.Z]]
                        e0 = node_e0.sum()
                        e = e - e0
                    if self.atomwise_normalization:
                        e /= sample[properties.n_atoms]
                    energies.append(e)
                    if self.scale_forces:
                        forces.append(sample[properties.forces])
                energies = torch.cat(energies)
                mean = torch.mean(energies).item()
                std = torch.std(energies).item()
                if self.scale_forces:
                    forces = torch.cat(forces)
                    std = torch.sqrt(torch.mean(forces * forces)).item()
                    logger.debug(f"Forces will be scaled by forces_rms: {std:.3f}.")
            else:
                mean, std = 0.0, 1.0
            self.mean, self.std = mean, std
        if self.atomwise_normalization:
            logger.debug("Atomwise model outputs will be normalized.")
        logger.debug(f"Model output properties will be scaled by {self.std:.3f}, shifted by {self.mean:.3f}.")
        return self.mean, self.std
    
    def __repr__(self):
        path_info = [f"train_path={self.train_path}, " if self.train_path is not None else "", f"val_path={self.val_path}" if self.val_path is not None else "", f", test_path={self.test_path}" if self.test_path is not None else ""]
        path_info = "".join(path_info)
        path_info += "\n" if path_info != "" else ""
        path_info += f"datapath={self.datapath}\n" if self.datapath is not None else ""
        data_info = f"Dataset size={self.datalen or self.num_train + self.num_val + self.num_test}, training dataset size={self.num_train}, validation dataset size={self.num_val}, test dataset size={self.num_test}.\n"
        scale_info = f"scale={self.std}, shift={self.mean}, atomwise_normalization={self.atomwise_normalization}, scale_forces={self.scale_forces}\n" if self.normalization else ""
        species_info = f"species={self.species}\n" if self.species is not None else ""
        e0_info = f"atomic_energies={self.atomic_energies}" if isinstance(self.atomic_energies, Dict) else ""
        neigh_info = f"avg_num_neighbors={self.avg_num_neighbors}\n" if self.avg_num_neighbors is not None else ""
        return f"{self.__class__.__name__}(" + path_info + data_info + scale_info + species_info + e0_info + neigh_info + ")"