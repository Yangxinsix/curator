import pytorch_lightning as pl
import torch
from typing import Union, Optional, List, Dict, Callable, Tuple, Literal
from dataclasses import dataclass, field
from ._transform import Transform
from ._neighborlist import NeighborListTransform, TorchNeighborList
from .dataset import collate_atomsdata, AseDataset
from .atoms_data import get_sample_atoms, get_sample_target
import math
from . import properties
from .properties import resolve_heads, normalize_head_flag
from ase.data import chemical_symbols, atomic_numbers
import json
import logging
from torch.utils.data import ConcatDataset, Dataset
from pytorch_lightning.utilities.combined_loader import CombinedLoader


logger = logging.getLogger(__name__)

DataTypeName = Literal["Ase", "Numpy", "Bamboo", "Sqlite3"]
SplitMode = Literal["random", "sequential"]

def _is_replay_name(name: str) -> bool:
    return str(name).lower().startswith("replay")


@dataclass
class DataContext:
    species: List[str] = field(default_factory=list)
    avg_num_neighbors: Optional[float] = None
    head_scale_shift: Dict[str, Dict[str, float]] = field(default_factory=dict)  # {head: {"mean": m, "std": s}}
    head_species_shift: Dict[str, Dict[int, float]] = field(default_factory=dict)  # {head: {Z: shift}}

class _DomainTaggedDataset(Dataset):
    def __init__(self, dataset: Dataset, domain_id: int, task: Optional[str] = None, weight: float = 1.0):
        self.dataset = dataset
        self.domain_id = int(domain_id)
        self.task = task
        self.weight = float(weight)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if isinstance(sample, dict):
            sample = sample.copy()
        sample[properties.domain] = torch.tensor([self.domain_id], dtype=torch.long)
        if self.task is not None:
            sample["task"] = self.task
        elif isinstance(sample, dict):
            sample["task"] = sample.get("task", "default")
        sample["weight"] = self.weight
        return sample

class AtomsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_type: DataTypeName = 'Ase',         # select from ['Ase', 'Numpy', 'Bamboo', 'Sqlite3']
        datapath: Union[List[str], str, None] = None,
        train_path: Union[List[str], str, None] = None,
        val_path: Union[List[str], str, None] = None,
        test_path: Union[List[str], str, None] = None,
        cutoff: Optional[float] = None,        # cutoff must be larger than the largest cutoff that needed in the model (i.e., different modules may need different cutoff)
        compute_neighbor_list: bool = True,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[Transform]] = None,
        collate_fn: Callable = collate_atomsdata,
        split_file: Optional[str] = None,
        num_train: Union[int, float, None] = None,
        num_val: Union[int, float, None] = 0.1,
        num_test: Union[int, float, None] = None,
        val_only: bool = True,
        train_val_split: SplitMode = "random",  # could be random or sequential
        shuffle: bool = True,
        num_workers: int = 1,
        pin_memory: bool = True,
        species: Union[List[str], str, None] = "auto",
        avg_num_neighbors: Union[float, str, None] = "auto",
        atomic_energies: Union[Dict[int, float], Dict[str, float], None, str] = "auto",
        normalization: bool = True,
        atomwise_normalization: bool = True,
        scale_by: Union[float, List[float], None] = None,
        shift_by: Union[float, List[float], None] = None,
        scale_forces: bool = False,
        default_dtype: torch.dtype = torch.get_default_dtype(),
        head_reference_by_species: Optional[Dict[str, Dict[Union[int, str], float]]] = None,
        heads: Optional[List] = None,
        rescale_shift_heads: Optional[List] = None,
    ) -> None:
        super().__init__()

        if isinstance(default_dtype, str):
            try:
                default_dtype = getattr(torch, default_dtype)
            except AttributeError as exc:
                raise ValueError(f"Unknown default_dtype '{default_dtype}'.") from exc

        self.datapath = datapath
        self.data_type = data_type
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.transforms = transforms if transforms is not None else []
        self.cutoff = cutoff
        self.compute_neighbor_list = compute_neighbor_list
        # batch size parameters
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or test_batch_size or batch_size // 2
        self.test_batch_size = test_batch_size or val_batch_size or batch_size // 2
        self.default_dtype = default_dtype
        if heads is None or (isinstance(heads, str) and heads.lower() == "auto"):
            heads = ["energy"]
        self.heads = heads
        self.rescale_shift_heads = rescale_shift_heads or []
        
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
        try:
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False
        self._pin_memory = pin_memory and cuda_available
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
        self.species = species
        self.avg_num_neighbors = avg_num_neighbors
        self.atomic_energies = atomic_energies
        self.mean = shift_by
        self.std = scale_by
        self.scale_forces = scale_forces
        self.head_reference_by_species = head_reference_by_species or {}
        self._scale_shift_cache: Dict[str, Tuple[float, float]] = {}
        self._species_logged = False
        
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
            
    
    def setup_dataset(self, data_type: DataTypeName, datapath: str) -> None:
        task = data_type.lower()
        if data_type == 'Ase':
            dataset = AseDataset(
                datapath,
                cutoff=self.cutoff,
                compute_neighbor_list=self.compute_neighbor_list,
                transforms=self.transforms,
                default_dtype=self.default_dtype,
                task=task,
            )
        elif data_type == 'Numpy':
            from .dataset import NumpyDataset
            dataset = NumpyDataset(
                datapath,
                cutoff=self.cutoff,
                compute_neighbor_list=self.compute_neighbor_list,
                transforms=self.transforms,
                default_dtype=self.default_dtype,
                task=task,
            )
        elif data_type == 'Bamboo':
            from .dataset import BambooDataset
            dataset = BambooDataset(datapath, task=task)
        elif data_type == 'Sqlite3':
            from .sql_database import Sqlite3Dataset
            dataset = Sqlite3Dataset(
                datapath,
                cutoff=self.cutoff,
                compute_neighbor_list=self.compute_neighbor_list,
                transforms=self.transforms,
                default_dtype=self.default_dtype,
                task=task,
            )
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

    def _get_species(self, force_process=False) -> Optional[List[str]]:
        if self.species == "auto" or force_process:
            numbers = []
            for sample in self.train_dataset:
                atoms = get_sample_atoms(sample)
                numbers.append(torch.unique(atoms[properties.Z]))
            numbers = torch.unique(torch.cat(numbers))
            self.species = [chemical_symbols[int(n)] for n in numbers]
            if not self._species_logged:
                logger.debug(f"Detected training species: {self.species}.")
                self._species_logged = True
        return self.species
            
    def _get_avg_num_neighbors(self) -> Optional[float]:
        if self.avg_num_neighbors == "auto":
            n_atoms = 0
            n_neighbors = 0
            for sample in self.train_dataset:
                atoms = get_sample_atoms(sample)
                n_atoms += atoms[properties.n_atoms].sum()
                # TODO: add compute_neighbor_list here if neighbors are not computed
                if not self.compute_neighbor_list and not any(isinstance(t, NeighborListTransform) for t in self.transforms):
                    torch_nl = TorchNeighborList(cutoff=self.cutoff, wrap_atoms=True, requires_grad=False, return_distance=False)
                    atoms = torch_nl(atoms)
                n_neighbors += atoms[properties.n_pairs].sum()
            self.avg_num_neighbors = n_neighbors.sum() / n_atoms.item()
            logger.debug(f"The average number of neighbors is calculated to be: {self.avg_num_neighbors:.3f}")
        return self.avg_num_neighbors
    
    def _get_head_per_species_shift(
        self,
        property_key: str,
        is_atomwise: Optional[bool] = None,
    ) -> Optional[Dict[int, float]]:
        """
        Compute per-species shift for a property (dict of atomic_number -> value).

        - For structure-level additive properties, solve a least-squares system
          on atom counts (energy-style).
        - For atomwise properties, average per-atom values per species.

        Returns None when the property is missing or cannot be reduced to a
        scalar per species.
        """
        numbers = [atomic_numbers[s] for s in self._get_species(force_process=True)]
        if len(numbers) == 0:
            return None

        def _normalize_keys(d: Dict[int, float]) -> Dict[int, float]:
            out = {}
            for k, v in d.items():
                idx = atomic_numbers[k] if isinstance(k, str) else k
                out[idx] = v
            return out

        # user-provided override per head
        if property_key in self.head_reference_by_species:
            shifts = _normalize_keys(self.head_reference_by_species[property_key])
            logger.debug(f"Using user-specified per-species reference for '{property_key}': {shifts}.")
            return shifts

        # user-provided atomic_energies acts as a default per-species shift for energy
        if property_key == properties.energy and isinstance(self.atomic_energies, Dict):
            return _normalize_keys(self.atomic_energies)

        # auto-compute only when explicitly requested by the caller
        if property_key == properties.energy and self.atomic_energies == "auto":
            is_atomwise = False if is_atomwise is None else is_atomwise
        if is_atomwise is None:
            is_atomwise = False

        if is_atomwise:
            sums = torch.zeros(len(numbers), dtype=self.default_dtype)
            counts = torch.zeros(len(numbers), dtype=torch.float)
            for sample in self.train_dataset:
                atoms = get_sample_atoms(sample)
                values = get_sample_target(sample, property_key)
                if values is None or properties.Z not in atoms:
                    continue
                if values.ndim > 1:
                    # Only scalar atomwise shifts are supported
                    logger.debug(f"Skip per-species shift for '{property_key}' with ndim={values.ndim}.")
                    return None
                z = atoms[properties.Z]
                for idx, z_val in enumerate(numbers):
                    mask = z == z_val
                    if mask.any():
                        sums[idx] += values[mask].sum()
                        counts[idx] += mask.sum()
            shifts = {}
            for idx, z_val in enumerate(numbers):
                if counts[idx] > 0:
                    shifts[z_val] = (sums[idx] / counts[idx]).item()
            if shifts:
                logger.debug(f"Computed per-species reference for atomwise '{property_key}': {shifts}.")
                return shifts
            return None

        # structure-level additive property: solve least squares on counts
        len_train = len(self.train_dataset)
        A = torch.zeros((len_train, len(numbers)), dtype=self.default_dtype)
        B = torch.zeros((len_train,), dtype=self.default_dtype)
        row = 0
        for sample in self.train_dataset:
            atoms = get_sample_atoms(sample)
            value = get_sample_target(sample, property_key)
            if value is None or properties.Z not in atoms:
                continue
            if value.ndim > 0:
                logger.debug(f"Skip per-species shift for '{property_key}' with shape {tuple(value.shape)}.")
                continue
            B[row] = value
            for j, z in enumerate(numbers):
                A[row, j] = torch.count_nonzero(atoms[properties.Z] == z)
            row += 1

        if row == 0:
            return None
        A = A[:row]
        B = B[:row]
        try:
            coeffs = torch.linalg.lstsq(A, B, rcond=None)[0]
        except torch.linalg.LinAlgError:
            logger.warning(f"Failed to compute per-species shift for '{property_key}' via lstsq; using zeros.")
            coeffs = torch.zeros_like(A[0])
        shifts = {z: coeffs[i].item() for i, z in enumerate(numbers)}
        logger.debug(f"Computed per-species reference for '{property_key}': {shifts}.")
        return shifts
    
    def _get_scale_shift(
        self,
        property_key: str = properties.energy,
        atomwise_normalization: Optional[bool] = None,
        per_species_shift: Optional[Dict[int, float]] = None,
    ) -> Tuple[float, float]:
        """
        Compute (shift, scale) statistics for a given property key. Generic: no special-casing beyond optional per-species shift.
        """
        if property_key in self._scale_shift_cache:
            return self._scale_shift_cache[property_key]

        if not self.normalization:
            self._scale_shift_cache[property_key] = (0.0, 1.0)
            return self._scale_shift_cache[property_key]

        use_atomwise_norm = self.atomwise_normalization if atomwise_normalization is None else atomwise_normalization

        # optional per-species shift subtraction if available (must be provided explicitly)
        per_species_tensor = None
        if per_species_shift is not None:
            per_species_tensor = torch.zeros((119,), dtype=self.default_dtype)
            for k, v in per_species_shift.items():
                per_species_tensor[atomic_numbers[k] if isinstance(k, str) else k] = v

        values = []
        for sample in self.train_dataset:
            atoms = get_sample_atoms(sample)
            v = get_sample_target(sample, property_key)
            if v is None:
                continue
            if per_species_tensor is not None and properties.Z in atoms:
                node_shift = per_species_tensor[atoms[properties.Z]]
                if v.shape[:1] == node_shift.shape[:1]:
                    v = v - node_shift
                else:
                    v = v - node_shift.sum()
            if use_atomwise_norm and properties.n_atoms in atoms and ((not isinstance(v, torch.Tensor)) or v.numel() == 1):
                v = v / atoms[properties.n_atoms]
            values.append(v)

        if len(values) == 0:
            raise KeyError(f"Property '{property_key}' not found in training dataset samples.")

        vals = torch.cat(values) if values[0].numel() > 1 else torch.stack(values).reshape(-1)
        mean = torch.mean(vals).item()
        std = torch.std(vals).item()
        if property_key == properties.energy and self.scale_forces:
            std = self._get_rms(property_key=properties.forces)
            logger.debug(f"Energy scale will use forces RMS: {std:.3f}.")

        self._scale_shift_cache[property_key] = (mean, std if std != 0.0 else 1.0)
        msg = (
            f"Computed normalization stats for '{property_key}': "
            f"shift={self._scale_shift_cache[property_key][0]:.3f}, "
            f"scale={self._scale_shift_cache[property_key][1]:.3f}"
        )
        if per_species_shift is not None:
            msg += " (using per-species reference for computation)."
        logger.debug(msg)
        return self._scale_shift_cache[property_key]

    def _get_rms(self, property_key: str = properties.forces) -> float:
        """
        Compute RMS for a given property key.
        """
        values = []
        for sample in self.train_dataset:
            v = get_sample_target(sample, property_key)
            if v is not None:
                values.append(v)
        if not values:
            raise KeyError(f"Property '{property_key}' not found in training dataset samples.")
        values = torch.cat(values)
        rms = torch.sqrt(torch.mean(values * values)).item()
        logger.debug(f"Computed root mean square for '{property_key}': {rms:.3f}.")
        return rms

    def build_context(self, heads: List) -> "DataContext":
        # heads may be HeadConfig or dict-like with key field
        ctx = DataContext()
        ctx.species = self._get_species() or []
        ctx.avg_num_neighbors = self._get_avg_num_neighbors()

        resolved = resolve_heads(heads)
        for h in resolved:
            key = h.key
            domains = getattr(h, "domains", None)
            if domains is not None and "default" not in domains and len(domains) > 0:
                # if head is restricted to specific domains and this datamodule is not named, still compute stats;
                # filtering per-domain happens when contexts are consumed.
                pass
            atomwise_norm = h.atomwise_normalization

            # determine per-species shift request
            per_species_cfg = h.per_species_shift
            is_atomwise = h.is_atomwise
            per_species_vals = None
            if isinstance(per_species_cfg, dict):
                per_species_vals = {atomic_numbers[k] if isinstance(k, str) else k: v for k, v in per_species_cfg.items()}
            elif isinstance(per_species_cfg, str) and per_species_cfg == "auto":
                try:
                    per_species_vals = self._get_head_per_species_shift(key, is_atomwise=is_atomwise)
                except Exception:
                    per_species_vals = None
            elif per_species_cfg is None:
                # fall back to datamodule-level manual config
                try:
                    per_species_vals = self.head_reference_by_species.get(key, None)
                    if per_species_vals is not None:
                        per_species_vals = {atomic_numbers[k] if isinstance(k, str) else k: v for k, v in per_species_vals.items()}
                except Exception:
                    per_species_vals = None

            scale_mode = normalize_head_flag(h.scale_by)
            shift_mode = normalize_head_flag(h.shift_by)
            mean = None
            std = None
            if scale_mode == "default" or shift_mode == "default":
                try:
                    mean, std = self._get_scale_shift(
                        property_key=key,
                        atomwise_normalization=atomwise_norm,
                        per_species_shift=per_species_vals,
                    )
                except Exception:
                    mean, std = None, None

            if scale_mode == "rms":
                try:
                    std = self._get_rms(property_key=key)
                except Exception:
                    pass
            elif isinstance(scale_mode, (int, float)) and not isinstance(scale_mode, bool):
                std = float(scale_mode)

            if isinstance(shift_mode, (int, float)) and not isinstance(shift_mode, bool):
                mean = float(shift_mode)

            if scale_mode is None or std is None:
                std = 1.0
            if shift_mode is None or mean is None:
                mean = 0.0

            ctx.head_scale_shift[key] = {"mean": float(mean), "std": float(std)}

            if per_species_vals is not None:
                ctx.head_species_shift[key] = per_species_vals
        return ctx
    
    def __repr__(self):
        return self._format_table()

    def __str__(self):
        # Avoid Lightning's default __str__ which inspects dataloaders.
        return self.__repr__()

    def log_summary(self) -> str:
        heads = self._heads_summary()
        return self._format_table(heads=heads)

    def _heads_summary(self) -> str:
        heads = list(self.heads) if self.heads is not None else ["energy"]
        if isinstance(heads, str):
            heads = [heads]
        if self.rescale_shift_heads:
            for h in self.rescale_shift_heads:
                if h not in heads:
                    heads.append(h)

        ctx = None
        try:
            resolved = resolve_heads(heads)
            ctx = self.build_context(resolved)
        except Exception:
            ctx = None

        head_parts = []
        for h in resolve_heads(heads):
            stats = ctx.head_scale_shift.get(h.key, {}) if ctx is not None else {}
            shift = stats.get("mean")
            scale = stats.get("std")
            shift_str = f"{shift:.3f}" if shift is not None else "None"
            scale_str = f"{scale:.3f}" if scale is not None else "None"
            head_parts.append(f"{h.key}(shift={shift_str},scale={scale_str})")
        return ", ".join(head_parts)

    def _format_table(self, heads: Optional[str] = None) -> str:
        train_size = len(self._train_dataset) if self._train_dataset is not None else (self.num_train if isinstance(self.num_train, (int, float)) else 0)
        val_size = len(self._val_dataset) if self._val_dataset is not None else (self.num_val if isinstance(self.num_val, (int, float)) else 0)
        test_size = len(self._test_dataset) if self._test_dataset is not None else (self.num_test if isinstance(self.num_test, (int, float)) else 0)
        path = self.datapath or self.train_path or self.val_path or self.test_path or "N/A"

        headers = ["Train", "Val", "Test", "Batch", "Cutoff", "Path"]
        row = [str(train_size), str(val_size), str(test_size), str(self.batch_size), str(self.cutoff), str(path)]
        domain_name = getattr(self, "domain_name", None)
        if domain_name is not None:
            headers = ["Domain"] + headers
            row = [str(domain_name)] + row
        if heads is not None:
            headers = headers + ["Heads"]
            row = row + [heads]

        fixed = {
            "Domain": 12,
            "Train": 8,
            "Val": 8,
            "Test": 8,
            "Batch": 8,
            "Cutoff": 8,
            "Path": 48,
            "Heads": 60,
        }
        widths = [
            max(fixed.get(headers[i], 0), len(headers[i]), len(row[i]))
            for i in range(len(headers))
        ]

        def fmt(values):
            return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

        line = "-+-".join("-" * w for w in widths)
        table = "\n".join([fmt(headers), line, fmt(row)])
        return f"{self.__class__.__name__}(\n{table}\n)"


def build_datamodule(datapath=None, domain_weights: Optional[Dict[str, float]] = None, **kwargs):
    """
    Factory that returns a single AtomsDataModule or a MultiDomainDataModule
    when ``datapath`` is provided as a dict. Domain entries can override any
    AtomsDataModule argument; unspecified fields fall back to the shared kwargs.
    """
    if isinstance(datapath, dict):
        domain_modules = {}
        collected_weights = {}
        shared_kwargs = {k: v for k, v in kwargs.items() if k != "datapath"}
        items = list(datapath.items())
        replay_items = [(name, cfg) for name, cfg in items if _is_replay_name(name)]
        other_items = [(name, cfg) for name, cfg in items if not _is_replay_name(name)]
        for domain, domain_cfg in replay_items + other_items:
            domain_kwargs = dict(shared_kwargs)
            if isinstance(domain_cfg, dict):
                if "weight" in domain_cfg:
                    collected_weights[domain] = float(domain_cfg["weight"])
                domain_kwargs.update({k: v for k, v in domain_cfg.items() if k not in ("datapath",)})
                domain_kwargs["datapath"] = domain_cfg.get("datapath", None)
            else:
                domain_kwargs["datapath"] = domain_cfg
            domain_modules[domain] = AtomsDataModule(**domain_kwargs)
        merged_weights = dict(domain_weights or {})
        merged_weights.update(collected_weights)
        return MultiDomainDataModule(domain_modules, domain_weights=merged_weights)
    return AtomsDataModule(datapath=datapath, **kwargs)

class DataModuleFactory:
    """
    Thin wrapper to let Hydra instantiate a factory. Hydra expects `_target_`
    to resolve to a class; overriding `__new__` returns the actual datamodule.
    """

    def __new__(cls, datapath=None, **kwargs):
        return build_datamodule(datapath=datapath, **kwargs)

class MultiDomainDataModule(pl.LightningDataModule):
    """
    Utility wrapper that combines train datasets across domains while keeping
    domain-specific validation and test loaders.
    """

    def __init__(
        self,
        domain_modules: Dict[str, AtomsDataModule],
        domain_weights: Optional[Dict[str, float]] = None,
        train_batch_size: Optional[int] = None,
        train_num_workers: Optional[int] = None,
        train_shuffle: Optional[bool] = None,
        train_pin_memory: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.domain_modules = domain_modules
        self.domain_weights = domain_weights or {}
        self.domain_to_id = {name: i for i, name in enumerate(self.domain_modules.keys())}
        self.id_to_domain = {i: name for name, i in self.domain_to_id.items()}
        self._train_batch_size = train_batch_size
        self._train_num_workers = train_num_workers
        self._train_shuffle = train_shuffle
        self._train_pin_memory = train_pin_memory

        self._train_dataset = None
        self._val_loaders = None
        self._test_loaders = None
        self._setup_logged = False

    def setup(self, stage: Optional[str] = None) -> None:
        for name, dm in self.domain_modules.items():
            if not self._setup_logged:
                logger.debug(f"Building data module for: {name}")
            dm.setup(stage)
            domain_id = self.domain_to_id[name]
            domain_weight = self.domain_weights.get(name, 1.0)
            if dm._train_dataset is not None:
                dm._train_dataset = _DomainTaggedDataset(dm._train_dataset, domain_id, task=name, weight=domain_weight)
            if dm._val_dataset is not None:
                dm._val_dataset = _DomainTaggedDataset(dm._val_dataset, domain_id, task=name, weight=domain_weight)
            if dm._test_dataset is not None:
                dm._test_dataset = _DomainTaggedDataset(dm._test_dataset, domain_id, task=name, weight=domain_weight)
            dm._train_dataloader = None
            dm._val_dataloader = None
            dm._test_dataloader = None

        train_datasets = [dm.train_dataset for dm in self.domain_modules.values() if dm.train_dataset is not None]
        self._train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 0 else None
        self._val_loaders = [dl for dm in self.domain_modules.values() if (dl := dm.val_dataloader()) is not None]
        self._test_loaders = [dl for dm in self.domain_modules.values() if (dl := dm.test_dataloader()) is not None]
        self._setup_logged = True

    def log_summary(self) -> str:
        tables = []
        for name, dm in self.domain_modules.items():
            prev_domain = getattr(dm, "domain_name", None)
            try:
                dm.domain_name = name
                tables.append(dm.log_summary())
            finally:
                if prev_domain is None:
                    try:
                        delattr(dm, "domain_name")
                    except AttributeError:
                        pass
                else:
                    dm.domain_name = prev_domain
        return "\n".join(tables)

    def train_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        loaders = {name: dm.train_dataloader() for name, dm in self.domain_modules.items() if dm.train_dataset is not None}
        if not loaders:
            return None
        return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self):
        return self._val_loaders

    def test_dataloader(self):
        return self._test_loaders

    def __repr__(self):
        domain_info = ", ".join(self.domain_modules.keys())
        n_train = sum(len(dm.train_dataset) for dm in self.domain_modules.values() if dm.train_dataset is not None)
        n_val = [len(dm.val_dataset) for dm in self.domain_modules.values() if dm.val_dataset is not None]
        n_test = [len(dm.test_dataset) for dm in self.domain_modules.values() if dm.test_dataset is not None]
        return f"{self.__class__.__name__}(domains=[{domain_info}], train_size={n_train}, val_sizes={n_val}, test_sizes={n_test})"

    def __str__(self):
        return self.__repr__()

    def build_contexts(self, heads: List) -> Dict[str, DataContext]:
        contexts: Dict[str, DataContext] = {}
        species_union = set()
        avg_neighbors = []
        for name, dm in self.domain_modules.items():
            ctx = dm.build_context(heads)
            contexts[str(self.domain_to_id[name])] = ctx
            species_union.update(ctx.species or [])
            if ctx.avg_num_neighbors is not None:
                avg_neighbors.append(ctx.avg_num_neighbors)
        # global context (union species, mean of avg_neighbors if available)
        global_ctx = DataContext()
        global_ctx.species = sorted(list(species_union))
        if avg_neighbors:
            global_ctx.avg_num_neighbors = max(avg_neighbors)
        contexts["global"] = global_ctx
        return contexts
