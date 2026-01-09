from . import properties
from .atoms_data import get_sample_atoms, get_sample_target
from typing import List, Dict, Tuple, Union, Optional, Sequence
from ase.data import atomic_names, atomic_numbers
import torch
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
from ase.io import read, write
from ._data_reader import Trajectory
from ase import Atoms
from pathlib import PosixPath, Path
from ase.io.trajectory import TrajectoryReader, SlicedTrajectory
from omegaconf import ListConfig
import os
import tarfile
import zipfile
import shutil
import gzip
import bz2
import lzma
from urllib.parse import urlparse
from urllib.request import urlopen
import logging
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

logger = logging.getLogger(__name__)

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
    elif isinstance(ase_db, SlicedTrajectory):
        db = ase_db

    return db

def _prepare_data_source(
    url: str,
    cache_dir: Optional[Union[str, PosixPath]] = None,
    required_elements: Optional[Sequence[Union[str, int]]] = None,
    match: str = "all",
    extract: bool = True,
    filename: Optional[str] = None,
    save_filtered: bool = True,
    filtered_filename: Optional[str] = None,
    ase_read_kwargs: Optional[Dict] = None,
) -> List[Atoms]:
    if cache_dir is None:
        cache_path = Path.home() / ".cache_data"
    else:
        cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(url)
    url_name = os.path.basename(parsed.path)
    local_name = filename or url_name or "downloaded_data"
    download_path = cache_path / local_name

    if not download_path.exists():
        cached_matches = sorted(cache_path.glob(local_name))
        if cached_matches:
            download_path = cached_matches[0]
            logger.info("Found cached file %s; skipping download", download_path)
        else:
            logger.info("Downloading data from %s", url)
            with urlopen(url) as response:
                total = int(response.headers.get("Content-Length", 0))
                with open(download_path, "wb") as handle:
                    if tqdm is None:
                        shutil.copyfileobj(response, handle)
                    else:
                        with tqdm(
                            total=total,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=f"Downloading {download_path.name}",
                        ) as progress:
                            while True:
                                chunk = response.read(1024 * 1024)
                                if not chunk:
                                    break
                                handle.write(chunk)
                                progress.update(len(chunk))
            logger.info("Saved data to %s", download_path)
    else:
        logger.info("Using cached file at %s", download_path)

    data_paths = [download_path]
    extracted_root = None
    if extract:
        if zipfile.is_zipfile(download_path):
            extracted_root = cache_path / f"{download_path.stem}_extracted"
            if not extracted_root.exists():
                extracted_root.mkdir(parents=True, exist_ok=True)
                logger.info("Extracting zip archive: %s", download_path)
                with zipfile.ZipFile(download_path, "r") as zf:
                    zf.extractall(extracted_root)
        elif tarfile.is_tarfile(download_path):
            extracted_root = cache_path / f"{download_path.stem}_extracted"
            if not extracted_root.exists():
                extracted_root.mkdir(parents=True, exist_ok=True)
                logger.info("Extracting tar archive: %s", download_path)
                with tarfile.open(download_path, "r:*") as tf:
                    tf.extractall(extracted_root)
        elif download_path.suffix in {".gz", ".bz2", ".xz"}:
            extracted_root = cache_path / f"{download_path.stem}_extracted"
            if not extracted_root.exists():
                extracted_root.mkdir(parents=True, exist_ok=True)
            output_path = extracted_root / download_path.with_suffix("").name
            if not output_path.exists():
                logger.info("Decompressing file: %s", download_path)
                if download_path.suffix == ".gz":
                    opener = gzip.open
                elif download_path.suffix == ".bz2":
                    opener = bz2.open
                else:
                    opener = lzma.open
                with opener(download_path, "rb") as compressed, open(output_path, "wb") as out_handle:
                    shutil.copyfileobj(compressed, out_handle)

    if extracted_root is not None:
        data_paths = sorted([p for p in extracted_root.rglob("*") if p.is_file()])
        logger.info("Found %d files after extraction", len(data_paths))

    ase_read_kwargs = ase_read_kwargs or {}
    atoms_list: List[Atoms] = []
    last_error: Optional[Exception] = None
    logger.info("Reading structures with ASE from %d path(s)", len(data_paths))
    for path in data_paths:
        try:
            read_result = read(str(path), ":", **ase_read_kwargs)
        except Exception as exc:
            last_error = exc
            if "format" in ase_read_kwargs:
                continue
            try:
                read_result = read(str(path), ":", format="extxyz", **ase_read_kwargs)
            except Exception as exc_ext:
                last_error = exc_ext
                if path.suffix == ".xyz":
                    try:
                        from ase.io.extxyz import iread_xyz
                        with open(path, "r") as handle:
                            for atoms in iread_xyz(handle, index=":"):
                                atoms_list.append(atoms)
                        continue
                    except Exception as exc_stream:
                        last_error = exc_stream
                        continue
                continue
        if isinstance(read_result, Atoms):
            atoms_list.append(read_result)
        else:
            atoms_list.extend(read_result)

    if not atoms_list:
        if last_error is not None:
            raise ValueError(
                f"No readable structures found for '{download_path}'. Last error: {last_error}"
            )
        raise ValueError(f"No readable structures found for '{download_path}'.")

    if not required_elements:
        logger.info("No element filter applied; returning %d structures", len(atoms_list))
        return atoms_list

    required_numbers = set()
    for elem in required_elements:
        if isinstance(elem, str):
            if elem.isdigit():
                required_numbers.add(int(elem))
            else:
                required_numbers.add(atomic_numbers[elem])
        else:
            required_numbers.add(int(elem))

    if match not in {"all", "any"}:
        raise ValueError("match must be 'all' or 'any'.")

    logger.info(
        "Filtering structures with elements=%s (match=%s)",
        list(required_elements),
        match,
    )
    filtered = []
    for atoms in atoms_list:
        present = set(atoms.get_atomic_numbers().tolist())
        if match == "all":
            keep = required_numbers.issubset(present)
        else:
            keep = bool(required_numbers & present)
        if keep:
            filtered.append(atoms)
    logger.info("Filtered structures: %d / %d", len(filtered), len(atoms_list))
    if save_filtered:
        if filtered_filename:
            filtered_path = cache_path / filtered_filename
        else:
            filtered_path = cache_path / f"{download_path.stem}_filtered.xyz"
        if filtered:
            write(str(filtered_path), filtered, format="extxyz")
            logger.info("Saved filtered structures to %s", filtered_path)
            print(f"Filtered data saved to: {filtered_path}")
        else:
            logger.info("No filtered structures to save")
    return filtered

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
        sample = dataset[i]
        B[i] = get_sample_target(sample, properties.energy)
        for j, z in enumerate(numbers):
            atoms = get_sample_atoms(sample)
            A[i, j] = torch.count_nonzero(atoms[properties.Z] == z)
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
