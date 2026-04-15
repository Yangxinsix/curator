from . import properties
from .atoms_data import atoms_data_from_dict, get_sample_atoms, get_sample_target
from .collate_atoms_data import collate_atoms_data
from typing import List, Dict, Tuple, Union, Optional, Sequence
import torch
from torch.utils.data import DataLoader, Dataset
import math
import contextlib
import numpy as np
from ase.io import read
from ase.data import atomic_numbers
from ._data_reader import Trajectory, CombinedTrajectoryReader
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
    from tqdm.auto import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None
    logging_redirect_tqdm = None

logger = logging.getLogger(__name__)

def _split_slice_spec(path: str) -> tuple[str, Optional[str]]:
    if "@" not in path:
        return path, None
    base, spec = path.rsplit("@", 1)
    if not base:
        return path, None
    if spec == "":
        spec = None
    return base, spec


def _parse_slice_spec(spec: Optional[str]):
    if spec is None:
        return None
    spec = spec.strip()
    if spec == "":
        return None
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) > 3:
            raise ValueError(f"Invalid slice spec '{spec}'")
        def _to_int(value: str):
            return int(value) if value not in ("", None) else None
        start = _to_int(parts[0])
        stop = _to_int(parts[1]) if len(parts) > 1 else None
        step = _to_int(parts[2]) if len(parts) > 2 else None
        return slice(start, stop, step)
    return int(spec)


def _read_trajectory_path(path: str, *args, **kwargs):
    base, spec = _split_slice_spec(path)
    slice_obj = _parse_slice_spec(spec)
    if base.endswith(".traj"):
        reader = Trajectory(base)
        if slice_obj is None:
            return reader
        sliced = reader[slice_obj]
        if isinstance(sliced, Atoms):
            return [sliced]
        return sliced
    if slice_obj is None:
        if args or "index" in kwargs:
            result = read(base, *args, **kwargs)
        else:
            result = read(base, ":")
    else:
        if "index" in kwargs:
            raise TypeError("index is specified both via @slice syntax and keyword arguments")
        result = read(base, slice_obj, *args, **kwargs)
    if isinstance(result, Atoms):
        return [result]
    return result


def read_trajectory(ase_db, *args, **kwargs):
    if isinstance(ase_db, (str, PosixPath)):
        ase_db = str(ase_db)  # Convert PosixPath to string if necessary
        if ase_db.startswith(("http://", "https://")):
            return _prepare_data_source(ase_db)
        db = _read_trajectory_path(ase_db, *args, **kwargs)
    elif isinstance(ase_db, (list, ListConfig)):
        if all(isinstance(item, Atoms) for item in ase_db):
            db = ase_db
        elif all(isinstance(item, (str, PosixPath)) for item in ase_db):
            readers = []
            atoms_list: List[Atoms] = []
            for item in ase_db:
                item = str(item)
                base, _ = _split_slice_spec(item)
                if os.path.getsize(base) == 0:
                    continue
                entry = _read_trajectory_path(item, *args, **kwargs)
                if isinstance(entry, (TrajectoryReader, SlicedTrajectory)):
                    readers.append(entry)
                else:
                    atoms_list.extend(entry)
            if readers and not atoms_list:
                if len(readers) == 1:
                    db = readers[0]
                else:
                    db = CombinedTrajectoryReader.from_readers(readers)
            else:
                for reader in readers:
                    atoms_list.extend(list(reader))
                db = atoms_list
        else:
            db = []
            for item in ase_db:
                if isinstance(item, (str, PosixPath)):
                    item = str(item)
                    base, _ = _split_slice_spec(item)
                    if os.path.getsize(base):
                        entry = _read_trajectory_path(item, *args, **kwargs)
                        if isinstance(entry, (TrajectoryReader, SlicedTrajectory)):
                            db.extend(list(entry))
                        else:
                            db.extend(entry)
                elif isinstance(item, Atoms):
                    db.append(item)
    elif isinstance(ase_db, TrajectoryReader):
        db = ase_db
    elif isinstance(ase_db, SlicedTrajectory):
        db = ase_db

    return db


def iter_atoms(
    data_source,
    reader=None,
    device=None,
    dtype=None,
    requires_grad: bool = True,
    desc: Optional[str] = None,
    task: str = "ase",
):
    atoms_iter = read_trajectory(data_source)
    total = len(atoms_iter) if hasattr(atoms_iter, "__len__") else None
    iterator = atoms_iter
    log_ctx = logging_redirect_tqdm() if logging_redirect_tqdm is not None else contextlib.nullcontext()
    if tqdm is not None and desc is not None:
        iterator = tqdm(
            atoms_iter,
            desc=desc,
            total=total,
        )
    with log_ctx:
        try:
            for atoms in iterator:
                if reader is None:
                    yield atoms
                    continue
                atoms_dict = reader(atoms)
                atoms_data = atoms_data_from_dict(atoms_dict, task=task)
                atoms_data = atoms_data.to(device=device, dtype=dtype)
                inputs = atoms_data.to_dict()
                if requires_grad and properties.positions in inputs:
                    pos = inputs[properties.positions]
                    if isinstance(pos, torch.Tensor):
                        pos.requires_grad_()
                yield atoms, inputs
        finally:
            if hasattr(atoms_iter, "close"):
                try:
                    atoms_iter.close()
                except Exception:
                    pass
            for attr in ("trajectory", "reader", "_reader"):
                obj = getattr(atoms_iter, attr, None)
                if obj is not None and hasattr(obj, "close"):
                    try:
                        obj.close()
                    except Exception:
                        pass


def iter_batches(
    dataset: Union[Dataset, DataLoader],
    batch_size: int,
    device=None,
    dtype: Optional[torch.dtype] = None,
    desc: Optional[str] = None,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
):
    if pin_memory is None:
        pin_memory = str(device).startswith("cuda") if device is not None else False
    if isinstance(dataset, DataLoader):
        loader = dataset
    else:
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_atoms_data,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    iterator = loader
    use_tqdm = tqdm is not None and desc is not None
    if use_tqdm:
        iterator = tqdm(
            loader,
            desc=desc,
            total=len(loader),
        )
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    log_every = None
    if desc is not None and not use_tqdm and total_batches is not None:
        log_every = max(1, total_batches // 10)
        logger.info("%s: 0/%d batches", desc, total_batches)
    for batch_idx, batch in enumerate(iterator, start=1):
        if log_every is not None and (
            batch_idx == total_batches or batch_idx == 1 or batch_idx % log_every == 0
        ):
            logger.info("%s: %d/%d batches", desc, batch_idx, total_batches)
        if hasattr(batch, "to"):
            yield batch.to(device=device, dtype=dtype)
        else:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    if dtype is not None and v.is_floating_point():
                        batch[k] = v.to(device, dtype=dtype)
                    else:
                        batch[k] = v.to(device)
            yield batch

def _prepare_data_source(
    url: str,
    cache_dir: Optional[Union[str, PosixPath]] = None,
    extract: bool = True,
    filename: Optional[str] = None,
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

    logger.info("Loaded %d structures from %s", len(atoms_list), download_path)
    return atoms_list

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
