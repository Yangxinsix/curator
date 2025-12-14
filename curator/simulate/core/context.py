from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, List, Sequence
from ase import Atoms
from curator.data import read_trajectory
from omegaconf import ListConfig

SequenceLike = (list, tuple, np.ndarray, ListConfig)


def _as_sequence(obj: Any) -> Optional[List[Any]]:
    """Normalize supported sequence-like inputs to a list (but not strings)."""
    if isinstance(obj, SequenceLike):
        return list(obj)
    return None


def _load_single_source(
    source: Union[str, Atoms],
    start: Optional[Union[int, Sequence[int]]] = None,
    all_frames: bool = False,
) -> Union[Atoms, List[Atoms]]:
    """Load atoms from a single source (Atoms or trajectory path)."""
    # Already an atoms object
    if isinstance(source, Atoms):
        seq_start = _as_sequence(start)
        if seq_start is not None:
            return [source.copy() for _ in seq_start]
        return source.copy() if not all_frames else [source.copy()]

    if not os.path.isfile(source):
        raise FileNotFoundError(f"init_traj not found: {source}")

    images = read_trajectory(source, index=":")
    if not images:
        raise RuntimeError(f"No frames in {source}")

    start_seq = _as_sequence(start)
    if start_seq is not None:
        return [images[int(idx)].copy() for idx in start_seq]

    if start is not None and all_frames:
        return [images[int(start)].copy()]

    if all_frames:
        return [img.copy() for img in images]

    idx = np.random.randint(len(images)) if start is None else int(start)
    return images[idx].copy()


def load_atoms_or_traj(
    init_traj: Union[str, Atoms, List[Atoms], Sequence[str]],
    start: Optional[Union[int, Sequence[int], Sequence[Optional[int]]]] = None,
    all_frames: bool = False,
) -> Union[Atoms, List[Atoms]]:
    """
    Load an Atoms or list of Atoms from a path, object, or sequence of sources.

    - init_traj can be a single Atoms, a path, or a sequence of either.
    - start can be an int or a sequence of indices to select specific frames.
    - When batch/all_frames is True, selected frames are returned as a list.
    """
    traj_seq = _as_sequence(init_traj)
    start_seq = _as_sequence(start)

    # Multiple trajectory sources
    if traj_seq is not None:
        # Align per-source start indices; broadcast a single start value if provided.
        start_list: List[Optional[Union[int, Sequence[int]]]]
        if start_seq is None:
            start_list = [start for _ in traj_seq]
        elif len(traj_seq) == 1:
            # keep full sequence for the single source to allow multi-frame selection
            start_list = [start_seq]
        elif len(start_seq) == 1 and len(traj_seq) > 1:
            start_list = [start_seq[0] for _ in traj_seq]
        elif len(start_seq) != len(traj_seq):
            raise ValueError("Length of start_index must match init_traj when both are sequences.")
        else:
            start_list = list(start_seq)

        loaded: List[Atoms] = []
        for src, st in zip(traj_seq, start_list):
            atoms = _load_single_source(src, start=st, all_frames=all_frames)
            if isinstance(atoms, list):
                loaded.extend(atoms)
            else:
                loaded.append(atoms)
        return loaded

    # Single source (Atoms or path)
    return _load_single_source(init_traj, start=start, all_frames=all_frames)

@dataclass
class SimContext:
    """Shared context for engines & callbacks."""
    atoms: Union[Atoms, List[Atoms]] = None
    step: int = 0
    engine: Optional["BaseEngine"] = None
    simulator: Optional["Simulator"] = None
    state: Dict[str, Any] = field(default_factory=dict)
