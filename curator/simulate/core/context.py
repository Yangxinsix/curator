from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
from ase import Atoms
from curator.data import read_trajectory

def load_atoms_or_traj(init_traj: Union[str, Atoms], start: Optional[int] = None) -> Atoms:
    """Load a single Atoms either from an Atoms object or a multi-frame file."""
    if isinstance(init_traj, Atoms):
        return init_traj.copy()
    if not os.path.isfile(init_traj):
        raise FileNotFoundError(f"init_traj not found: {init_traj}")
    images = read_trajectory(init_traj, index=":")
    if not images:
        raise RuntimeError(f"No frames in {init_traj}")
    idx = np.random.randint(len(images)) if start is None else start
    return images[idx]

@dataclass
class SimContext:
    """Shared context for engines & callbacks."""
    atoms: Optional[Atoms] = None
    step: int = 0
    engine: Optional["BaseEngine"] = None
    simulator: Optional["Simulator"] = None
    state: Dict[str, Any] = field(default_factory=dict)