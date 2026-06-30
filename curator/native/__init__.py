from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from curator.native._neighbors import neighbor_pairs as _neighbor_pairs


@dataclass(frozen=True)
class NeighborPairs:
    i: np.ndarray
    j: np.ndarray
    offsets: np.ndarray
    distance: np.ndarray

    @property
    def count(self) -> int:
        return int(self.i.shape[0])

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "i": self.i,
            "j": self.j,
            "offsets": self.offsets,
            "distance": self.distance,
        }


def neighbor_pairs(
    positions: Any,
    cutoff: float,
    *,
    cell: Any | None = None,
    pbc: Any | None = None,
    atomic_numbers: Any | None = None,
    pair_cutoffs: Any | None = None,
    num_threads: int = 0,
) -> NeighborPairs:
    result = _neighbor_pairs(
        np.ascontiguousarray(positions, dtype=np.float64),
        float(cutoff),
        None if cell is None else np.ascontiguousarray(cell, dtype=np.float64),
        [False, False, False] if pbc is None else [bool(v) for v in pbc],
        None if atomic_numbers is None else np.ascontiguousarray(atomic_numbers, dtype=np.int32),
        None if pair_cutoffs is None else np.ascontiguousarray(pair_cutoffs, dtype=np.float64),
        int(num_threads),
    )
    i = np.frombuffer(result["i"], dtype=np.int32)
    j = np.frombuffer(result["j"], dtype=np.int32)
    sx = np.frombuffer(result["sx"], dtype=np.int8)
    sy = np.frombuffer(result["sy"], dtype=np.int8)
    sz = np.frombuffer(result["sz"], dtype=np.int8)
    offsets = np.stack([sx, sy, sz], axis=1)
    return NeighborPairs(
        i=i,
        j=j,
        offsets=offsets,
        distance=np.frombuffer(result["distance"], dtype=np.float64),
    )


__all__ = ["NeighborPairs", "neighbor_pairs"]
