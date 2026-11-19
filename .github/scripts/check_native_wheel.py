"""Minimal smoke test for an installed CURATOR binary wheel."""

from __future__ import annotations

import importlib
import importlib.machinery

import numpy as np

from curator.native import neighbor_pairs


native_module = importlib.import_module("curator.native._neighbors")
module_path = str(native_module.__file__)
assert any(module_path.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES), module_path

positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
assert neighbor_pairs(positions, 1.0).count == 1

periodic = neighbor_pairs(
    np.zeros((1, 3), dtype=np.float64),
    2.1,
    cell=np.eye(3, dtype=np.float64) * 2.0,
    pbc=[True, True, True],
)
assert periodic.count == 3

print(f"CURATOR native wheel OK: {module_path}")
