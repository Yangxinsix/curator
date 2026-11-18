"""Compatibility imports for EFA models saved before the package migration.

Keep these module globals: pickle resolves the original fully qualified class
names through this module, and historical Hydra configurations use this path.
New code should import from :mod:`curator.layer.efa`.
"""
from .efa.adapter import EFAAdapter
from .efa.core import EuclideanFastAttention, _lebedev_grid
from .efa.geometry import encoding_grid, rotary_encoding, structure_cells

__all__ = ["EFAAdapter", "EuclideanFastAttention"]
