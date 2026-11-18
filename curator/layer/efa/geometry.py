"""Geometry for EFA's spherical and official raw-lattice encodings.

No wrapping, reciprocal-space substitution, or electrostatic kernel is applied.
Keeping this stateless preserves the existing adapter checkpoint layout.
"""
import torch

from curator.data import properties


def structure_cells(data, num_structures):
    """Return one row-wise cell or None per structure, validating boundary flags."""
    if properties.pbc not in data:
        raise ValueError("EFAAdapter requires explicit pbc flags")
    pbc = data[properties.pbc]
    allowed = [(num_structures, 3)]
    if num_structures == 1:
        allowed.append((3,))
    if pbc.dtype != torch.bool or tuple(pbc.shape) not in allowed:
        raise ValueError("EFA pbc must be boolean with shape (num_structures, 3)")
    pbc = pbc.reshape(num_structures, 3)
    periodic = pbc.all(dim=-1)
    if bool((pbc.any(dim=-1) != periodic).any()):
        raise NotImplementedError("EFA supports FFF and TTT boundaries; partial periodicity is unsupported")
    if not bool(periodic.any()):
        return [None] * num_structures
    if properties.cell not in data:
        raise ValueError("Periodic EFA requires a cell for every structure")
    cells = data[properties.cell]
    if tuple(cells.shape) not in [(3 * num_structures, 3), (num_structures, 3, 3)]:
        raise ValueError("EFA cell must have shape (3*num_structures, 3) or (num_structures, 3, 3)")
    cells = cells.reshape(num_structures, 3, 3)
    return [cells[i] if bool(periodic[i]) else None for i in range(num_structures)]


def encoding_grid(positions, cell, spherical_grid, spherical_weights):
    """Choose the fixed spherical rule or three raw lattice rows weighted by 1/3."""
    if cell is None:
        if spherical_grid.numel() == 0:
            raise ValueError("FFF EFA requires num_directions; None selects a periodic-only core")
        return spherical_grid.to(positions), spherical_weights.to(positions)
    if cell.shape != (3, 3) or not cell.is_floating_point():
        raise ValueError("Periodic EFA cell must be a floating-point (3, 3) tensor")
    if not bool(torch.isfinite(cell).all()) or bool(torch.linalg.det(cell.detach()) == 0):
        raise ValueError("Periodic EFA cell must be finite and nonsingular")
    return cell.to(positions), positions.new_full((3,), 1 / 3)


def rotary_encoding(values, positions, directions, frequencies):
    """Encode channel pairs, with a common translation gauge for numerical accuracy."""
    positions = positions - positions.mean(0, keepdim=True)
    phase = (positions @ directions.T)[..., None] * frequencies.to(positions)
    cosine, sine = phase.cos()[:, :, None, :], phase.sin()[:, :, None, :]
    even, odd = values[:, None, :, 0::2], values[:, None, :, 1::2]
    rotated = torch.stack((even * cosine - odd * sine,
                           even * sine + odd * cosine), dim=-1)
    return rotated.flatten(2)
