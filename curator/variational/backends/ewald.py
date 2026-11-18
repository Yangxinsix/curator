"""Differentiable point-charge kernels: 3D Ewald or isolated Coulomb."""

from typing import Dict, List

import torch

from curator.data import properties
from curator.layer._ewald import EwaldSummation

from ..operators import DenseQuadraticOperator


class EwaldChargeOperator:
    """Point-charge operator with explicit periodic or isolated boundaries.

    TTT uses the existing Ewald convention; FFF uses untruncated isolated
    Coulomb interactions. Partially periodic cells require a separate kernel
    and are rejected rather than being silently treated as 3D periodic.
    """

    def __init__(self, ewald: EwaldSummation):
        self.ewald = ewald

    def geometry_kernels(self, data: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Build complete kernels independently of the GNN neighbor cutoff.

        The selected backend finds discrete periodic neighbors from detached geometry; their
        displacements are reconstructed from the live tensors. Thus coordinate
        and cell derivatives include both real and reciprocal terms. No tensor
        is cached across forward/optimizer steps. The returned per-geometry
        kernels can be reused across conditions, proposal blocks and energies.

        Explicit TTT uses the unchanged 3D Ewald calculation; FFF uses all
        isolated pairs, independent of both neighbor and Ewald cutoffs. There
        is no periodic image, neutralizing background, or Ewald self term in
        the isolated case. Alpha is a numerical Ewald splitting parameter,
        not a physical Gaussian charge width. Partial periodicity is rejected.
        ``kernels`` remains the lower-level periodic API for a complete list.
        """
        from ase import Atoms
        from ase.neighborlist import neighbor_list

        backend = getattr(self.ewald, "neighbor_backend", "ase")
        if backend not in ("ase", "matscipy", "auto"):
            raise ValueError("neighbor_backend must be ase, matscipy, or auto")
        periodic_neighbors = neighbor_list
        if backend in ("matscipy", "auto"):
            try:
                from matscipy.neighbours import neighbour_list as periodic_neighbors
            except ImportError:
                if backend == "matscipy":
                    raise ImportError("matscipy neighbor backend requires the optional matscipy package")
                periodic_neighbors = neighbor_list

        positions = data[properties.positions]
        counts = data[properties.n_atoms]
        cells = data.get(properties.cell, positions.new_zeros((counts.numel(), 3, 3))).reshape(-1, 3, 3)
        pbc = data.get(properties.pbc)
        if (counts.ndim != 1 or cells.shape[0] != counts.numel()
                or int(counts.sum()) != positions.shape[0] or bool((counts <= 0).any())):
            raise ValueError("Ewald geometry metadata must match positions and cells")
        if pbc is None or pbc.numel() != 3 * counts.numel():
            raise ValueError("Charge response requires explicit 3D periodicity (TTT) or isolated boundaries (FFF)")
        pbc = pbc.reshape(-1, 3).bool()
        periodic = pbc.all(dim=1)
        if bool((pbc.any(dim=1) != periodic).any()):
            raise ValueError("Charge response requires 3D periodicity (TTT) or isolated boundaries (FFF); partial periodicity is unsupported")
        if not bool(torch.isfinite(cells).all()) or bool((torch.linalg.det(cells[periodic]).abs() < 1e-12).any()):
            raise ValueError("Ewald charge response requires finite cells and nonsingular periodic cells")
        if not bool(torch.isfinite(positions).all()):
            raise ValueError("Ewald positions must be finite")
        kernels = []
        offset = 0
        for count, cell, is_periodic in zip(counts.tolist(), cells, periodic.tolist()):
            local_positions = positions[offset:offset + count]
            if not is_periodic:
                kernels.append(self.isolated_kernel(local_positions))
                offset += count
                continue
            atoms = Atoms(numbers=[1] * count,
                          positions=local_positions.detach().cpu().numpy(),
                          cell=cell.detach().cpu().numpy(), pbc=True)
            # Both APIs exclude only the zero-shift self pair by default and
            # include every periodic image within the cutoff. Rebuild each call:
            # neither geometry nor topology is cached across moving cells/atoms.
            with torch.profiler.record_function("acvr.ewald_neighbor_indices"):
                center, neighbor, shifts = periodic_neighbors("ijS", atoms, self.ewald.cutoff)
            edges = torch.stack((torch.as_tensor(center, device=positions.device),
                                 torch.as_tensor(neighbor, device=positions.device)), dim=1).long()
            shifts = torch.as_tensor(shifts, dtype=positions.dtype, device=positions.device)
            difference = local_positions[edges[:, 1]] - local_positions[edges[:, 0]] + shifts @ cell
            kernels.extend(self.kernels({
                properties.positions: local_positions,
                properties.cell: cell.unsqueeze(0),
                properties.n_atoms: counts.new_tensor([count]),
                properties.edge_idx: edges,
                properties.edge_dist: torch.linalg.vector_norm(difference, dim=-1),
            }))
            offset += count
        return kernels

    @staticmethod
    def isolated_kernel(positions: torch.Tensor) -> torch.Tensor:
        """K_ij = e^2/(4 pi epsilon_0 r_ij) for i != j, K_ii = 0.

        Positions are Angstroms, charge is elementary charge, and energy
        ``0.5*q@K@q`` is eV. Atomic self hardness belongs to the internal
        energy. Avoid evaluating sqrt at a zero diagonal: masking an already
        singular 1/r would leave NaNs in higher coordinate derivatives.
        Coincident distinct atoms are an invalid point-charge geometry.
        """
        count = positions.shape[0]
        diagonal = torch.eye(count, dtype=torch.bool, device=positions.device)
        displacement = positions[:, None, :] - positions[None, :, :]
        squared_distance = displacement.square().sum(dim=-1)
        if bool(((squared_distance == 0) & ~diagonal).any()):
            raise ValueError("Distinct point charges cannot occupy the same position")
        safe_squared_distance = torch.where(diagonal, torch.ones_like(squared_distance), squared_distance)
        return torch.where(diagonal, torch.zeros_like(squared_distance),
                           safe_squared_distance.rsqrt()) * EwaldSummation.CONV_FACT

    def kernels(self, data: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        if properties.pbc in data and not bool(data[properties.pbc].all()):
            return self.geometry_kernels(data)
        kernels = self.ewald.get_ewald_kernel(
            data[properties.cell],
            data[properties.n_atoms],
            data[properties.positions],
            data[properties.edge_dist],
            data[properties.edge_idx],
        )
        return [0.5 * (kernel + kernel.transpose(0, 1)) for kernel in kernels]

    def energy(self, data: Dict[str, torch.Tensor], charge: torch.Tensor) -> torch.Tensor:
        if properties.pbc in data and not bool(data[properties.pbc].all()):
            counts = data[properties.n_atoms]
            if charge.ndim != 1 or int(counts.sum()) != charge.numel():
                raise ValueError("charge must be rank 1 and match n_atoms")
            return torch.stack([0.5 * q @ kernel @ q for q, kernel in
                                zip(charge.split(counts.tolist()), self.geometry_kernels(data))])
        updated = data.copy()
        updated[properties.atomic_charge] = charge
        return self.ewald(updated)

    def matvec(self, data: Dict[str, torch.Tensor], charge: torch.Tensor) -> torch.Tensor:
        counts = data[properties.n_atoms]
        if counts.ndim != 1:
            raise ValueError("n_atoms must be a rank-1 tensor")
        if int(counts.sum().item()) != charge.numel():
            raise ValueError("charge length does not match n_atoms")
        kernels = self.kernels(data)
        if len(kernels) != counts.numel():
            raise ValueError("Ewald kernel count does not match n_atoms")
        products = []
        offset = 0
        for count, kernel in zip(counts.tolist(), kernels):
            if kernel.shape != (count, count):
                raise ValueError("Ewald kernel shape does not match n_atoms")
            if kernel.dtype != charge.dtype or kernel.device != charge.device:
                raise ValueError("Ewald kernels must match charge dtype and device")
            products.append(kernel @ charge[offset : offset + count])
            offset += count
        if offset != charge.numel():
            raise ValueError("Ewald batch metadata did not consume every charge")
        return torch.cat(products) if products else charge.new_zeros((0,))

    def bind(
        self,
        data: Dict[str, torch.Tensor],
        structure_index: int = 0,
    ) -> DenseQuadraticOperator:
        """Bind one reference structure to the state-only operator protocol.

        The Torch reference backend materializes its small dense Ewald kernel.
        Production LAMMPS/PPPM bindings will implement the same ``energy`` and
        ``matvec`` protocol without materializing that kernel.
        """

        kernels = self.kernels(data)
        if structure_index < 0:
            structure_index += len(kernels)
        if structure_index < 0 or structure_index >= len(kernels):
            raise IndexError(structure_index)
        return DenseQuadraticOperator(kernels[structure_index])
