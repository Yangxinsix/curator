from torch import nn
import torch
from curator.data import properties
from typing import Optional
import math
try:
    from torch_scatter import scatter_add
except ImportError:
    from curator.utils import scatter_add
from scipy import constants

class EwaldSummation(nn.Module):
    """
    Calculate Ewald summation given atomic charges.
    """
    CONV_FACT = 1e10 * constants.e / (4 * math.pi * constants.epsilon_0)        # convert units to eV
    def __init__(
        self,
        cutoff=None,
        k_cutoff=None,
        alpha=0.4,
        acc_factor=12.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.acc_factor = acc_factor
        self.accf = math.sqrt(math.log(10**acc_factor))
        self.alpha = alpha
        self.cutoff = cutoff or self.accf / self.alpha
        self.k_cutoff = k_cutoff or 2 * self.alpha * self.accf

    def forward(self, data: properties.Type, ewald_kernel: Optional[torch.Tensor] = None) -> properties.Type:
        # screen neighbor list
        mask = data[properties.edge_dist] < self.cutoff
        edge_dist = data[properties.edge_dist][mask]
        edge_idx = data[properties.edge_idx][mask]

        E_real = self.real_space_energy(
            edge_dist,
            edge_idx,
            data[properties.atomic_charge],
        )

        E_recip = self.reciprocal_space_energy(
            data[properties.cell],
            data[properties.n_atoms],
            data[properties.positions],
            data[properties.atomic_charge],
        )

        E_self = self.self_energy(data[properties.atomic_charge])

        return scatter_add(E_real + E_recip + E_self, data[properties.image_idx])
        
    def real_space_energy(
        self,
        edge_dist,
        edge_idx,
        atomic_charges,
    ) -> torch.Tensor:
        """
        Compute the real-space (short-range) part of the Ewald sum:

            E_real = (1/2) sum_{i != j} sum_{n in images} 
                       q_i q_j erfc(alpha * r_nij) / r_nij

        Neighbor list is obtained from elsewhere
        """
        dist = torch.erfc(self.alpha * edge_dist) / edge_dist
        E_real = atomic_charges[edge_idx[:, 0]] * atomic_charges[edge_idx[:, 1]] * dist
        E_real = 1 / 2 * scatter_add(E_real, edge_idx[:, 0])     # double counted

        return E_real * EwaldSummation.CONV_FACT

    def reciprocal_space_energy(
        self, 
        cell,
        num_atoms,
        positions,
        atomic_charges,
    ) -> torch.Tensor:
        """
        Perform the reciprocal space summation. The calculation is based on:
        E_recip = 1/(2 * V * pi) * sum_{k != 0} exp(-k^2 / (4 alpha^2)) * |rho(k)|^2 / k^2
        where
        rho(k) = sum_{j=1,N} q_j exp(-i k.r_j)
        """
        cell = cell.reshape(-1, 3, 3)
        volumes = torch.sum(cell[:, 0] * cell[:, 1].cross(cell[:, 2], dim=-1), dim=1)
        prefactor = (1.0 / (2.0 * volumes * math.pi))

        offset = 0
        E_recip_list = []
        for c, n in zip(cell, num_atoms):
            # get positions and volumes
            n = n.item()
            pos = positions[offset:offset+n]      # (N,3)
            q = atomic_charges[offset:offset+n]
            offset += n

            # calculate structure factor: rho(k) = sum_j q_j exp(i k dot r_j)
            k_vec, k_sq = self.get_reciprocal_k_vectors(c, self.k_cutoff)   # (M,3), (M,)
            k_dot_r = (k_vec.unsqueeze(0) * pos.unsqueeze(1)).sum(-1)       # (1,M,3) * (N,1,3) -> (N,M)
            real_part = q.unsqueeze(1) * torch.cos(k_dot_r)  # (N,M)
            imag_part = q.unsqueeze(1) * torch.sin(k_dot_r)  # (N,M)
            rho_sq = real_part**2 + imag_part**2             # (N,M)

            # Damping factor exp(-k^2/(4 alpha^2))
            damping  = torch.exp(- k_sq / (4.0 * self.alpha ** 2))          # (M,)
            E_recip_list.append(torch.sum(damping * rho_sq / k_sq, dim=-1))  # (N,)

        E_recip = prefactor * torch.concat(E_recip_list)
        return E_recip * EwaldSummation.CONV_FACT
    
    def self_energy(self, atomic_charges) -> torch.Tensor: 
        """
        Self-energy correction:

            E_self = - (alpha / sqrt(pi)) * sum_i q_i^2

        We subtract this once, because in the splitting approach each charge interacts
        with its own Gaussian screening cloud.
        """
        sum_q_sq = atomic_charges ** 2
        E_self = - self.alpha / math.sqrt(math.pi) * sum_q_sq

        return E_self * EwaldSummation.CONV_FACT

    def get_ewald_kernel(
        self,
        cell,
        num_atoms,
        positions,
        edge_dist,
        edge_idx,
    ):
        # reciprocal part
        cell = cell.reshape(-1, 3, 3)
        volumes = torch.sum(cell[:, 0] * cell[:, 1].cross(cell[:, 2], dim=-1), dim=1)
        prefactor = (1.0 / (2.0 * volumes * math.pi))

        # real part
        dist = torch.erfc(self.alpha * edge_dist) / edge_dist
        real_matrix = torch.zeros(edge_idx.max().item(), edge_idx.max().item(), dtype=dist.dtype, device=dist.device)

        offset = 0
        kernel_list = []
        for c, n, p in zip(cell, num_atoms, prefactor):
            # get positions and volumes
            n = n.item()
            pos = positions[offset:offset+n]      # (N,3)

            # real part
            real_per_image = real_matrix[offset:offset+n, offset:offset+n]
            offset += n

            # calculate structure factor: rho(k) = sum_j q_j exp(i k dot r_j)
            k_vec, k_sq = self.get_reciprocal_k_vectors(c, self.k_cutoff)   # (M,3), (M,)
            k_dot_r = (k_vec.unsqueeze(1) * pos.unsqueeze(0)).sum(-1)       # (M,1,3) * (1,N,3) -> (M,N)   this place is slightly different from reciprocal energy part
            phases = torch.exp(1j * k_dot_r)                                # (M,N)
            M = phases.unsqueeze(1) * phases.unsqueeze(0).conj()            # (M,N,N)
            M = M.real

            # Damping factor exp(-k^2/(4 alpha^2))
            damping  = torch.exp(- k_sq / (4.0 * self.alpha ** 2))          # (M,)
            recip_per_image = torch.sum(damping * M / k_sq, dim=-1) * p   # (N,N)

            # add up self part
            kernel_matrix = real_per_image + recip_per_image + torch.eye(n, dtype=dist.dtype, device=dist.device) * (-self.alpha / math.sqrt(math.pi) * 2)
            kernel_list.append(kernel_matrix) 

        return kernel_list

    @classmethod
    def get_reciprocal_k_vectors(cls, cell, k_cut):
        """
        Generate all reciprocal vectors k = B @ n such that |k| <= k_cut.
        
        Parameters
        ----------
        cell : torch.Tensor, shape (3, 3)
            Real space lattice matrix.
        k_cut : float
            Cutoff for |k|.

        Returns
        -------
        k_vectors : torch.Tensor, shape (M, 3)
            Valid reciprocal vectors where |k| <= k_cut and k != 0.
        k_sq : torch.Tensor, shape (M,)
            Squared magnitudes of the returned k_vectors.
        """

        recip_cell = 2.0 * math.pi * torch.inverse(cell).T
        
        # 1. Estimate max integer index n_max to capture all k up to k_cut
        #    We'll use the minimum norm of B's columns as a guide.
        b_col_norms = torch.norm(recip_cell, dim=0)  # length of each reciprocal-lattice vector
        n_range_list = torch.ceil(k_cut / b_col_norms).long()
        n_range = [torch.arange(-n, n + 1, device=cell.device, dtype=cell.dtype) for n in n_range_list]

        # 2. Build integer grid: n_x, n_y, n_z in [-n_max, ..., n_max]
        nx, ny, nz = torch.meshgrid(n_range, indexing='ij')
        nx_flat = nx.flatten()  # shape (N^3,)
        ny_flat = ny.flatten()
        nz_flat = nz.flatten()

        # 3. Convert (n_x, n_y, n_z) -> k = B @ n
        n_xyz = torch.stack([nx_flat, ny_flat, nz_flat], dim=0)  # (3, N^3)
        k_matrix = (recip_cell @ n_xyz).T  # (N^3, 3)

        # 4. Compute squared magnitude and filter by k_cut and exclude k=0
        k_sq_full = (k_matrix**2).sum(dim=1)
        mask = (k_sq_full <= k_cut**2) & (k_sq_full > 0)  # exclude zero
        k_vectors = k_matrix[mask]
        k_sq = k_sq_full[mask]

        return k_vectors, k_sq