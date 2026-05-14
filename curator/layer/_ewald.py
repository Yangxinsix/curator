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
        E_recip = (2 * pi / V) * sum_{k != 0} exp(-k^2 / (4 alpha^2)) * |rho(k)|^2 / k^2
        where
        rho(k) = sum_{j=1,N} q_j exp(-i k.r_j)
        
        IMPORTANT: |rho(k)|^2 is the squared magnitude of the TOTAL structure factor,
        computed as S_real^2 + S_imag^2 where S_real = sum_j q_j cos(k.r_j) and
        S_imag = sum_j q_j sin(k.r_j). The energy is then distributed equally among atoms.
        """
        cell = cell.reshape(-1, 3, 3)
        volumes = torch.sum(cell[:, 0] * cell[:, 1].cross(cell[:, 2], dim=-1), dim=1)
        prefactor = (2.0 * math.pi / volumes)

        offset = 0
        E_recip_list = []
        for c, n, p in zip(cell, num_atoms, prefactor):
            # get positions and volumes
            n = n.item()
            pos = positions[offset:offset+n]      # (N,3)
            q = atomic_charges[offset:offset+n]   # (N,)
            offset += n

            # calculate structure factor: rho(k) = sum_j q_j exp(i k dot r_j)
            k_vec, k_sq = self.get_reciprocal_k_vectors(c, self.k_cutoff)   # (M,3), (M,)
            k_dot_r = (k_vec.unsqueeze(0) * pos.unsqueeze(1)).sum(-1)       # (N,M)
            
            # CORRECT: Sum over atoms FIRST, then compute |rho(k)|^2
            # S_real(k) = sum_j q_j * cos(k.r_j)
            # S_imag(k) = sum_j q_j * sin(k.r_j)
            S_real = (q.unsqueeze(1) * torch.cos(k_dot_r)).sum(dim=0)  # (M,)
            S_imag = (q.unsqueeze(1) * torch.sin(k_dot_r)).sum(dim=0)  # (M,)
            rho_sq = S_real**2 + S_imag**2  # |rho(k)|^2, shape (M,)

            # Damping factor exp(-k^2/(4 alpha^2))
            damping = torch.exp(-k_sq / (4.0 * self.alpha ** 2))  # (M,)
            
            # Total reciprocal energy for this image (scalar)
            E_recip_total = p * torch.sum(damping * rho_sq / k_sq)
            
            # Distribute equally among atoms for per-atom energy output
            # This is necessary because the reciprocal energy is a collective property
            E_recip_per_atom = E_recip_total / n
            E_recip_list.append(E_recip_per_atom.expand(n))

        E_recip = torch.concat(E_recip_list)
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
        """
        Compute the Ewald kernel matrix J_ij for QEQ.
        
        The kernel relates to the Coulomb interaction via:
            E_coulomb = (1/2) sum_{ij} q_i J_ij q_j
        
        J_ij = J_ij^real + J_ij^recip + J_ii^self
        
        where:
        - J_ij^real = erfc(alpha * r_ij) / r_ij  (short-range, from neighbor list)
        - J_ij^recip = (4*pi/V) sum_k exp(-k^2/4alpha^2)/k^2 * cos(k.(r_i-r_j))
        - J_ii^self = -2 * alpha / sqrt(pi)  (self-interaction correction)
        """
        cell = cell.reshape(-1, 3, 3)
        volumes = torch.sum(cell[:, 0] * cell[:, 1].cross(cell[:, 2], dim=-1), dim=1)
        prefactor = (4.0 * math.pi / volumes)

        # Build real-space kernel from neighbor list
        # erfc(alpha * r_ij) / r_ij for each pair
        real_kernel_values = torch.erfc(self.alpha * edge_dist) / edge_dist
        
        # Determine total number of atoms
        total_atoms = edge_idx.max().item() + 1
        
        # Initialize full real-space matrix (will be sparse but we use dense for simplicity)
        real_matrix = torch.zeros(total_atoms, total_atoms, dtype=edge_dist.dtype, device=edge_dist.device)
        
        # Fill in the real-space kernel matrix from neighbor list
        # J_ij^real = erfc(alpha * r_ij) / r_ij
        real_matrix[edge_idx[:, 0], edge_idx[:, 1]] = real_kernel_values
        # Symmetrize (neighbor list may not be symmetric)
        real_matrix = 0.5 * (real_matrix + real_matrix.T)

        offset = 0
        kernel_list = []
        for c, n, p in zip(cell, num_atoms, prefactor):
            n = n.item()
            pos = positions[offset:offset+n]  # (N,3)

            # Extract real-space kernel for this image
            real_per_image = real_matrix[offset:offset+n, offset:offset+n]

            # Reciprocal space kernel: (4*pi/V) sum_k exp(-k^2/4alpha^2)/k^2 * cos(k.(r_i-r_j))
            k_vec, k_sq = self.get_reciprocal_k_vectors(c, self.k_cutoff)  # (M,3), (M,)
            k_dot_r = (k_vec.unsqueeze(1) * pos.unsqueeze(0)).sum(-1)  # (M,N)
            phases = torch.exp(1j * k_dot_r)  # (M,N)
            
            # Outer product: e^{ik.r_i} * e^{-ik.r_j} = e^{ik.(r_i - r_j)}
            # Sum over k: sum_k f(k) * cos(k.(r_i - r_j))
            # Using: Re[e^{ik.r_i} * e^{-ik.r_j}] = cos(k.(r_i - r_j))
            outer = phases.unsqueeze(2) * phases.unsqueeze(1).conj()  # (M,N,N)
            outer_real = outer.real  # cos(k.(r_i - r_j))

            # Damping factor and sum over k
            damping = torch.exp(-k_sq / (4.0 * self.alpha ** 2))  # (M,)
            # Reshape for broadcasting: (M,1,1) * (M,N,N) / (M,1,1)
            recip_per_image = torch.sum(
                damping.unsqueeze(1).unsqueeze(2) * outer_real / k_sq.unsqueeze(1).unsqueeze(2), 
                dim=0
            ) * p  # (N,N)
            
            offset += n

            # Self-interaction correction (diagonal)
            # Factor of 2 because this is J_ii, and E_self = -alpha/sqrt(pi) * q_i^2
            # In J matrix: J_ii^self = -2 * alpha / sqrt(pi)
            self_correction = torch.eye(n, dtype=edge_dist.dtype, device=edge_dist.device) * (-2.0 * self.alpha / math.sqrt(math.pi))

            # Total kernel matrix (multiply by CONV_FACT for eV units)
            kernel_matrix = (real_per_image + recip_per_image + self_correction) * EwaldSummation.CONV_FACT
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