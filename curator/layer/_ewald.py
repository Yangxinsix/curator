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
    """Calculate a 3D-periodic Ewald energy in physical eV units.

    The omitted ``k=0`` mode corresponds to conducting (tin-foil) boundary
    conditions.  By default, non-neutral cells are made well-defined with a
    uniform neutralizing background; set ``neutralizing_background=False``
    only when intentionally reproducing the uncorrected legacy expression.
    """
    CONV_FACT = 1e10 * constants.e / (4 * math.pi * constants.epsilon_0)        # convert units to eV
    def __init__(
        self,
        cutoff=None,
        k_cutoff=None,
        alpha=0.4,
        acc_factor=12.0,
        neutralizing_background=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.acc_factor = acc_factor
        self.accf = math.sqrt(math.log(10**acc_factor))
        self.alpha = alpha
        self.cutoff = cutoff or self.accf / self.alpha
        self.k_cutoff = k_cutoff or 2 * self.alpha * self.accf
        self.neutralizing_background = neutralizing_background

    def forward(
        self,
        data: properties.Type,
        ewald_kernel: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        E_background = self.background_energy(
            data[properties.cell],
            data[properties.image_idx],
            data[properties.atomic_charge],
        )

        return (
            scatter_add(E_real + E_self, data[properties.image_idx], dim=0)
            + E_recip
            + E_background
        )
        
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
        E_real = 1 / 2 * scatter_add(
            E_real,
            edge_idx[:, 0],
            dim=0,
            out=torch.zeros_like(atomic_charges),
        )     # double counted

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
        E_recip = 2 * pi / V * sum_{k != 0} exp(-k^2 / (4 alpha^2)) * |rho(k)|^2 / k^2
        where
        rho(k) = sum_{j=1,N} q_j exp(-i k.r_j)

        Returns one reciprocal-space energy per structure in the batch.
        """
        cell = cell.reshape(-1, 3, 3)
        volumes = torch.abs(torch.sum(cell[:, 0] * cell[:, 1].cross(cell[:, 2], dim=-1), dim=1))
        prefactor = 2.0 * math.pi / volumes

        offset = 0
        E_recip_list = []
        for c, n, p in zip(cell, num_atoms, prefactor):
            # get positions and volumes
            n = n.item()
            pos = positions[offset:offset+n]      # (N,3)
            q = atomic_charges[offset:offset+n]
            offset += n

            # calculate structure factor: rho(k) = sum_j q_j exp(i k dot r_j)
            k_vec, k_sq = self.get_reciprocal_k_vectors(c, self.k_cutoff)   # (M,3), (M,)
            if k_sq.numel() == 0:
                E_recip_list.append(pos.new_zeros(()))
                continue

            k_dot_r = pos @ k_vec.T
            rho_real = torch.sum(q.unsqueeze(1) * torch.cos(k_dot_r), dim=0)
            rho_imag = torch.sum(q.unsqueeze(1) * torch.sin(k_dot_r), dim=0)
            rho_sq = rho_real.square() + rho_imag.square()

            # Damping factor exp(-k^2/(4 alpha^2))
            damping  = torch.exp(- k_sq / (4.0 * self.alpha ** 2))          # (M,)
            E_recip_list.append(torch.sum(damping * rho_sq / k_sq) * p)

        E_recip = torch.stack(E_recip_list)
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

    def background_energy(self, cell, image_idx, atomic_charges) -> torch.Tensor:
        """Return the uniform neutralizing-background correction.

        Omitting the reciprocal-space ``k=0`` mode defines a charged periodic
        cell only together with a uniform background charge.  In the present
        Ewald convention its contribution is

            E_bg = -pi Q^2 / (2 alpha^2 V).

        It vanishes exactly for neutral structures.
        """
        cell = cell.reshape(-1, 3, 3)
        if not self.neutralizing_background:
            return cell.new_zeros(cell.shape[0])

        volumes = torch.abs(torch.linalg.det(cell))
        total_charge = scatter_add(atomic_charges, image_idx, dim=0)
        return (
            -math.pi
            * total_charge.square()
            / (2.0 * self.alpha**2 * volumes)
            * EwaldSummation.CONV_FACT
        )

    def get_ewald_kernel(
        self,
        cell,
        num_atoms,
        positions,
        edge_dist,
        edge_idx,
    ):
        """
        Return one per-structure kernel K in eV such that 0.5 * q^T K q
        reproduces the Ewald energy for that structure.
        """
        # reciprocal part
        cell = cell.reshape(-1, 3, 3)
        volumes = torch.abs(torch.sum(cell[:, 0] * cell[:, 1].cross(cell[:, 2], dim=-1), dim=1))
        prefactor = 2.0 * math.pi / volumes

        # real part
        real_mask = edge_dist < self.cutoff
        edge_dist = edge_dist[real_mask]
        edge_idx = edge_idx[real_mask]
        dist = torch.erfc(self.alpha * edge_dist) / edge_dist

        offset = 0
        kernel_list = []
        for c, n, p, volume in zip(cell, num_atoms, prefactor, volumes):
            # get positions and volumes
            n = n.item()
            pos = positions[offset:offset+n]      # (N,3)
            local_mask = (edge_idx[:, 0] >= offset) & (edge_idx[:, 0] < offset + n)
            local_edges = edge_idx[local_mask] - offset
            local_dist = dist[local_mask]

            # real part
            real_per_image = torch.zeros((n, n), dtype=dist.dtype, device=dist.device)
            if local_edges.numel() > 0:
                real_per_image.index_put_(
                    (local_edges[:, 0], local_edges[:, 1]),
                    local_dist,
                    accumulate=True,
                )
            offset += n

            # calculate structure factor: rho(k) = sum_j q_j exp(i k dot r_j)
            k_vec, k_sq = self.get_reciprocal_k_vectors(c, self.k_cutoff)   # (M,3), (M,)
            if k_sq.numel() == 0:
                recip_per_image = torch.zeros((n, n), dtype=dist.dtype, device=dist.device)
            else:
                k_dot_r = pos @ k_vec.T
                phase_diff = k_dot_r.unsqueeze(1) - k_dot_r.unsqueeze(0)

                # 0.5 * q^T K q uses a pair kernel, so the reciprocal prefactor is doubled here.
                damping = torch.exp(- k_sq / (4.0 * self.alpha ** 2)).view(1, 1, -1)
                recip_per_image = 2.0 * p * torch.sum(
                    damping * torch.cos(phase_diff) / k_sq.view(1, 1, -1),
                    dim=-1,
                )

            # add up self part
            kernel_matrix = real_per_image + recip_per_image + torch.eye(
                n,
                dtype=dist.dtype,
                device=dist.device,
            ) * (-2.0 * self.alpha / math.sqrt(math.pi))
            if self.neutralizing_background:
                # 0.5 q^T K_bg q = -pi (sum_i q_i)^2 / (2 alpha^2 V).
                kernel_matrix = kernel_matrix - torch.ones(
                    (n, n),
                    dtype=dist.dtype,
                    device=dist.device,
                ) * (math.pi / (self.alpha**2 * volume))
            kernel_list.append(kernel_matrix * EwaldSummation.CONV_FACT)

        return kernel_list

    @classmethod
    def get_reciprocal_k_vectors(cls, cell, k_cut):
        """
        Generate all reciprocal vectors ``k = n @ B`` such that ``|k| <= k_cut``.
        
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

        # Curator/ASE store direct lattice vectors as rows.  The reciprocal
        # lattice vectors are therefore also rows of B = 2 pi A^{-T}, and a
        # reciprocal vector is k = n @ B.
        recip_cell = 2.0 * math.pi * torch.linalg.inv(cell).T

        # This is a rigorous component-wise bound, including skew cells:
        # n_i = a_i dot k / (2 pi), hence |n_i| <= |a_i| k_cut / (2 pi).
        n_range_list = torch.ceil(
            k_cut * torch.linalg.vector_norm(cell, dim=1) / (2.0 * math.pi)
        ).long()
        n_range = [torch.arange(-n, n + 1, device=cell.device, dtype=cell.dtype) for n in n_range_list]

        # 2. Build integer grid: n_x, n_y, n_z in [-n_max, ..., n_max]
        nx, ny, nz = torch.meshgrid(n_range, indexing='ij')
        nx_flat = nx.flatten()  # shape (N^3,)
        ny_flat = ny.flatten()
        nz_flat = nz.flatten()

        # 3. Convert (n_x, n_y, n_z) -> k = n @ B
        n_xyz = torch.stack([nx_flat, ny_flat, nz_flat], dim=1)  # (N^3, 3)
        k_matrix = n_xyz @ recip_cell  # (N^3, 3)

        # 4. Compute squared magnitude and filter by k_cut and exclude k=0
        k_sq_full = (k_matrix**2).sum(dim=1)
        mask = (k_sq_full <= k_cut**2) & (k_sq_full > 0)  # exclude zero
        k_vectors = k_matrix[mask]
        k_sq = k_sq_full[mask]

        return k_vectors, k_sq
