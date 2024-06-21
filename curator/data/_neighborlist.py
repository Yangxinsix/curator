from abc import abstractmethod
import torch
from torch import nn
import numpy as np
from typing import Dict, Tuple
from . import properties
from ._transform import Transform
import sys, warnings
from ase import Atoms
try:
    import asap3
except ModuleNotFoundError:
    warnings.warn("Failed to import asap3 module for fast neighborlist")

def wrap_positions(positions: torch.Tensor, cell: torch.Tensor, eps: float=1e-7) -> torch.Tensor:
    """Wrap positions into the unit cell"""
    # wrap atoms outside of the box
    scaled_pos = positions @ torch.linalg.inv(cell) + eps
    scaled_pos %= 1.0
    scaled_pos -= eps
    return scaled_pos @ cell

class BatchNeighborList(nn.Module):
    """Batch neighbor list"""
    def __init__(self, cutoff: float, requires_grad: bool =True, wrap_atoms: bool =True, return_distance: bool=True) -> None:
        """Batch neighbor list
        
        Args:
            cutoff (float): Cutoff radius
            requires_grad (bool, optional): Whether to calculate gradients. Defaults to True.
            wrap_atoms (bool, optional): Whether to wrap atoms into the unit cell. Defaults to True.
        """
        super().__init__()
        self.requires_grad = requires_grad
        self.return_distance = return_distance
        self.torch_nl = TorchNeighborList(cutoff, requires_grad=requires_grad, wrap_atoms=wrap_atoms, return_distance=return_distance)
    
    def forward(self, data: properties.Type) -> properties.Type:
        if self.requires_grad:
            data[properties.positions].requires_grad_()
        num_offset = torch.zeros_like(data[properties.n_atoms])
        num_offset[1:] = data[properties.n_atoms][:-1]
        num_offset = torch.cumsum(num_offset, dim=0)
        batch_pairs, batch_pair_diff, batch_pair_dist = [], [], []
        for i, num_atoms in enumerate(data[properties.n_atoms]):
            atoms_dict = {
                properties.R: data[properties.positions][num_offset[i]:num_offset[i]+num_atoms],
                properties.cell: data[properties.cell][i*3: (i+1)*3],
            }
            atoms_dict = self.torch_nl(atoms_dict)
            batch_pairs.append(atoms_dict[properties.edge_idx] + num_offset[i])
            batch_pair_diff.append(atoms_dict[properties.edge_diff])
            if self.return_distance:
                batch_pair_dist.append(atoms_dict[properties.edge_dist])

        data[properties.edge_idx] = torch.cat(batch_pairs)
        data[properties.edge_diff] = torch.cat(batch_pair_diff)
        if self.return_distance:
            data[properties.edge_dist] = torch.cat(batch_pair_dist)
        
        return data

class NeighborListTransform(Transform):
    """
    Base classs for calculating neighbor list
    """
    def __init__(
        self,
        cutoff: float,
        requires_grad: bool = False,
        return_distance: bool = False,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.requires_grad = requires_grad
        self.return_distance = return_distance
    
    def forward(self, data: properties.Type) -> properties.Type:
        if properties.cell in data:
            edge_info = self._build_neighbor_list(data[properties.positions], data[properties.cell])
        else:
            edge_info = self._simple_neighbor_list(data[properties.positions])
        data.update(edge_info)
        data[properties.n_pairs] = torch.tensor([data[properties.edge_idx].shape[0]])
        
        if self.requires_grad:
            data[properties.edge_diff].requires_grad_()
        if self.return_distance and properties.edge_dist not in data:
            data[properties.edge_dist] = torch.linalg.norm(data[properties.edge_diff], dim=1)
        return data
    
    @abstractmethod
    def _build_neighbor_list(self, pos: torch.Tensor, cell: torch.Tensor):
        raise NotImplementedError
    
    def _simple_neighbor_list(
        self,
        pos: torch.Tensor,
    ) -> properties.Type:
        dist_mat = torch.cdist(pos, pos)
        mask = dist_mat < self.cutoff
        mask.fill_diagonal_(False)
        pairs = torch.argwhere(mask)
        n_diff = pos[pairs[:, 1]] - pos[pairs[:, 0]]
        outputs = {
            properties.edge_idx: pairs,
            properties.edge_diff: n_diff,
        }
        return outputs


class TorchNeighborList(NeighborListTransform):
    """Neighbor list implemented via PyTorch. The correctness is verified by comparing results to ASE and asap3.
    This class enables the direct calculation of gradients dE/dR.
    
    The speed of this neighbor list algorithm is faster than ase while being about 2 times slower than asap3 if use a sing CPU.
    If use a GPU, it is usually slightly faster than asap3.
    
    Note that current GNN implementations used a lot `atomicAdd` operations, which can result in non-deterministic behavior in the model.
    Model predictions (forces) will be erroneous if using a neighbor list algorithm that different with model training.
    """
    def __init__(
        self,
        *args,
        wrap_atoms: bool=True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        disp_mat = torch.cartesian_prod(
            torch.arange(-1, 2),
            torch.arange(-1, 2),
            torch.arange(-1, 2),
        )
        self.wrap_atoms = wrap_atoms
        self.register_buffer('disp_mat', disp_mat, persistent=False)
    
    def _build_neighbor_list(self, positions: torch.Tensor, cell: torch.Tensor) -> properties.Type:
        # calculate padding size. It is useful for all kinds of cells
        if self.requires_grad:
            positions.requires_grad_()
        wrapped_pos = wrap_positions(positions, cell) if self.wrap_atoms else positions
        norm_a = cell[1].cross(cell[2], dim=-1).norm()
        norm_b = cell[2].cross(cell[0], dim=-1).norm()
        norm_c = cell[0].cross(cell[1], dim=-1).norm()
        volume = torch.sum(cell[0] * cell[1].cross(cell[2], dim=-1))

        # get padding size and padding matrix to generate padded atoms. Use minimal image convention
        padding_a = torch.ceil(self.cutoff * norm_a / volume).long()
        padding_b = torch.ceil(self.cutoff * norm_b / volume).long()
        padding_c = torch.ceil(self.cutoff * norm_c / volume).long()

        padding_mat = torch.cartesian_prod(
            torch.arange(-padding_a, padding_a+1, device=padding_a.device),
            torch.arange(-padding_b, padding_b+1, device=padding_a.device),
            torch.arange(-padding_c, padding_c+1, device=padding_a.device),
        ).to(cell.dtype)
        padding_size = (2 * padding_a + 1) * (2 * padding_b + 1) * (2 * padding_c + 1)

        # padding, calculating cell numbers and shapes
        padded_pos = (wrapped_pos.unsqueeze(1) + padding_mat @ cell).view(-1, 3)
        padded_cpos = torch.floor(padded_pos / self.cutoff).long()
        corner = torch.min(padded_cpos, dim=0)[0]                 # the cell at the corner
        padded_cpos -= corner
        c_pos_shap = torch.max(padded_cpos, dim=0)[0] + 1         # c_pos starts from 0
        num_cells = int(torch.prod(c_pos_shap).item())
        count_vec = torch.ones_like(c_pos_shap)
        count_vec[0] = c_pos_shap[1] * c_pos_shap[2]
        count_vec[1] = c_pos_shap[2]
    
        padded_cind = torch.sum(padded_cpos * count_vec, dim=1)
        padded_gind = torch.arange(padded_cind.shape[0], device=count_vec.device) + 1                                 # global index of padded atoms, starts from 1
        padded_rind = torch.arange(positions.shape[0], device=count_vec.device).repeat_interleave(padding_size)                  # local index of padded atoms in the unit cell

        # atom cell position and index
        atom_cpos = torch.floor(wrapped_pos / self.cutoff).long() - corner
        atom_cind = torch.sum(atom_cpos * count_vec, dim=1)

        # atom neighbors' cell position and index
        atom_cnpos = atom_cpos.unsqueeze(1) + self.disp_mat
        atom_cnind = torch.sum(atom_cnpos * count_vec, dim=-1)
        
        # construct a C x N matrix to store the cell atom list, this is the most expensive part.
        padded_cind_sorted, padded_cind_args = torch.sort(padded_cind, stable=True)
        cell_ind, indices, cell_atom_num = torch.unique_consecutive(padded_cind_sorted, return_inverse=True, return_counts=True)
        max_cell_anum = int(cell_atom_num.max().item())
        global_cell_ind = torch.zeros(
            (num_cells, max_cell_anum, 2),
            dtype=c_pos_shap.dtype, 
            device=c_pos_shap.device,
        )
        cell_aind = torch.nonzero(torch.arange(max_cell_anum, device=count_vec.device).repeat(cell_atom_num.shape[0], 1) < cell_atom_num.unsqueeze(-1))[:, 1]
        global_cell_ind[padded_cind_sorted, cell_aind, 0] = padded_gind[padded_cind_args]
        global_cell_ind[padded_cind_sorted, cell_aind, 1] = padded_rind[padded_cind_args]

        # masking
        atom_nind = global_cell_ind[atom_cnind]
        pair_i, neigh, j = torch.where(atom_nind[:, :, :, 0])
        pair_j = atom_nind[pair_i, neigh, j, 1]
        pair_j_padded = atom_nind[pair_i, neigh, j, 0] - 1          # remember global index of padded atoms starts from 1
        pair_diff = padded_pos[pair_j_padded] - wrapped_pos[pair_i]
        if self.requires_grad:
            pair_diff.requires_grad_()
        pair_dist = torch.norm(pair_diff, dim = 1)
        mask = torch.logical_and(pair_dist < self.cutoff, pair_dist > 0.01)   # 0.01 for numerical stability
        pairs = torch.hstack((pair_i.unsqueeze(-1), pair_j.unsqueeze(-1)))
        
        outputs = {
            properties.edge_idx: pairs[mask],
            properties.edge_diff: pair_diff[mask],
        }
        if self.return_distance:
            outputs[properties.edge_dist] = pair_dist[mask]
        return outputs
        
class Asap3NeighborList(NeighborListTransform):
    def __init__(
        self,
        *args,
        return_cell_displacements: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not ("asap3" in sys.modules):
            raise ModuleNotFoundError("This neighbor list implementation needs ASAP3 module!")
        self.nblist = asap3.FullNeighborList(self.cutoff, atoms=None)
        self.return_cell_displacements = return_cell_displacements
    
    def _build_neighbor_list(
        self, 
        pos: torch.Tensor, 
        cell: torch.Tensor, 
    ) -> properties.Type:
        atoms = Atoms(positions=pos, pbc=True, cell=cell)
        self.nblist.check_and_update(atoms)
        pair_i_idx = []
        pair_j_idx = []
        n_diff = []
        if self.return_cell_displacements:
            atom_positions = atoms.get_positions()
            cell_displacements = []
            
            for i in range(len(atoms)):
                indices, diff, _ = self.nblist.get_neighbors(i)
                pair_i_idx += [i] * len(indices)               # local index of pair i
                pair_j_idx.append(indices)   # local index of pair j
                n_diff.append(diff)
                pos_i = atom_positions[i]
                pos_j = atom_positions[indices]
                displacement = diff + pos_i - pos_j
                cell_displacements.append(displacement)
                
            cell_displacements = np.concatenate(cell_displacements)
            cell_displacements = torch.as_tensor(cell_displacements, dtype=torch.float)
            
        else:
            for i in range(len(atoms)):
                indices, diff, _ = self.nblist.get_neighbors(i)
                pair_i_idx += [i] * len(indices)               # local index of pair i
                pair_j_idx.append(indices)   # local index of pair j
                n_diff.append(diff)

        pair_j_idx = np.concatenate(pair_j_idx)
        pairs = np.stack((pair_i_idx, pair_j_idx), axis=1)
        n_diff = np.concatenate(n_diff)
        pairs = torch.as_tensor(pairs)
        n_diff = torch.as_tensor(n_diff, dtype=torch.float)
        
        outputs = {
            properties.edge_idx: pairs,
            properties.edge_diff: n_diff,
        }
        if self.return_cell_displacements:
            outputs[properties.cell_displacements] = cell_displacements   
        return outputs