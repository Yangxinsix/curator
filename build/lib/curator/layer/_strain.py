import torch
from curator.data import properties

class Strain(torch.nn.Module):
    """
    copied from schnetpack with a little revision.
    see https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/atomistic/response.py
    """
    def forward(self, data: properties.Type) -> properties.Type:
        cell = data[properties.cell].view(-1, 3, 3)
        strain = torch.zeros_like(cell, requires_grad=True)
        data[properties.strain] = strain
        strain = strain.transpose(1, 2)

        # strain cell
        cell = cell + torch.matmul(cell, strain)
        data[properties.cell] = cell.view(-1, 3)

        # strain positions
        image_idx = data[properties.image_idx]
        strain_i = strain[image_idx]                                                  # strain on atom
        data[properties.positions] = data[properties.positions] + torch.matmul(
            data[properties.positions][:, None, :], strain_i
        ).squeeze(1)

        idx_i = data[properties.edge_idx][:, 0]
        strain_ij = strain_i[idx_i]                                                   # strain on pairs
        data[properties.cell_displacements] = data[properties.cell_displacements] + torch.matmul(
            data[properties.cell_displacements][:, None, :], strain_ij
        ).squeeze(1)
        
        return data