import torch
from curator.data import properties, BatchNeighborList

@torch.jit.script
def get_pair_distance(data: properties.Type, force_process: bool=False) -> properties.Type:
    if properties.edge_dist not in data or force_process:
        pos = data[properties.positions]
        edge = data[properties.edge_idx]
        edge_diff = pos[edge[:, 1]] - pos[edge[:, 0]]
        if properties.cell in data:
            edge_diff += data[properties.cell_displacements]
        data[properties.edge_diff] = edge_diff 
        data[properties.edge_dist] = torch.linalg.norm(edge_diff, dim=1)
    
    return data

class PairwiseDistance(torch.nn.Module):
    """This class is used to process neighbor list computed from asap3 or compute neighbor list using the internal neighbor list algorithm.
    It has three functions:
    1). obtain edge offset to process batched indices in the neighbor list.
    2). compute neighbor list if it is not provided in inputs.
    3). compute distance from edge difference or coordinates.
    """
    def __init__(
        self,
        compute_neighborlist: bool = False,
        compute_distance_from_R: bool = False,
        compute_forces: bool = True
    ) -> None:
        super().__init__()
        self.compute_neighborlist = compute_neighborlist
        self.compute_distance_from_R = compute_distance_from_R
        self.compute_forces = compute_forces
        self.batch_nl = None
        if self.compute_neighborlist:
            self.batch_nl = BatchNeighborList(self.cutoff, requires_grad=True, wrap_atoms=True)
    
    def forward(self, data: properties.Type) -> properties.Type:
        if self.batch_nl is not None:
            data[properties.edge_idx], data[properties.edge_diff], data[properties.edge_dist] = self.batch_nl(data)
        elif self.compute_distance_from_R:
            if self.compute_forces:
                    data[properties.positions].requires_grad_()
            data = get_pair_distance(data)
        else:
            if self.compute_forces:
                data[properties.edge_diff].requires_grad_()
            data[properties.edge_dist] = torch.linalg.norm(data[properties.edge_diff], dim=1)
        
        return data