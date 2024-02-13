import torch
from torch import nn
from typing import Dict, List, Optional, DefaultDict
from curator.data import BatchNeighborList

def sinc_expansion(edge_dist: torch.Tensor, edge_size: int, cutoff: float):
    """
    calculate sinc radial basis function:
    
    sin(n *pi*d/d_cut)/d
    """
    n = torch.arange(edge_size, device=edge_dist.device) + 1
    return torch.sin(edge_dist.unsqueeze(-1) * n * torch.pi / cutoff) / edge_dist.unsqueeze(-1)

def cosine_cutoff(edge_dist: torch.Tensor, cutoff: float):
    """
    Calculate cutoff value based on distance.
    This uses the cosine Behler-Parinello cutoff function:

    f(d) = 0.5*(cos(pi*d/d_cut)+1) for d < d_cut and 0 otherwise
    """

    return torch.where(
        edge_dist < cutoff,
        0.5 * (torch.cos(torch.pi * edge_dist / cutoff) + 1),
        torch.tensor(0.0, device=edge_dist.device, dtype=edge_dist.dtype),
    )

class PainnMessage(nn.Module):
    """Message function"""
    def __init__(self, node_size: int, edge_size: int, cutoff: float):
        super().__init__()
        
        self.edge_size = edge_size
        self.node_size = node_size
        self.cutoff = cutoff
        
        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        
        self.filter_layer = nn.Linear(edge_size, node_size * 3)
        
    def forward(self, node_scalar, node_vector, edge, edge_diff, edge_dist):
        # remember to use v_j, s_j but not v_i, s_i        
        filter_weight = self.filter_layer(sinc_expansion(edge_dist, self.edge_size, self.cutoff))
        filter_weight = filter_weight * cosine_cutoff(edge_dist, self.cutoff).unsqueeze(-1)
        scalar_out = self.scalar_message_mlp(node_scalar)        
        filter_out = filter_weight * scalar_out[edge[:, 1]]
        
        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out, 
            self.node_size,
            dim = 1,
        )
        
        # num_pairs * 3 * node_size, num_pairs * node_size
        message_vector =  node_vector[edge[:, 1]] * gate_state_vector.unsqueeze(1) 
        edge_vector = gate_edge_vector.unsqueeze(1) * (edge_diff / edge_dist.unsqueeze(-1)).unsqueeze(-1)
        message_vector = message_vector + edge_vector
        
        # sum message
        residual_scalar = torch.zeros_like(node_scalar)
        residual_vector = torch.zeros_like(node_vector)
        residual_scalar.index_add_(0, edge[:, 0], message_scalar)
        residual_vector.index_add_(0, edge[:, 0], message_vector)
        
        # new node state
        new_node_scalar = node_scalar + residual_scalar
        new_node_vector = node_vector + residual_vector
        
        return new_node_scalar, new_node_vector

class PainnUpdate(nn.Module):
    """Update function"""
    def __init__(self, node_size: int):
        super().__init__()
        
        self.update_U = nn.Linear(node_size, node_size)
        self.update_V = nn.Linear(node_size, node_size)
        
        self.update_mlp = nn.Sequential(
            nn.Linear(node_size * 2, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        
    def forward(self, node_scalar, node_vector):
        Uv = self.update_U(node_vector)
        Vv = self.update_V(node_vector)
        
        Vv_norm = torch.linalg.norm(Vv, dim=1)
        mlp_input = torch.cat((Vv_norm, node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)
        
        a_vv, a_sv, a_ss = torch.split(
            mlp_output,                                        
            node_vector.shape[-1],                                       
            dim = 1,
        )
        
        delta_v = a_vv.unsqueeze(1) * Uv
        inner_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_sv * inner_prod + a_ss
        
        return node_scalar + delta_s, node_vector + delta_v

class PainnModel(nn.Module):
    """PainnModel without edge updating"""
    def __init__(
        self, 
        num_interactions: int, 
        node_size: int, 
        cutoff: float,
        normalization: bool=True,
        target_mean: List[float]=[0.0],
        target_stddev: List[float]=[1.0],
        atomwise_normalization: bool=True,
        compute_neighborlist: bool=False,
        **kwargs,
    ):
        super().__init__()
        
        num_embedding = 119   # number of all elements
        self.cutoff = cutoff
        self.num_interactions = num_interactions
        self.node_size = node_size
        self.edge_size = 20
        self.compute_neighborlist = compute_neighborlist
        if self.compute_neighborlist:
            self.batch_nl = BatchNeighborList(self.cutoff, requires_grad=True, wrap_atoms=True)
            
        # Setup atom embeddings
        self.atom_embedding = nn.Embedding(num_embedding, node_size)

        # Setup message-passing layers
        self.message_layers = nn.ModuleList(
            [
                PainnMessage(self.node_size, self.edge_size, self.cutoff)
                for _ in range(self.num_interactions)
            ]
        )
        self.update_layers = nn.ModuleList(
            [
                PainnUpdate(self.node_size)
                for _ in range(self.num_interactions)
            ]            
        )
        
        # Setup readout function
        self.readout_mlp = nn.Sequential(
            nn.Linear(self.node_size, self.node_size),
            nn.SiLU(),
            nn.Linear(self.node_size, 1),
        )

        # Normalisation constants
        self.register_buffer("normalization", torch.tensor(normalization))
        self.register_buffer("atomwise_normalization", torch.tensor(atomwise_normalization))
        self.register_buffer("normalize_stddev", torch.tensor(target_stddev[0]))
        self.register_buffer("normalize_mean", torch.tensor(target_mean[0]))

    def forward(self, input_dict: Dict[str, torch.Tensor], compute_forces: bool=True) -> Dict[str, torch.Tensor]:
        num_atoms = input_dict['num_atoms']
        total_atoms = int(torch.sum(num_atoms))
        # torch neighbor list enables gradient calculation, and versatile cutoff choices
        if self.compute_neighborlist:
            edge, edge_diff, edge_dist = self.batch_nl(input_dict)
        else:
            num_pairs = input_dict['num_pairs']
            edge = input_dict['pairs']
            edge_diff = input_dict['n_diff']
            total_atoms = int(torch.sum(num_atoms))

            # edge offset. Add offset to edges to get indices of pairs in a batch but not a structure
            edge_offset = torch.zeros_like(num_atoms)
            edge_offset[1:] = num_atoms[:-1]
            edge_offset = torch.cumsum(edge_offset, dim=0)
            edge_offset = torch.repeat_interleave(edge_offset, num_pairs)
            edge = edge + edge_offset.unsqueeze(-1)        

            if compute_forces:
                edge_diff.requires_grad_()
            edge_dist = torch.linalg.norm(edge_diff, dim=1)
        
        node_scalar = self.atom_embedding(input_dict['elems'])
        node_vector = torch.zeros((total_atoms, 3, self.node_size),
                                  device=edge_diff.device,
                                  dtype=edge_diff.dtype,
                                 )
        
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            node_scalar, node_vector = message_layer(node_scalar, node_vector, edge, edge_diff, edge_dist)
            node_scalar, node_vector = update_layer(node_scalar, node_vector)
        
        node_scalar = self.readout_mlp(node_scalar)
        node_scalar.squeeze_()

        image_idx = torch.arange(input_dict['num_atoms'].shape[0],
                                 device=edge.device,
                                )
        image_idx = torch.repeat_interleave(image_idx, num_atoms)
        
        energy = torch.zeros_like(input_dict['num_atoms'], dtype=edge_diff.dtype)  
        energy.index_add_(0, image_idx, node_scalar)

        # Apply (de-)normalization
        if self.normalization:
            normalizer = self.normalize_stddev
            energy = normalizer * energy
            mean_shift = self.normalize_mean
            if self.atomwise_normalization:
                mean_shift = input_dict["num_atoms"] * mean_shift
            energy = energy + mean_shift

        result_dict = {'energy': energy}
        
        if compute_forces:
            grad_outputs : List[Optional[torch.Tensor]] = [torch.ones_like(energy)]    # for model deploy
            dE_ddiff = torch.autograd.grad(
                [energy,],
                [edge_diff,],
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=True,
            )
            dE_ddiff = torch.zeros_like(edge_diff) if dE_ddiff is None else dE_ddiff[0]   # for torch.jit.script
            assert dE_ddiff is not None
            
            # diff = R_j - R_i, so -dE/dR_j = -dE/ddiff, -dE/R_i = dE/ddiff
            i_forces = torch.zeros((total_atoms, 3), device=edge_diff.device, dtype=edge_diff.dtype)
            j_forces = torch.zeros_like(i_forces)
            i_forces.index_add_(0, edge[:, 0], dE_ddiff)
            j_forces.index_add_(0, edge[:, 1], -dE_ddiff)
            forces = i_forces + j_forces
            
            result_dict['forces'] = forces
            
        return result_dict

class PainnEnsemble(nn.Module):
    """
    Ensemble model of curator. This is used to get uncertainty informations.
    """
    def __init__(self, models: List[PainnModel]) -> None:
        super().__init__()
        self.models = nn.ModuleList([model for model in models])
        self.cutoff = models[0].cutoff

    def forward(self, input_dict: Dict[str, torch.Tensor], compute_forces: bool=True) -> Dict[str, torch.Tensor]:
        energy = []
        forces = []
        for model in self.models:
            out = model(input_dict, compute_forces)
            energy.append(out['energy'].detach())
            forces.append(out['forces'].detach())
        
        image_idx = torch.arange(
            input_dict['num_atoms'].shape[0],
            device=out['energy'].device,
        )
        image_idx = torch.repeat_interleave(image_idx, input_dict['num_atoms'])
        
        energy = torch.stack(energy)
        forces = torch.stack(forces)
        f_scatter = torch.zeros(input_dict['num_atoms'].shape[0], device=out['energy'].device)
        result_dict ={
            'energy': torch.mean(energy, dim=0),
            'forces': torch.mean(forces, dim=0),
            'e_var': torch.var(energy, dim=0),
            'e_sd': torch.std(energy, dim=0),
            'f_var': f_scatter.index_add(0, image_idx, torch.var(forces, dim=0).mean(dim=1)) / input_dict['num_atoms'],
        }
        
        result_dict['f_sd'] = result_dict['f_var'].sqrt()
        if 'energy' in input_dict.keys():
            e_diff = result_dict['energy'] - input_dict['energy']
            f_diff = result_dict['forces'] - input_dict['forces']
            result_dict['e_ae'] = torch.abs(e_diff)
            result_dict['e_se'] = torch.square(e_diff)
            result_dict['f_ae'] = f_scatter.index_add(0, image_idx, torch.abs(f_diff).mean(dim=1)) / input_dict['num_atoms']
            result_dict['f_se'] = f_scatter.index_add(0, image_idx, torch.square(f_diff).mean(dim=1)) / input_dict['num_atoms']

        return result_dict   
