import torch
from torch import nn
from curator.data import properties

class PainnUpdate(nn.Module):
    """Update function"""
    def __init__(self, num_features: int):
        super().__init__()
        
        self.update_U = nn.Linear(num_features, num_features)
        self.update_V = nn.Linear(num_features, num_features)
        
        self.update_mlp = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 3),
        )
        
    def forward(self, data: properties.Type) -> properties.Type:
        node_scalar = data[properties.node_feat]
        node_vector = data[properties.node_vect]
        
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
        
        data[properties.node_feat] = node_scalar + delta_s
        data[properties.node_vect] = node_vector + delta_v
        
        return data