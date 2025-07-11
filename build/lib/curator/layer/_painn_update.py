import torch
from torch import nn
from curator.data import properties

class PainnUpdate(nn.Module):
    """Update function"""
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

        self.update_U = nn.Linear(num_features, num_features)
        self.update_V = nn.Linear(num_features, num_features)
        
        self.update_mlp = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 3),
        )
        
    def forward(
        self, 
        node_feat: torch.Tensor,
    ) -> torch.Tensor:
        node_scalar, node_vector = torch.split(node_feat, [self.num_features * 1, self.num_features * 3], dim=1)
        node_vector = node_vector.reshape([-1, 3, self.num_features])

        Uv = self.update_U(node_vector)
        Vv = self.update_V(node_vector)
        
        Vv_norm = torch.linalg.norm(Vv, dim=1)
        mlp_input = torch.cat((Vv_norm, node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)
        
        a_vv, a_sv, a_ss = torch.split(
            mlp_output,                                        
            self.num_features,                                       
            dim = 1,
        )
        
        delta_v = a_vv.unsqueeze(1) * Uv
        inner_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_sv * inner_prod + a_ss
        
        residual_node_feat = torch.cat([delta_s, delta_v.reshape(-1, self.num_features * 3)], dim=1)
        node_feat += residual_node_feat
        
        return node_feat