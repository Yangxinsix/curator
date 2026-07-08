import torch
from torch import nn
from e3nn import o3
from curator.data import properties

try:
    from torch_scatter import scatter_add
except ImportError:  # pragma: no cover
    from curator.utils import scatter_add


class MACEReadoutAdapter(nn.Module):
    """Adapter to reuse MACE readout blocks with Curator's concatenated node features."""

    def __init__(
        self,
        readouts: nn.ModuleList,
        num_interactions: int,
        hidden_irreps: o3.Irreps,
        head_idx: int | None = None,
    ):
        super().__init__()
        self.readouts = nn.ModuleList(list(readouts))
        self.num_interactions = num_interactions
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.head_idx = head_idx

        self.in_features_list = [self.hidden_irreps.dim] * (num_interactions - 1)
        self.in_features_list.append(self.hidden_irreps[0].dim)
        invariant_dim = int(sum(mul * ir.dim for mul, ir in self.hidden_irreps if ir.l == 0 and ir.p == 1))
        self.invariant_features_list = [invariant_dim for _ in range(num_interactions)]

    def forward(self, data: properties.Type) -> properties.Type:
        node_feat = data[properties.node_feat]
        node_feat_list = torch.split(node_feat, self.in_features_list, dim=-1)
        node_heads = None
        if self.head_idx is not None:
            node_heads = torch.full(
                (node_feat.shape[0],),
                self.head_idx,
                dtype=torch.long,
                device=node_feat.device,
            )

        node_es_list = []
        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            if node_heads is not None and getattr(readout, "num_heads", 1) > 1:
                node_es = readout(node_feat_list[feat_idx], node_heads)
                if node_es.dim() > 1:
                    node_es = node_es[torch.arange(node_es.shape[0], device=node_es.device), node_heads]
            else:
                node_es = readout(node_feat_list[feat_idx])
                if node_es.dim() > 1 and self.head_idx is not None:
                    node_es = node_es[:, self.head_idx]
            node_es_list.append(node_es.squeeze(-1))

        node_energy = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        if properties.image_idx not in data:
            data[properties.image_idx] = torch.zeros(
                node_energy.shape[0],
                dtype=torch.long,
                device=node_energy.device,
            )
        data[properties.energy] = scatter_add(node_energy, data[properties.image_idx], dim=0)
        return data
