import torch
from torch import nn
from e3nn.util.jit import compile_mode
from ase import data as ase_data
from curator.data import properties

try:
    from torch_scatter import scatter_add
except ImportError:  # pragma: no cover
    from curator.utils import scatter_add


@torch.jit.script
def _poly_envelope(x: torch.Tensor, r_max: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    r_over_r_max = x / r_max
    envelope = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(r_over_r_max, p)
        + p * (p + 2.0) * torch.pow(r_over_r_max, p + 1)
        - (p * (p + 1.0) / 2.0) * torch.pow(r_over_r_max, p + 2)
    )
    return envelope * (x < r_max)


@compile_mode("script")
class ZBLBasis(nn.Module):
    """Ziegler-Biersack-Littmark (ZBL) pair repulsion with polynomial cutoff."""

    p: torch.Tensor

    def __init__(self, p: int = 6, trainable: bool = False):
        super().__init__()
        self.register_buffer(
            "c",
            torch.tensor([0.1818, 0.5099, 0.2802, 0.02817], dtype=torch.get_default_dtype()),
        )
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(ase_data.covalent_radii, dtype=torch.get_default_dtype()),
        )
        if trainable:
            self.a_exp = nn.Parameter(torch.tensor(0.300, requires_grad=True))
            self.a_prefactor = nn.Parameter(torch.tensor(0.4543, requires_grad=True))
        else:
            self.register_buffer("a_exp", torch.tensor(0.300))
            self.register_buffer("a_prefactor", torch.tensor(0.4543))

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if edge_index.dim() == 2 and edge_index.shape[0] == 2:
            sender = edge_index[0]
            receiver = edge_index[1]
        else:
            sender = edge_index[:, 0]
            receiver = edge_index[:, 1]

        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(-1)
        Z_u = node_atomic_numbers[sender].to(torch.int64)
        Z_v = node_atomic_numbers[receiver].to(torch.int64)
        a = self.a_prefactor * 0.529 / (torch.pow(Z_u, self.a_exp) + torch.pow(Z_v, self.a_exp))
        r_over_a = x / a
        phi = (
            self.c[0] * torch.exp(-3.2 * r_over_a)
            + self.c[1] * torch.exp(-0.9423 * r_over_a)
            + self.c[2] * torch.exp(-0.4028 * r_over_a)
            + self.c[3] * torch.exp(-0.2016 * r_over_a)
        )
        v_edges = (14.3996 * Z_u * Z_v) / x * phi
        r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        envelope = _poly_envelope(x, r_max, self.p)
        v_edges = 0.5 * v_edges * envelope
        v_edges = v_edges.squeeze(-1) if v_edges.dim() > 1 and v_edges.shape[-1] == 1 else v_edges
        v_nodes = scatter_add(v_edges, receiver, dim=0, dim_size=node_attrs.size(0))
        return v_nodes.squeeze(-1) if v_nodes.dim() > 1 and v_nodes.shape[-1] == 1 else v_nodes

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(c={self.c})"


class PairRepulsionEnergy(nn.Module):
    """Add ZBL pair-repulsion energy to the total energy before scale/shift."""

    def __init__(self, pair_fn: nn.Module, atomic_numbers: torch.Tensor):
        super().__init__()
        self.pair_fn = pair_fn
        self.register_buffer("atomic_numbers", torch.as_tensor(atomic_numbers, dtype=torch.long))

    def forward(self, data: properties.Type) -> properties.Type:
        if properties.edge_dist not in data or properties.edge_idx not in data:
            return data
        if properties.node_attr not in data:
            return data
        if properties.image_idx not in data:
            data[properties.image_idx] = torch.zeros(
                data[properties.n_atoms].item(),
                dtype=data[properties.edge_idx].dtype,
                device=data[properties.edge_idx].device,
            )

        pair_node_energy = self.pair_fn(
            data[properties.edge_dist],
            data[properties.node_attr],
            data[properties.edge_idx],
            self.atomic_numbers.to(data[properties.edge_dist].device),
        )
        if pair_node_energy.dim() > 1 and pair_node_energy.shape[-1] == 1:
            pair_node_energy = pair_node_energy.squeeze(-1)
        pair_energy = scatter_add(pair_node_energy, data[properties.image_idx], dim=0)
        if properties.energy in data:
            data[properties.energy] = data[properties.energy] + pair_energy
        else:
            data[properties.energy] = pair_energy
        return data
