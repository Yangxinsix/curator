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

    def __init__(
        self,
        p: int = 6,
        trainable: bool = False,
        screening_exponent: float = 0.300,
        screening_length: float = 0.4543 * 0.529,
        phi_coefficients: tuple[float, float, float, float] = (0.1818, 0.5099, 0.2802, 0.02817),
        phi_exponents: tuple[float, float, float, float] = (3.2, 0.9423, 0.4028, 0.2016),
        energy_prefactor: float = 14.3996 * 0.5,
        cutoff: float | None = None,
        cutoff_by_species: bool = True,
        scatter_to: str = "receiver",
    ):
        super().__init__()
        self.register_buffer(
            "c",
            torch.tensor(phi_coefficients, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "phi_exponents",
            torch.tensor(phi_exponents, dtype=torch.get_default_dtype()),
        )
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "energy_prefactor",
            torch.as_tensor(float(energy_prefactor), dtype=torch.float64),
        )
        self.register_buffer(
            "covalent_radii",
            torch.tensor(ase_data.covalent_radii, dtype=torch.get_default_dtype()),
        )
        self.cutoff_by_species = bool(cutoff_by_species)
        self.scatter_to = str(scatter_to)
        if self.scatter_to not in {"receiver", "center"}:
            raise ValueError(f"Unsupported scatter_to={scatter_to!r}; expected 'receiver' or 'center'.")
        if cutoff is None:
            self.cutoff = None
        else:
            self.register_buffer(
                "cutoff",
                torch.as_tensor(float(cutoff), dtype=torch.get_default_dtype()),
            )
        if trainable:
            self.screening_exponent = nn.Parameter(torch.tensor(float(screening_exponent), requires_grad=True))
            self.screening_length = nn.Parameter(torch.tensor(float(screening_length), requires_grad=True))
        else:
            self.register_buffer("screening_exponent", torch.tensor(float(screening_exponent)))
            self.register_buffer("screening_length", torch.tensor(float(screening_length)))

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
            edge_center = edge_index[0]
            edge_neighbor = edge_index[1]
        else:
            edge_center = edge_index[:, 0]
            edge_neighbor = edge_index[:, 1]
        if node_attrs.dim() > 1:
            species_index = torch.argmax(node_attrs, dim=1)
        else:
            species_index = node_attrs.reshape(-1).to(torch.long)
        node_atomic_numbers = atomic_numbers[species_index].reshape(-1, 1)
        Z_u = node_atomic_numbers[edge_center].to(x.dtype)
        Z_v = node_atomic_numbers[edge_neighbor].to(x.dtype)
        screening_exponent = getattr(self, "screening_exponent", None)
        if screening_exponent is None:
            screening_exponent = torch.tensor(0.300, device=x.device)
        screening_length = getattr(self, "screening_length", None)
        if screening_length is None:
            screening_length = torch.tensor(0.4543 * 0.529, device=x.device)
        phi_exponents = getattr(self, "phi_exponents", None)
        if phi_exponents is None:
            phi_exponents = torch.tensor((3.2, 0.9423, 0.4028, 0.2016), device=x.device)
        energy_prefactor = getattr(self, "energy_prefactor", None)
        if energy_prefactor is None:
            energy_prefactor = torch.tensor(14.3996 * 0.5, device=x.device)
        covalent_radii = getattr(self, "covalent_radii", None)
        if covalent_radii is None:
            covalent_radii = torch.tensor(ase_data.covalent_radii, device=x.device)
        screening_arg = (
            torch.pow(Z_u, screening_exponent.to(x.dtype))
            + torch.pow(Z_v, screening_exponent.to(x.dtype))
        ) * x / screening_length.to(x.dtype)
        phi = (
            self.c[0] * torch.exp(-phi_exponents[0].to(x.dtype) * screening_arg)
            + self.c[1] * torch.exp(-phi_exponents[1].to(x.dtype) * screening_arg)
            + self.c[2] * torch.exp(-phi_exponents[2].to(x.dtype) * screening_arg)
            + self.c[3] * torch.exp(-phi_exponents[3].to(x.dtype) * screening_arg)
        )
        v_edges = energy_prefactor.to(x.dtype) * (Z_u * Z_v) / x * phi
        cutoff_by_species = getattr(self, "cutoff_by_species", True)
        scatter_to = getattr(self, "scatter_to", "receiver")
        if cutoff_by_species:
            r_max = covalent_radii[Z_u.to(torch.int64)] + covalent_radii[Z_v.to(torch.int64)]
        else:
            cutoff = getattr(self, "cutoff", None)
            if cutoff is None:
                raise RuntimeError("ZBLBasis requires `cutoff` when cutoff_by_species=False.")
            r_max = cutoff.to(x.dtype)
        envelope = _poly_envelope(x, r_max, self.p)
        v_edges = v_edges * envelope
        v_edges = v_edges.squeeze(-1) if v_edges.dim() > 1 and v_edges.shape[-1] == 1 else v_edges
        scatter_index = edge_neighbor if scatter_to == "receiver" else edge_center
        v_nodes = scatter_add(v_edges, scatter_index, dim=0, dim_size=node_attrs.size(0))
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
        if properties.atomic_types in data:
            node_attrs = data[properties.atomic_types]
        elif properties.node_attr in data:
            node_attrs = data[properties.node_attr]
        else:
            return data
        if properties.image_idx not in data:
            data[properties.image_idx] = torch.zeros(
                data[properties.n_atoms].item(),
                dtype=data[properties.edge_idx].dtype,
                device=data[properties.edge_idx].device,
            )

        pair_node_energy = self.pair_fn(
            data[properties.edge_dist],
            node_attrs,
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
