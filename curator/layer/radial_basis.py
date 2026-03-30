import torch
import math
from torch import nn
from e3nn import o3
from curator.data import properties
from e3nn.util.jit import compile_mode
from .cutoff import CutoffFunction, PolynomialCutoff
import abc
import ase.data
from typing import Optional

class RadialBasis(torch.nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self):
        pass


class SineBasis(RadialBasis):
    """
    calculate sinc radial basis function:
    
    sin(n *pi*d/d_cut)/d
    """
    def __init__(self, cutoff: float, num_basis: int):
        super().__init__()
        self.cutoff = cutoff
        self.num_basis = num_basis
        self.irreps_out = o3.Irreps([(num_basis, (0, 1))])
        
    def forward(self, edge_dist: torch.Tensor) -> torch.Tensor:
        n = torch.arange(self.num_basis, device=edge_dist.device) + 1
        radial_basis = torch.sin(edge_dist.unsqueeze(-1) * n * torch.pi / self.cutoff) / edge_dist.unsqueeze(-1)
        return radial_basis
        
def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs.unsqueeze(-1) - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y

class GaussianBasis(RadialBasis):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, cutoff: float, n_rbf: int, start: float = 0.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`mu_{N_g}`
            start: center of first Gaussian function, :math:`mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super().__init__()
        self.cutoff = cutoff
        self.n_rbf = n_rbf
        self.irreps_out = o3.Irreps([(n_rbf, (0, 1))])

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)

class BesselBasis(RadialBasis):
    cutoff: float
    prefactor: float

    def __init__(self, cutoff: float, num_basis: int=8, trainable: bool=False, sqrt_prefactor: bool=False):
        r"""Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        cutoff : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.cutoff = float(cutoff)
        self.prefactor = math.sqrt(2.0 / self.cutoff) if sqrt_prefactor else 2.0 / self.cutoff
        # output edge dist irreps
        self.irreps_out = o3.Irreps([(num_basis, (0, 1))])

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi / self.cutoff
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1))

        return self.prefactor * (numerator / x.unsqueeze(-1))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(cutoff={self.cutoff}, num_basis={len(self.bessel_weights)}, prefactor={self.prefactor}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )
            
@compile_mode("script")
class RadialBasisEdgeEncoding(torch.nn.Module):
    out_field: str

    def __init__(
        self,
        basis: RadialBasis,
        cutoff_fn: CutoffFunction,
        distance_transform: Optional[nn.Module] = None,
        atomic_numbers: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.basis = basis
        self.cutoff_fn = cutoff_fn
        # Optional MACE-style distance preprocessing (Agnesi/Soft); requires atomic numbers.
        self.distance_transform = distance_transform
        if atomic_numbers is not None:
            self.register_buffer(
                "atomic_numbers",
                torch.as_tensor(atomic_numbers, dtype=torch.long),
            )
        else:
            self.atomic_numbers = None
        
        # output edge dist irreps
        self.irreps_out = self.basis.irreps_out

    def forward(self, data: properties.Type) -> properties.Type:
        edge_dist = data[properties.edge_dist]
        cutoff = self.cutoff_fn(edge_dist)
        edge_dist_for_basis = edge_dist
        distance_transform = getattr(self, "distance_transform", None)
        atomic_numbers = getattr(self, "atomic_numbers", None)
        if distance_transform is not None:
            if atomic_numbers is None:
                raise ValueError("distance_transform requires atomic_numbers to be set.")
            if edge_dist_for_basis.dim() == 1:
                edge_dist_for_basis = edge_dist_for_basis.unsqueeze(-1)
            edge_dist_for_basis = distance_transform(
                edge_dist_for_basis,
                data[properties.node_attr],
                data[properties.edge_idx],
                atomic_numbers.to(edge_dist_for_basis.device),
            )
        if edge_dist_for_basis.dim() > 1 and edge_dist_for_basis.shape[-1] == 1:
            edge_dist_for_basis = edge_dist_for_basis.squeeze(-1)
        data[properties.edge_dist_embedding] = self.basis(edge_dist_for_basis) * cutoff[:, None]
        
        return data


@compile_mode("script")
class AgnesiTransform(torch.nn.Module):
    """Agnesi transform used for radial distance preprocessing."""

    def __init__(
        self,
        q: float = 0.9183,
        p: float = 4.5791,
        a: float = 1.0805,
        trainable: bool = False,
    ):
        super().__init__()
        self.register_buffer("q", torch.tensor(q, dtype=torch.get_default_dtype()))
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer("a", torch.tensor(a, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        if trainable:
            self.a = torch.nn.Parameter(torch.tensor(1.0805, requires_grad=True))
            self.q = torch.nn.Parameter(torch.tensor(0.9183, requires_grad=True))
            self.p = torch.nn.Parameter(torch.tensor(4.5791, requires_grad=True))

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.dim() == 2 and edge_index.shape[0] == 2:
            sender = edge_index[0]
            receiver = edge_index[1]
        else:
            sender = edge_index[:, 0]
            receiver = edge_index[:, 1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[sender].to(torch.int64)
        Z_v = node_atomic_numbers[receiver].to(torch.int64)
        r_0: torch.Tensor = 0.5 * (self.covalent_radii[Z_u] + self.covalent_radii[Z_v])
        r_over_r_0 = x / r_0
        return (
            1
            + (
                self.a
                * torch.pow(r_over_r_0, self.q)
                / (1 + torch.pow(r_over_r_0, self.q - self.p))
            )
        ).reciprocal_()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(a={self.a:.4f}, q={self.q:.4f}, p={self.p:.4f})"
        )


@compile_mode("script")
class SoftTransform(torch.nn.Module):
    """Tanh-based smooth distance transformation."""

    def __init__(self, alpha: float = 4.0, trainable: bool = False):
        super().__init__()
        self.register_buffer(
            "alpha", torch.tensor(alpha, dtype=torch.get_default_dtype())
        )
        if trainable:
            self.alpha = torch.nn.Parameter(self.alpha.clone())
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )

    def compute_r_0(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.dim() == 2 and edge_index.shape[0] == 2:
            sender = edge_index[0]
            receiver = edge_index[1]
        else:
            sender = edge_index[:, 0]
            receiver = edge_index[:, 1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[sender].to(torch.int64)
        Z_v = node_atomic_numbers[receiver].to(torch.int64)
        r_0: torch.Tensor = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        return r_0

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        r_0 = self.compute_r_0(node_attrs, edge_index, atomic_numbers)
        p_0 = (3 / 4) * r_0
        p_1 = (4 / 3) * r_0
        m = 0.5 * (p_0 + p_1)
        alpha = self.alpha / (p_1 - p_0)
        s_x = 0.5 * (1.0 + torch.tanh(alpha * (x - m)))
        return p_0 + (x - p_0) * s_x

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha.item():.4f})"

class SphericalHarmonicEdgeAttrs(torch.nn.Module):
    def __init__(
        self,
        edge_sh_irreps: o3.Irreps,
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
    ):
        super().__init__()
        
        self.edge_sh_irreps = edge_sh_irreps
        self.sh = o3.SphericalHarmonics(
            self.edge_sh_irreps, edge_sh_normalize, edge_sh_normalization
        )
        # output edge diff irreps
        self.irreps_out = edge_sh_irreps

    def forward(self, data: properties.Type) -> properties.Type:
        data[properties.edge_diff_embedding] = self.sh(
            data[properties.edge_diff]
        )
        return data
