import torch
import math
from torch import nn
from e3nn import o3
from curator.data import properties
from e3nn.util.jit import compile_mode
from .cutoff import PolynomialCutoff

class SineBasis(torch.nn.Module):
    """
    calculate sinc radial basis function:
    
    sin(n *pi*d/d_cut)/d
    """
    def __init__(self, num_basis: int, cutoff: float):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        
    def forward(self, edge_dist: torch.Tensor) -> torch.Tensor:
        n = torch.arange(self.num_basis, device=edge_dist.device) + 1
        radial_basis = torch.sin(edge_dist.unsqueeze(-1) * n * torch.pi / self.cutoff) / edge_dist.unsqueeze(-1)
        return radial_basis
        
def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs.unsqueeze(-1) - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y

class GaussianBasis(torch.nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super().__init__()
        self.n_rbf = n_rbf

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

class BesselBasis(torch.nn.Module):
    cutoff: float
    prefactor: float

    def __init__(self, cutoff: float, num_basis: int=8, trainable: bool=True):
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
        self.prefactor = 2.0 / self.cutoff
        # output edge dist irreps
        self.irreps_out = o3.Irreps([(num_basis, (0, 1))])

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
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
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.cutoff)

        return self.prefactor * (numerator / x.unsqueeze(-1))
            
@compile_mode("script")
class RadialBasisEdgeEncoding(torch.nn.Module):
    out_field: str

    def __init__(
        self,
        cutoff,
        basis=BesselBasis,
        cutoff_fn=PolynomialCutoff,
        basis_kwargs={},
        cutoff_kwargs={},
    ):
        super().__init__()
        basis_kwargs['cutoff'] = cutoff
        cutoff_kwargs['cutoff'] = cutoff
        self.basis = basis(**basis_kwargs)
        self.cutoff_fn = cutoff_fn(**cutoff_kwargs)
        
        # output edge dist irreps
        self.irreps_out = self.basis.irreps_out

    def forward(self, data: properties.Type) -> properties.Type:
        edge_dist = data[properties.edge_dist]
        data[properties.edge_dist_embedding] = (
            self.basis(edge_dist) * self.cutoff_fn(edge_dist)[:, None]
        )
        
        return data

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