import torch
from torch import nn
from e3nn import o3
from e3nn.util.jit import compile_mode

from curator.data import properties

from curator.layer import (
    OneHotAtomEncoding,
    AtomwiseLinear,
    RadialBasisEdgeEncoding,
    BesselBasis,
    PolynomialCutoff,
    SphericalHarmonicEdgeAttrs,
    InteractionLayer,
)

from typing import OrderedDict, Dict, List, Optional, Union, Callable

from e3nn.util.jit import compile_mode
class NequipModel(torch.nn.Module):
    """Nequip model."""
    def __init__(
        self,
        cutoff: float,
        num_interactions: int,
        species: Optional[List[str]] = None,
        num_elements: Optional[int] = None,
        hidden_irreps: Union[o3.Irreps, str, None] = None,
        edge_sh_irreps: Union[o3.Irreps, str, None] = None,
        node_irreps: Union[o3.Irreps, str, None] = None,
        MLP_irreps: Union[o3.Irreps, str, None] = None,
        lmax: int = 2,
        parity: bool = True,
        num_features: Optional[int] = None,
        num_basis: int = 8,
        power: int = 6,
        # parameters for interaction blocks and convnet
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "ssp", "o": "abs"},
        convolution_kwargs: dict = {},
        **kwargs,
    ) -> None:
        """Nequip model.

        Args:
            cutoff (float): Cutoff radius
            num_interactions (int): Number of interaction blocks
            species (List[str]): List of species
            num_elements (Optional[int], optional): Number of elements. Defaults to None.
            hidden_irreps (Union[o3.Irreps, str, None], optional): Hidden irreps. Defaults to None.
            edge_sh_irreps (Union[o3.Irreps, str, None], optional): Edge irreps. Defaults to None.
            node_irreps (Union[o3.Irreps, str, None], optional): Node irreps. Defaults to None.
            MLP_irreps (Union[o3.Irreps, str, None], optional): MLP irreps. Defaults to None.
            lmax (int, optional): Maximum l value for spherical harmonics. Defaults to 2.
            parity (bool, optional): Parity. Defaults to True.
            num_features (Optional[int], optional): Number of features. Defaults to None.
            num_basis (int, optional): Number of basis. Defaults to 8.
            power (int, optional): Power of radial basis. Defaults to 6.
            resnet (bool, optional): ResNet. Defaults to False.
            nonlinearity_type (str, optional): Type of nonlinearity. Defaults to "gate".
            nonlinearity_scalars (Dict[int, Callable], optional): Nonlinearity for scalars. Defaults to {"e": "ssp", "o": "tanh"}.
            nonlinearity_gates (Dict[int, Callable], optional): Nonlinearity for gates. Defaults to {"e": "ssp", "o": "abs"}.
            convolution_kwargs (dict, optional): Convolution kwargs. Defaults to {}.
        """
        super().__init__()
        self.cutoff = cutoff
        self.num_features = num_features
        self.lmax = lmax
        self.parity = parity
        
        if num_elements is None:
            num_elements = len(species) if species is not None else 119
        
        ## handling irreps
        # chemical embedding irreps
        if node_irreps is None:
            self.node_irreps = o3.Irreps([(num_features, (0, 1))])
        elif isinstance(node_irreps, str):
            self.node_irreps = o3.Irreps(node_irreps)
        else:
            self.node_irreps = node_irreps
        # edge sphere harmonic irreps
        if edge_sh_irreps is None:
            self.edge_sh_irreps = o3.Irreps.spherical_harmonics(lmax, p=-1 if parity else 1)
        elif isinstance(edge_sh_irreps, str):
            self.edge_sh_irreps = o3.Irreps(edge_sh_irreps)
        else:
            self.edge_sh_irreps = edge_sh_irreps
        # hidden feature irreps
        if hidden_irreps is None:
            self.hidden_irreps = o3.Irreps(
                [
                    (num_features, (l, p))
                    for p in ((1, -1) if parity else (1,))
                    for l in range(lmax + 1)
                ]
            )
        elif isinstance(hidden_irreps, str):
            self.hidden_irreps = o3.Irreps(hidden_irreps)
        else:
            self.hidden_irreps = hidden_irreps
        # MLP_irreps
        if MLP_irreps is None:
            self.MLP_irreps = o3.Irreps([(max(1, num_features // 2), (0, 1))])
        elif isinstance(MLP_irreps, str):
            self.MLP_irreps = o3.Irreps(MLP_irreps)
        else:
            self.MLP_irreps = MLP_irreps
        
        self.embeddings = nn.ModuleDict()
        self.embeddings['onehot_embedding'] = OneHotAtomEncoding(num_elements=num_elements, species=species)
        self.embeddings['radial_basis'] = RadialBasisEdgeEncoding(
            basis=BesselBasis(cutoff=cutoff, num_basis=num_basis),
            cutoff_fn=PolynomialCutoff(cutoff=cutoff, power=power),
        )
        self.embeddings['sphere_harmonics'] = SphericalHarmonicEdgeAttrs(edge_sh_irreps=self.edge_sh_irreps)
        
        self.irreps_in = {
            properties.edge_diff_embedding: self.embeddings.sphere_harmonics.irreps_out,
            properties.edge_dist_embedding: self.embeddings.radial_basis.irreps_out,
        }
        self.irreps_in.update(self.embeddings.onehot_embedding.irreps_out)
        
        self.embeddings['chemical_embedding'] = AtomwiseLinear(
            irreps_in=self.irreps_in[properties.node_attr],
            irreps_out=self.node_irreps,
        )
        self.irreps_in[properties.node_feat] = self.embeddings.chemical_embedding.irreps_out
        
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            interaction = InteractionLayer(
                irreps_in=self.irreps_in, 
                feature_irreps_hidden=self.hidden_irreps,
                convolution_kwargs=convolution_kwargs,
                resnet=resnet,
                nonlinearity_type=nonlinearity_type,
                nonlinearity_scalars=nonlinearity_scalars,
                nonlinearity_gates=nonlinearity_gates,
            )
            self.interactions.append(interaction)
            self.irreps_in.update(interaction.irreps_out)
        
        self.readout_mlp = nn.Sequential(
            o3.Linear(
                irreps_in=self.irreps_in[properties.node_feat],
                irreps_out=self.MLP_irreps,
            ),
            o3.Linear(
                irreps_in=self.MLP_irreps, 
                irreps_out=o3.Irreps('1x0e'),
            ),
        )
        
    def forward(self, data: properties.Type) -> properties.Type:        
        for m in self.embeddings.values():
            data = m(data)
            
        for m in self.interactions:
            data = m(data)
        
        data[properties.atomic_energy] = self.readout_mlp(data[properties.node_feat]).squeeze()
        return data