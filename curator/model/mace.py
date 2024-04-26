import torch
from torch import nn
from e3nn import o3
from e3nn.nn import Activation
from e3nn.util.jit import compile_mode

from curator.layer import (
    OneHotAtomEncoding,
    AtomwiseLinear,
    AtomwiseNonLinear,
    RadialBasisEdgeEncoding,
    BesselBasis,
    PolynomialCutoff,
    SphericalHarmonicEdgeAttrs,
    RealAgnosticResidualInteractionBlock,
    EquivariantProductBasisBlock,
)
from curator.data import properties
from typing import List, Optional, Dict, Union, Callable

activation_fn = {
    "silu": torch.nn.SiLU(),
    "tanh": torch.tanh,
    "abs": torch.abs,
    "None": None,
}

@compile_mode('script')
class MACE(nn.Module):
    """MACE model."""
    def __init__(
        self,
        cutoff: float,
        num_interactions: int,
        correlation: Union[int, List[int]],
        species: List[str],
        num_elements: Optional[int] = None,
        hidden_irreps: Union[o3.Irreps, str, None] = None,
        edge_sh_irreps: Union[o3.Irreps, str, None] = None,
        node_irreps: Union[o3.Irreps, str, None] = None,
        MLP_irreps: Union[o3.Irreps, str, None] = None,
        avg_num_neighbors: Optional[float] = None,
        lmax: int = 2,
        parity: bool = True,
        num_features: Optional[int] = None,
        num_basis: int = 8,
        power: int = 6,
        gate: Union[str, Callable] = 'silu',
        **kwargs,
    ) -> None:
        """MACE model.

        Args:
            cutoff (float): Cutoff radius
            num_interactions (int): Number of interaction blocks
            correlation (int): Correlation type. 0 for dot product, 1 for cosine similarity
            species (List[str]): List of species
            num_elements (Optional[int], optional): Number of elements. Defaults to None.
            hidden_irreps (Union[o3.Irreps, str, None], optional): Hidden irreps. Defaults to None.
            edge_sh_irreps (Union[o3.Irreps, str, None], optional): Edge irreps. Defaults to None.
            node_irreps (Union[o3.Irreps, str, None], optional): Node irreps. Defaults to None.
            MLP_irreps (Union[o3.Irreps, str, None], optional): MLP irreps. Defaults to None.
            avg_num_neighbors (Optional[float], optional): Average number of neighbors. Defaults to None.
            lmax (int, optional): Maximum l value. Defaults to 2.
            parity (bool, optional): Parity. Defaults to True.
            num_features (Optional[int], optional): Number of features. Defaults to None.
            num_basis (int, optional): Number of radial basis. Defaults to 8.
            power (int, optional): Power of radial basis. Defaults to 6.
            gate (Union[str, Callable], optional): Activation function for gate. Defaults to 'silu'.
        """
        super().__init__()
        
        self.cutoff = cutoff
        self.lmax = lmax
        self.parity = parity
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions

        if num_elements is None:
            num_elements = len(species)
        
        # hidden feature irreps
        if hidden_irreps is not None:
            self.hidden_irreps = o3.Irreps(hidden_irreps) if isinstance(hidden_irreps, str) else hidden_irreps
        else:
            self.hidden_irreps = o3.Irreps(
                [
                    (num_features, (l, p))
                    for p in ((1, -1) if parity else (1,))
                    for l in range(lmax + 1)
                ]
            )
        # MACE prohibits some irreps like 0e, 1e to be used
        forbidden_ir = ['0o', '1e', '2o', '3e', '4o']
        self.hidden_irreps = o3.Irreps([irrep for irrep in self.hidden_irreps if str(irrep.ir) not in forbidden_ir])
        self.num_features = self.hidden_irreps.count(o3.Irrep(0, 1))

        ## handling irreps
        # chemical embedding irreps
        if node_irreps is None:
            self.node_irreps = o3.Irreps([(self.num_features, (0, 1))])
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
        
        # MLP_irreps
        if MLP_irreps is None:
            self.MLP_irreps = o3.Irreps([(max(1, self.num_features // 2), (0, 1))])
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
        
        interaction_irreps = (self.edge_sh_irreps * self.num_features).sort()[0].simplify()
        
        self.interactions = torch.nn.ModuleList()
        self.products = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()
        gate_fn = activation_fn[gate] if isinstance(gate, str) else gate
        # interaction blocks
        for i in range(num_interactions):
            hidden_irreps_out = str(self.hidden_irreps[0]) if i == num_interactions - 1 else self.hidden_irreps
            if i > 0:
                self.irreps_in[properties.node_feat] = self.hidden_irreps
            inter = RealAgnosticResidualInteractionBlock(
                irreps_in=self.irreps_in,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=inter.target_irreps if i == 0 else interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i],
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            
            if i == num_interactions - 1:
                readout = AtomwiseNonLinear(
                    irreps_in=hidden_irreps_out, 
                    MLP_irreps=self.MLP_irreps,
                    gate=gate_fn,
                )
            else:
                readout = o3.Linear(irreps_in=hidden_irreps_out, irreps_out=o3.Irreps('1x0e'))
            self.readouts.append(readout)
            
    def forward(self, data: properties.Type) -> properties.Type:
        # node_e0 = self.reference_energies[data[properties.Z]]
        # e0 = scatter_add(node_e0, data[properties.image_idx], dim_size=data[properties.n_atoms].shape[0])
        for m in self.embeddings.values():
            data = m(data)
        
        node_es_list = []
        node_feat = data[properties.node_feat]
        
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feat, sc = interaction(
                node_feat, 
                data[properties.node_attr],
                data[properties.edge_idx], 
                data[properties.edge_dist_embedding],
                data[properties.edge_diff_embedding],
            )
            node_feat = product(
                node_feats=node_feat,
                sc=sc,
                node_attrs=data[properties.node_attr],
            )
            node_es_list.append(readout(node_feat).squeeze())
        
        node_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        data[properties.atomic_energy] = node_es
        
        return data