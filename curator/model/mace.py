import torch
from torch import nn
from e3nn import o3
from e3nn.nn import Activation
from e3nn.util.jit import compile_mode
from collections import OrderedDict
from functools import partial
from typing import Optional, Any

from curator.layer import (
    OneHotAtomEncoding,
    AtomwiseLinear,
    AtomwiseNN,
    MACEAtomwiseNN,
    AtomwiseNonLinear,
    RadialBasisEdgeEncoding,
    BesselBasis,
    PolynomialCutoff,
    SphericalHarmonicEdgeAttrs,
    RealAgnosticResidualInteractionBlock,
    RealAgnosticInteractionBlock,
    EquivariantProductBasisBlock,
    AgnesiTransform,
    SoftTransform,
)
from curator.data import properties
from typing import Any, List, Optional, Dict, Union, Callable, Type, Literal
from ase.data import atomic_numbers
from curator.model.base import ParameterGroup, Representation, collect_unique_parameters

activation_fn = {
    "silu": torch.nn.SiLU(),
    "tanh": torch.tanh,
    "abs": torch.abs,
    "None": None,
}

@compile_mode('script')
class MACE(Representation):
    """MACE model."""
    def __init__(
        self,
        cutoff: float,
        num_interactions: int,
        correlation: Union[int, List[int]],
        interaction_cls: Type[nn.Module] = RealAgnosticResidualInteractionBlock,
        interaction_cls_first: Type[nn.Module] = RealAgnosticInteractionBlock,
        radial_MLP: Union[List[int], None] = None,
        species: Optional[List[str]] = None,
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
        readout: Union[AtomwiseNN, Type[AtomwiseNN], partial] = MACEAtomwiseNN,
        heads: Optional[list] = None,
        distance_transform: Optional[Union[Literal["agnesi", "soft", "none", ""], nn.Module]] = None,
        filter_forbidden_irreps: bool = True,
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
            num_heads (int, optional): Number of readout heads. When >1, per-head atomic
                energies are exposed at properties.atomic_energy_heads and averaged for
                properties.atomic_energy.
        """
        super().__init__(heads=heads)
        
        self.cutoff = cutoff
        self.parity = parity
        self.species = species
        self.filter_forbidden_irreps = filter_forbidden_irreps

        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions

        if num_elements is None:
            num_elements = len(species) if species is not None else 119
        
        # hidden feature irreps
        if hidden_irreps is not None:
            self.hidden_irreps = o3.Irreps(hidden_irreps) if isinstance(hidden_irreps, str) else hidden_irreps
            self.lmax = self.hidden_irreps.lmax
        else:
            self.hidden_irreps = o3.Irreps(
                [
                    (num_features, (l, p))
                    for p in ((1, -1) if parity else (1,))
                    for l in range(lmax + 1)
                ]
            ).sort()[0].simplify()
            self.lmax = lmax
        # MACE prohibits some irreps like 0e, 1e to be used; allow opt-out for strict conversions.
        if filter_forbidden_irreps:
            forbidden_ir = ['0o', '1e', '2o', '3e', '4o']
            self.hidden_irreps = o3.Irreps(
                [irrep for irrep in self.hidden_irreps if str(irrep.ir) not in forbidden_ir]
            )
        self.num_features = self.hidden_irreps.count(o3.Irrep(0, 1))

        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
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
            self.edge_sh_irreps = o3.Irreps.spherical_harmonics(self.lmax, p=-1 if parity else 1)
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
        if species is not None:
            self.register_buffer(
                "atomic_numbers",
                torch.tensor([atomic_numbers[s] for s in species], dtype=torch.long),
            )
        else:
            self.atomic_numbers = None
        # Resolve distance transform by name for config-friendly usage.
        if isinstance(distance_transform, str):
            name = distance_transform.lower()
            if name in ("none", ""):
                distance_transform = None
            elif name == "agnesi":
                distance_transform = AgnesiTransform()
            elif name == "soft":
                distance_transform = SoftTransform()
            else:
                raise ValueError(f"Unsupported distance_transform '{distance_transform}'")

        self.embeddings['radial_basis'] = RadialBasisEdgeEncoding(
            basis=BesselBasis(cutoff=cutoff, num_basis=num_basis, sqrt_prefactor=True),
            cutoff_fn=PolynomialCutoff(cutoff=cutoff, power=power),
            distance_transform=distance_transform,
            atomic_numbers=self.atomic_numbers,
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
        # interaction blocks
        # for last layer: only select scalar 0e
        # for first layer: 
        for i in range(num_interactions):
            hidden_irreps_out = str(self.hidden_irreps[0]) if i == num_interactions - 1 else self.hidden_irreps
            if i > 0:
                self.irreps_in[properties.node_feat] = self.hidden_irreps
            if i == 0:
                inter = interaction_cls_first(
                    irreps_in=self.irreps_in,
                    target_irreps=interaction_irreps,
                    hidden_irreps=hidden_irreps_out,
                    radial_MLP=radial_MLP,
                    avg_num_neighbors=avg_num_neighbors,
                )
            else:
                inter = interaction_cls(
                    irreps_in=self.irreps_in,
                    target_irreps=interaction_irreps,
                    hidden_irreps=hidden_irreps_out,
                    radial_MLP=radial_MLP,
                    avg_num_neighbors=avg_num_neighbors,
                )
            self.interactions.append(inter)
            
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=inter.target_irreps if i == 0 else interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i],
                num_elements=num_elements,
                use_sc="Residual" in str(interaction_cls_first) if i == 0 else True,
            )
            self.products.append(prod)

        readout = self._normalize_readout_factory(
            readout,
            base_cls=MACEAtomwiseNN,
        )

        # Setup readout function
        self.readout = self._instantiate_readout(
            readout,
            heads=self.heads,
            num_interactions=num_interactions,
            hidden_irreps=self.hidden_irreps,
            MLP_irreps=self.MLP_irreps,
        )

    def export_init_kwargs(self) -> Dict[str, Any]:
        correlation = None
        symmetric_contractions = getattr(self.products[0], "symmetric_contractions", None)
        contractions = getattr(symmetric_contractions, "contractions", None)
        if contractions:
            first = contractions[0]
            if hasattr(first, "correlation"):
                correlation = int(first.correlation)
            elif hasattr(first, "weights"):
                total = len(first.weights)
                if total % 3 == 0:
                    correlation = total // 3 - 1
        elif hasattr(symmetric_contractions, "sc"):
            contraction_degree = getattr(symmetric_contractions.sc, "contraction_degree", None)
            if contraction_degree is not None:
                correlation = int(contraction_degree)
        if correlation is None:
            raise AttributeError("Unable to infer MACE correlation from symmetric contractions.")

        distance_transform = getattr(self.embeddings.radial_basis, "distance_transform", None)
        if distance_transform is None:
            distance_transform_name = "none"
        elif distance_transform.__class__.__name__ == "AgnesiTransform":
            distance_transform_name = "agnesi"
        elif distance_transform.__class__.__name__ == "SoftTransform":
            distance_transform_name = "soft"
        else:
            distance_transform_name = distance_transform

        species = list(self.species or [])
        num_elements = getattr(self.embeddings.onehot_embedding, "num_elements", len(species) or None)
        rep_config: Dict[str, Any] = {
            "cutoff": self.cutoff,
            "num_interactions": len(self.interactions),
            "correlation": correlation,
            "interaction_cls": self.interactions[-1].__class__,
            "interaction_cls_first": self.interactions[0].__class__,
            "radial_MLP": list(self.interactions[0].conv_tp_weights.hs[1:-1]),
            "species": species,
            "num_elements": num_elements,
            "hidden_irreps": self.hidden_irreps,
            "edge_sh_irreps": self.edge_sh_irreps,
            "node_irreps": self.node_irreps,
            "MLP_irreps": self.MLP_irreps,
            "avg_num_neighbors": float(self.interactions[0].avg_num_neighbors),
            "num_basis": self.embeddings.radial_basis.basis.num_basis,
            "power": self.embeddings.radial_basis.cutoff_fn.p,
            "distance_transform": distance_transform_name,
            "filter_forbidden_irreps": getattr(self, "filter_forbidden_irreps", True),
        }
        readout = getattr(self, "readout", None)
        domain_modules = getattr(readout, "domain_modules", None)
        if domain_modules:
            from curator.layer import MultiDomainMACEAtomwiseNN

            domains = getattr(readout, "domains", None) or list(domain_modules.keys())
            heads_by_domain = {
                str(domain): list(module.heads)
                for domain, module in domain_modules.items()
                if hasattr(module, "heads")
            }
            readout_kwargs: Dict[str, Any] = {"domains": [str(domain) for domain in domains]}
            if heads_by_domain:
                readout_kwargs["heads_by_domain"] = heads_by_domain
            template_module = next(iter(domain_modules.values()))
            if getattr(template_module, "separate_heads", False):
                readout_kwargs["separate_heads"] = True
            rep_config["readout"] = partial(MultiDomainMACEAtomwiseNN, **readout_kwargs)
        elif readout is not None and hasattr(readout, "heads"):
            readout_kwargs: Dict[str, Any] = {"heads": list(readout.heads)}
            if getattr(readout, "separate_heads", False):
                readout_kwargs["separate_heads"] = True
            rep_config["readout"] = partial(readout.__class__, **readout_kwargs)
        return rep_config
            
    def forward(
        self,
        data: properties.Type,
        lammps_data: Optional[Any] = None,
        n_local: Optional[int] = None,
        n_ghost: Optional[int] = None,
    ) -> properties.Type:
        # add mask for local interaction part
        edge_cache = self._apply_cutoff_mask(data, self.cutoff)
        for m in self.embeddings.values():
            data = m(data)
        
        data[properties.node_embedding] = data[properties.node_feat]        # store node embedding for some modules (charge equilibration)
        
        node_feat = data[properties.node_feat]
        node_feat_list = []
        
        for interaction, product in zip(
            self.interactions, self.products
        ):
            node_feat, sc = interaction(
                node_feat, 
                data[properties.node_attr],
                data[properties.edge_idx], 
                data[properties.edge_dist_embedding],
                data[properties.edge_diff_embedding],
                lammps_data=lammps_data,
                n_local=n_local,
                n_ghost=n_ghost,
            )
            node_feat = product(
                node_feats=node_feat,
                sc=sc,
                node_attrs=data[properties.node_attr],
            )
            node_feat_list.append(node_feat)
        
        node_feat_list = torch.cat(node_feat_list, dim=-1)
        data[properties.node_feat] = node_feat_list

        # get properties
        data = self.readout(data)

        # restore neighbor list
        self._restore_cutoff_mask(data, edge_cache)
        return data

    def module_groups(self):
        groups = OrderedDict(
            (
                ("embeddings", [self.embeddings]),
                ("interactions", [self.interactions]),
                ("products", [self.products]),
                ("readout", [self.readout]),
            )
        )
        return groups

    def parameter_groups(self) -> List[ParameterGroup]:
        groups: List[ParameterGroup] = []
        seen: set[int] = set()

        embeddings = collect_unique_parameters([self.embeddings], seen=seen)
        if embeddings:
            groups.append(ParameterGroup(name="embeddings", params=embeddings))

        interactions_decay = []
        interactions_no_decay = []
        for name, param in self.interactions.named_parameters():
            if not isinstance(param, nn.Parameter):
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            if "linear.weight" in name or "skip_tp_full.weight" in name:
                interactions_decay.append(param)
            else:
                interactions_no_decay.append(param)
        if interactions_decay:
            groups.append(ParameterGroup(name="interactions_decay", params=interactions_decay))
        if interactions_no_decay:
            groups.append(ParameterGroup(name="interactions_no_decay", params=interactions_no_decay))

        products = collect_unique_parameters([self.products], seen=seen)
        if products:
            groups.append(ParameterGroup(name="products", params=products))

        readout = collect_unique_parameters([self.readout], seen=seen)
        if readout:
            groups.append(ParameterGroup(name="readout", params=readout))

        return groups
