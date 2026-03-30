import torch
from collections import OrderedDict
from torch import nn
from e3nn import o3
from e3nn.util.jit import compile_mode

from curator.data import properties
from curator.layer import (
    OneHotAtomEncoding,
    AtomwiseLinear,
    AtomwiseNN,
    RadialBasisEdgeEncoding,
    BesselBasis,
    PolynomialCutoff,
    SphericalHarmonicEdgeAttrs,
    InteractionLayer,
)
from curator.model.base import ParameterGroup, Representation

from typing import Dict, List, Optional, Sequence, Union, Callable, Type
from functools import partial


def _as_irreps(value: Union[o3.Irreps, str, None]) -> Optional[o3.Irreps]:
    if value is None:
        return None
    if isinstance(value, str):
        return o3.Irreps(value)
    return value


def _normalize_num_features(
    num_features: Union[int, Sequence[int], None],
    lmax: int,
) -> List[int]:
    if num_features is None:
        num_features = 32
    if isinstance(num_features, int):
        return [num_features] * (lmax + 1)
    num_features = list(num_features)
    if len(num_features) != lmax + 1:
        raise ValueError(
            f"`num_features` should have length `lmax + 1` ({lmax + 1}), got {num_features}."
        )
    return [int(n) for n in num_features]


def _build_hidden_irreps(
    num_features: Sequence[int],
    lmax: int,
    parity: bool,
) -> o3.Irreps:
    return o3.Irreps(
        [
            (num_features[l], (l, p))
            for l in range(lmax + 1)
            for p in (
                (1, -1) if parity else ((1,) if l % 2 == 0 else (-1,))
            )
        ]
    )


@compile_mode("script")
class Nequip(Representation):
    """NequIP-style representation aligned with NequIP 0.17 defaults."""

    def __init__(
        self,
        cutoff: float,
        num_interactions: int,
        species: Optional[List[str]] = None,
        num_elements: Optional[int] = None,
        hidden_irreps: Union[o3.Irreps, str, None] = None,
        edge_sh_irreps: Union[o3.Irreps, str, None] = None,
        node_irreps: Union[o3.Irreps, str, None] = None,
        lmax: int = 1,
        parity: bool = True,
        num_features: Union[int, Sequence[int], None] = 32,
        type_embed_num_features: Optional[int] = None,
        num_basis: int = 8,
        power: int = 6,
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Optional[Dict[int, Callable]] = None,
        nonlinearity_gates: Optional[Dict[int, Callable]] = None,
        radial_mlp_depth: int = 1,
        radial_mlp_width: int = 128,
        readout_mlp_hidden_layers_depth: int = 0,
        readout_mlp_hidden_layers_width: Optional[int] = None,
        readout_mlp_nonlinearity: Optional[Union[str, Callable]] = "silu",
        convolution_kwargs: Optional[dict] = None,
        readout: Union[AtomwiseNN, Type[AtomwiseNN], partial] = AtomwiseNN,
        use_cueq: bool = False,
        heads: Optional[list] = None,
        **kwargs,
    ) -> None:
        super().__init__(heads=heads)
        if num_interactions <= 0:
            raise ValueError("`num_interactions` must be positive.")

        self.cutoff = cutoff
        self.num_interactions = num_interactions
        self.lmax = lmax
        self.parity = parity
        self.species = species
        self.use_cueq = use_cueq
        self.radial_mlp_depth = int(radial_mlp_depth)
        self.radial_mlp_width = int(radial_mlp_width)
        self.readout_mlp_hidden_layers_depth = readout_mlp_hidden_layers_depth
        self.readout_mlp_hidden_layers_width = readout_mlp_hidden_layers_width
        self.readout_mlp_nonlinearity = readout_mlp_nonlinearity

        self._enable_cueq(use_cueq)

        if nonlinearity_scalars is None:
            nonlinearity_scalars = {"e": "silu", "o": "tanh"}
        if nonlinearity_gates is None:
            nonlinearity_gates = {"e": "silu", "o": "tanh"}
        convolution_kwargs = {} if convolution_kwargs is None else dict(convolution_kwargs)

        if num_elements is None:
            num_elements = len(species) if species is not None else 119

        feature_multiplicities = _normalize_num_features(num_features, lmax)
        self.num_features = list(feature_multiplicities)
        if type_embed_num_features is None:
            type_embed_num_features = feature_multiplicities[0]
        self.type_embed_num_features = int(type_embed_num_features)

        base_hidden_irreps = _as_irreps(hidden_irreps)
        if base_hidden_irreps is None:
            base_hidden_irreps = _build_hidden_irreps(
                num_features=feature_multiplicities,
                lmax=lmax,
                parity=parity,
            )
        self.hidden_irreps = base_hidden_irreps

        self.node_irreps = _as_irreps(node_irreps)
        if self.node_irreps is None:
            self.node_irreps = o3.Irreps([(type_embed_num_features, (0, 1))])

        self.edge_sh_irreps = _as_irreps(edge_sh_irreps)
        if self.edge_sh_irreps is None:
            self.edge_sh_irreps = o3.Irreps.spherical_harmonics(lmax=lmax)

        scalar_width = feature_multiplicities[0]
        if hidden_irreps is not None:
            scalar_width = max(1, self.hidden_irreps.count(o3.Irrep(0, 1)))
        self.final_layer_irreps = o3.Irreps([(scalar_width, (0, 1))])
        self.feature_irreps_hidden_list = [self.hidden_irreps] * max(0, num_interactions - 1)
        self.feature_irreps_hidden_list.append(self.final_layer_irreps)

        self.embeddings = nn.ModuleDict()
        self.embeddings["onehot_embedding"] = OneHotAtomEncoding(
            num_elements=num_elements,
            species=species,
        )
        self.embeddings["radial_basis"] = RadialBasisEdgeEncoding(
            basis=BesselBasis(cutoff=cutoff, num_basis=num_basis),
            cutoff_fn=PolynomialCutoff(cutoff=cutoff, power=power),
        )
        self.embeddings["sphere_harmonics"] = SphericalHarmonicEdgeAttrs(
            edge_sh_irreps=self.edge_sh_irreps
        )

        self.irreps_in = {
            properties.edge_diff_embedding: self.embeddings["sphere_harmonics"].irreps_out,
            properties.edge_dist_embedding: self.embeddings["radial_basis"].irreps_out,
        }
        self.irreps_in.update(self.embeddings["onehot_embedding"].irreps_out)

        self.embeddings["chemical_embedding"] = AtomwiseLinear(
            irreps_in=self.irreps_in[properties.node_attr],
            irreps_out=self.node_irreps,
        )
        self.irreps_in[properties.node_feat] = self.embeddings["chemical_embedding"].irreps_out
        self.irreps_in[properties.node_embedding] = self.embeddings["chemical_embedding"].irreps_out

        self.nonlinearity_type = nonlinearity_type
        self.nonlinearity_scalars = nonlinearity_scalars
        self.nonlinearity_gates = nonlinearity_gates
        self.convolution_kwargs = dict(convolution_kwargs)

        self.interactions = nn.ModuleList()
        for layer_i, feature_irreps_hidden in enumerate(self.feature_irreps_hidden_list):
            layer_conv_kwargs = dict(convolution_kwargs)
            layer_conv_kwargs["radial_mlp_depth"] = radial_mlp_depth
            layer_conv_kwargs["radial_mlp_width"] = radial_mlp_width
            layer_conv_kwargs["use_sc"] = layer_i != 0
            layer_conv_kwargs["is_first_layer"] = layer_i == 0
            layer_conv_kwargs["self_connection_field"] = properties.node_embedding

            interaction = InteractionLayer(
                irreps_in=self.irreps_in,
                feature_irreps_hidden=feature_irreps_hidden,
                convolution_kwargs=layer_conv_kwargs,
                resnet=(layer_i != 0) and resnet,
                nonlinearity_type=nonlinearity_type,
                nonlinearity_scalars=nonlinearity_scalars,
                nonlinearity_gates=nonlinearity_gates,
            )
            self.interactions.append(interaction)
            self.irreps_in.update(interaction.irreps_out)

        if readout_mlp_hidden_layers_width is None:
            readout_mlp_hidden_layers_width = self.irreps_in[properties.node_feat].dim
        self.readout_mlp_hidden_layers_width = readout_mlp_hidden_layers_width

        readout_kwargs = {
            "in_features": self.irreps_in[properties.node_feat],
            "use_e3nn": True,
            "n_hidden_layers": readout_mlp_hidden_layers_depth,
            "activation": readout_mlp_nonlinearity if readout_mlp_nonlinearity is not None else "None",
        }
        if readout_mlp_hidden_layers_depth > 0:
            hidden_irreps = o3.Irreps(f"{readout_mlp_hidden_layers_width}x0e")
            readout_kwargs["n_hidden"] = [
                hidden_irreps for _ in range(readout_mlp_hidden_layers_depth)
            ]
        readout = self._normalize_readout_factory(
            readout,
            base_cls=AtomwiseNN,
        )
        self.readout = self._instantiate_readout(
            readout,
            heads=self.heads,
            **readout_kwargs,
        )

    def export_init_kwargs(self) -> Dict[str, object]:
        mapper = getattr(self.embeddings.onehot_embedding, "type_mapper", None)
        if mapper is not None:
            species = list(mapper.symbol_to_type.keys())
        else:
            species = list(self.species or [])
        num_elements = getattr(self.embeddings.onehot_embedding, "num_elements", len(species))

        return {
            "cutoff": self.cutoff,
            "num_interactions": len(self.interactions),
            "species": species,
            "num_elements": num_elements,
            "hidden_irreps": self.hidden_irreps,
            "edge_sh_irreps": self.edge_sh_irreps,
            "node_irreps": self.node_irreps,
            "lmax": self.lmax,
            "parity": self.parity,
            "num_features": self.num_features,
            "type_embed_num_features": self.type_embed_num_features,
            "num_basis": self.embeddings.radial_basis.basis.num_basis,
            "power": self.embeddings.radial_basis.cutoff_fn.p,
            "resnet": self.interactions[0].resnet,
            "nonlinearity_type": self.nonlinearity_type,
            "nonlinearity_scalars": self.nonlinearity_scalars,
            "nonlinearity_gates": self.nonlinearity_gates,
            "radial_mlp_depth": self.radial_mlp_depth,
            "radial_mlp_width": self.radial_mlp_width,
            "readout_mlp_hidden_layers_depth": self.readout_mlp_hidden_layers_depth,
            "readout_mlp_hidden_layers_width": self.readout_mlp_hidden_layers_width,
            "readout_mlp_nonlinearity": self.readout_mlp_nonlinearity,
            "convolution_kwargs": self.convolution_kwargs,
            "use_cueq": getattr(self, "use_cueq", False),
        }

    def forward(self, data: properties.Type) -> properties.Type:
        edge_cache = self._apply_cutoff_mask(data, self.cutoff)
        for module in self.embeddings.values():
            data = module(data)

        data[properties.node_embedding] = data[properties.node_feat]

        node_feat = data[properties.node_feat]
        for interaction in self.interactions:
            node_feat = interaction(
                node_feat,
                data[properties.node_attr],
                data[properties.node_embedding],
                data[properties.edge_idx],
                data[properties.edge_dist_embedding],
                data[properties.edge_diff_embedding],
            )

        data[properties.node_feat] = node_feat
        data = self.readout(data)

        self._restore_cutoff_mask(data, edge_cache)
        return data

    def module_groups(self):
        return OrderedDict(
            (
                ("embeddings", [self.embeddings]),
                ("interactions", [self.interactions]),
                ("readout", [self.readout]),
            )
        )

    def parameter_groups(self) -> List[ParameterGroup]:
        return super().parameter_groups()
