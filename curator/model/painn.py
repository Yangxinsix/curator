from __future__ import annotations

import copy
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import torch
from torch import nn

from curator.data import properties
from curator.layer import (
    AtomwiseNN,
    CosineCutoff,
    PainnMessage,
    PainnUpdate,
    RadialBasisEdgeEncoding,
    ResidualAdd,
    SineBasis,
    VarianceScale,
)
from curator.model.base import ParameterGroup, Representation
from curator.model.features import ScalarVectorFeatureSpec


ModuleFactory = Callable[..., nn.Module]
LayerScaleSpec = Union[nn.Module, float, None]


class Painn(Representation):
    """PaiNN representation with composable numerical-stability components."""

    def __init__(
        self,
        num_interactions: int,
        num_features: int,
        cutoff: float,
        num_basis: int = 20,
        cutoff_fn: Optional[nn.Module] = None,
        radial_basis: Optional[nn.Module] = None,
        readout: Union[AtomwiseNN, Type[AtomwiseNN], partial] = AtomwiseNN,
        heads: Optional[list] = None,
        num_elements: int = 119,
        atomic_number_offset: int = 0,
        atom_embedding: Optional[nn.Module] = None,
        activation: ModuleFactory = nn.SiLU,
        scalar_norm: Optional[ModuleFactory] = None,
        message_residual_scale: float = 1.0,
        state_vector_scale: float = 1.0,
        message_vector_scale: float = 1.0,
        inner_product_scale: float = 1.0,
        scalar_update_scale: float = 1.0,
        norm_eps: float = 1e-8,
        vector_bias: bool = False,
        update_scalar_first: bool = False,
        layer_scales: Optional[Sequence[LayerScaleSpec]] = None,
        linear_initializer: Optional[Callable[[nn.Module], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(heads=heads)
        if (
            num_interactions <= 0
            or num_features <= 0
            or num_basis <= 0
            or num_elements <= 0
        ):
            raise ValueError(
                "num_interactions, num_features, num_basis, and num_elements "
                "must be positive."
            )

        self.cutoff = float(cutoff)
        self.num_interactions = int(num_interactions)
        self.num_features = int(num_features)
        self.num_basis = int(num_basis)
        self.num_elements = int(num_elements)
        self.atomic_number_offset = int(atomic_number_offset)
        self.activation_factory = activation
        self.scalar_norm_factory = scalar_norm
        self.message_residual_scale = float(message_residual_scale)
        self.state_vector_scale = float(state_vector_scale)
        self.message_vector_scale = float(message_vector_scale)
        self.inner_product_scale = float(inner_product_scale)
        self.scalar_update_scale = float(scalar_update_scale)
        self.norm_eps = float(norm_eps)
        self.vector_bias = bool(vector_bias)
        self.update_scalar_first = bool(update_scalar_first)
        self.linear_initializer = linear_initializer

        self.atom_embedding = (
            nn.Embedding(self.num_elements, self.num_features)
            if atom_embedding is None
            else atom_embedding
        )
        self.radial_encoding = RadialBasisEdgeEncoding(
            basis=(
                SineBasis(cutoff=self.cutoff, num_basis=self.num_basis)
                if radial_basis is None
                else radial_basis
            ),
            cutoff_fn=(
                CosineCutoff(cutoff=self.cutoff)
                if cutoff_fn is None
                else cutoff_fn
            ),
        )

        def new_activation() -> nn.Module:
            return activation()

        def new_scalar_norm() -> nn.Module:
            return nn.Identity() if scalar_norm is None else scalar_norm(self.num_features)

        self.message_layers = nn.ModuleList(
            [
                PainnMessage(
                    num_features=self.num_features,
                    num_basis=self.num_basis,
                    activation=new_activation(),
                    scalar_norm=new_scalar_norm(),
                    state_vector_scale=self.state_vector_scale,
                    message_vector_scale=self.message_vector_scale,
                    resnet=False,
                    linear_initializer=linear_initializer,
                )
                for _ in range(self.num_interactions)
            ]
        )
        self.update_layers = nn.ModuleList(
            [
                PainnUpdate(
                    num_features=self.num_features,
                    activation=new_activation(),
                    inner_product_scale=self.inner_product_scale,
                    scalar_update_scale=self.scalar_update_scale,
                    eps=self.norm_eps,
                    vector_bias=self.vector_bias,
                    scalar_first=self.update_scalar_first,
                    resnet=False,
                    linear_initializer=linear_initializer,
                )
                for _ in range(self.num_interactions)
            ]
        )
        self.message_residuals = nn.ModuleList(
            [
                ResidualAdd(self.message_residual_scale)
                for _ in range(self.num_interactions)
            ]
        )
        self.layer_scales = nn.ModuleList(
            self._make_layer_scales(layer_scales)
        )

        readout = self._normalize_readout_factory(readout, base_cls=AtomwiseNN)
        self.readout = self._instantiate_readout(
            readout,
            heads=self.heads,
            in_features=self.num_features,
        )

    def _make_layer_scales(
        self,
        layer_scales: Optional[Sequence[LayerScaleSpec]],
    ) -> List[nn.Module]:
        if layer_scales is None:
            return [nn.Identity() for _ in range(self.num_interactions)]
        if len(layer_scales) != self.num_interactions:
            raise ValueError(
                "layer_scales must contain one entry per PaiNN interaction."
            )
        return [
            copy.deepcopy(spec)
            if isinstance(spec, nn.Module)
            else VarianceScale(spec)
            for spec in layer_scales
        ]

    @property
    @torch.jit.unused
    def radial_basis(self) -> nn.Module:
        return self.radial_encoding.basis

    @property
    @torch.jit.unused
    def cutoff_fn(self) -> nn.Module:
        return self.radial_encoding.cutoff_fn

    def export_init_kwargs(self) -> Dict[str, Any]:
        rep_config = {
            "num_interactions": self.num_interactions,
            "num_features": self.num_features,
            "cutoff": self.cutoff,
            "num_basis": self.num_basis,
            "num_elements": self.num_elements,
            "atomic_number_offset": self.atomic_number_offset,
            "atom_embedding": copy.deepcopy(self.atom_embedding),
            "cutoff_fn": copy.deepcopy(self.cutoff_fn),
            "radial_basis": copy.deepcopy(self.radial_basis),
            "activation": self.activation_factory,
            "scalar_norm": self.scalar_norm_factory,
            "message_residual_scale": self.message_residual_scale,
            "state_vector_scale": self.state_vector_scale,
            "message_vector_scale": self.message_vector_scale,
            "inner_product_scale": self.inner_product_scale,
            "scalar_update_scale": self.scalar_update_scale,
            "norm_eps": self.norm_eps,
            "vector_bias": self.vector_bias,
            "update_scalar_first": self.update_scalar_first,
            "layer_scales": copy.deepcopy(list(self.layer_scales)),
            "linear_initializer": self.linear_initializer,
        }
        if hasattr(self.readout, "heads"):
            readout_kwargs: Dict[str, Any] = {"heads": list(self.readout.heads)}
            if getattr(self.readout, "separate_heads", False):
                readout_kwargs["separate_heads"] = True
            rep_config["readout"] = partial(
                self.readout.__class__,
                **readout_kwargs,
            )
        return rep_config

    def forward(
        self,
        data: properties.Type,
        lammps_data: Optional[Any] = None,
        n_local: Optional[int] = None,
        n_ghost: Optional[int] = None,
    ) -> properties.Type:
        edge_cache = self._apply_cutoff_mask(data, self.cutoff)
        self.radial_encoding(data)

        atom_types = data[properties.Z] - self.atomic_number_offset
        if not torch.jit.is_scripting() and (
            bool(torch.any(atom_types < 0))
            or bool(torch.any(atom_types >= self.num_elements))
        ):
            raise ValueError(
                "Atomic numbers are outside the configured embedding range."
            )
        node_scalar = self.atom_embedding(atom_types)
        node_vector = torch.zeros(
            (node_scalar.shape[0], 3, self.num_features),
            device=data[properties.edge_diff].device,
            dtype=data[properties.edge_diff].dtype,
        )
        data[properties.node_embedding] = node_scalar

        for message, update, residual, layer_scale in zip(
            self.message_layers,
            self.update_layers,
            self.message_residuals,
            self.layer_scales,
        ):
            node_feat = torch.cat(
                [node_scalar, node_vector.reshape(-1, 3 * self.num_features)],
                dim=-1,
            )
            message_delta = message(
                node_feat,
                data[properties.edge_idx],
                data[properties.edge_dist],
                data[properties.edge_diff],
                data[properties.edge_dist_embedding],
                lammps_data=lammps_data,
                n_local=n_local,
                n_ghost=n_ghost,
            )
            delta_scalar, delta_vector = torch.split(
                message_delta,
                [self.num_features, 3 * self.num_features],
                dim=-1,
            )
            node_scalar = residual(
                node_scalar[: delta_scalar.shape[0]],
                delta_scalar,
            )
            node_vector = (
                node_vector[: delta_vector.shape[0]]
                + delta_vector.reshape(-1, 3, self.num_features)
            )

            update_delta = update(
                torch.cat(
                    [
                        node_scalar,
                        node_vector.reshape(-1, 3 * self.num_features),
                    ],
                    dim=-1,
                )
            )
            delta_scalar, delta_vector = torch.split(
                update_delta,
                [self.num_features, 3 * self.num_features],
                dim=-1,
            )
            node_scalar = layer_scale(node_scalar + delta_scalar)
            node_vector = (
                node_vector
                + delta_vector.reshape(-1, 3, self.num_features)
            )

        data[properties.node_feat] = node_scalar
        data[properties.node_final_feature] = node_scalar
        data[properties.node_vect] = node_vector
        data = self.readout(data)
        self._restore_cutoff_mask(data, edge_cache)
        return data

    @torch.jit.unused
    def direct_force_feature_spec(self) -> ScalarVectorFeatureSpec:
        return ScalarVectorFeatureSpec(
            scalar_key=properties.node_feat,
            vector_key=properties.node_vect,
            scalar_channels=self.num_features,
            vector_channels=self.num_features,
        )

    def module_groups(self):
        return OrderedDict(
            (
                ("embedding", [self.atom_embedding, self.radial_encoding]),
                ("message_layers", [self.message_layers]),
                ("update_layers", [self.update_layers]),
                ("readout", [self.readout]),
            )
        )

    def parameter_groups(self) -> List[ParameterGroup]:
        return super().parameter_groups()


__all__ = ["Painn"]
