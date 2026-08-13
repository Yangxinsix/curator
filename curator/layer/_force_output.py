"""Architecture-independent direct-force output modules."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

import torch
from e3nn import o3
from torch import nn

from curator.data import properties

from ._ops import Linear
from .gated_equivariant import GatedEquivariantBlock

if TYPE_CHECKING:
    from curator.model.features import IrrepsFeatureSpec, ScalarVectorFeatureSpec


class ForceOutput(nn.Module):
    """Marker base class for modules that produce atomic forces."""

    produces_forces: bool = True


class ScalarVectorForceHead(nn.Module):
    """Predict forces from separate invariant scalar and equivariant vector features."""

    def __init__(
        self,
        spec: ScalarVectorFeatureSpec,
        hidden_channels: Optional[int] = None,
        block_kwargs: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        hidden_channels = (
            max(1, min(spec.scalar_channels, spec.vector_channels) // 2)
            if hidden_channels is None
            else int(hidden_channels)
        )
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")

        self.scalar_key = spec.scalar_key
        self.vector_key = spec.vector_key
        self.scalar_channels = spec.scalar_channels
        self.vector_channels = spec.vector_channels
        self.hidden_channels = hidden_channels
        if block_kwargs is None:
            block_kwargs = ({}, {})
        if len(block_kwargs) != 2:
            raise ValueError("block_kwargs must contain exactly two mappings.")
        reserved = {
            "scalar_in",
            "vector_in",
            "scalar_out",
            "vector_out",
        }
        for kwargs in block_kwargs:
            overlap = reserved.intersection(kwargs)
            if overlap:
                raise ValueError(
                    "Block dimensions are controlled by ScalarVectorForceHead; "
                    f"remove {sorted(overlap)} from block_kwargs."
                )

        self.blocks = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    scalar_in=spec.scalar_channels,
                    vector_in=spec.vector_channels,
                    scalar_out=hidden_channels,
                    vector_out=hidden_channels,
                    **copy.deepcopy(dict(block_kwargs[0])),
                ),
                GatedEquivariantBlock(
                    scalar_in=hidden_channels,
                    vector_in=hidden_channels,
                    scalar_out=1,
                    vector_out=1,
                    **copy.deepcopy(dict(block_kwargs[1])),
                ),
            ]
        )

    def forward(self, data: properties.Type) -> torch.Tensor:
        scalars = data[self.scalar_key]
        vectors = data[self.vector_key]
        if scalars.ndim != 2 or scalars.shape[-1] != self.scalar_channels:
            raise ValueError(
                f"Expected scalar features with shape [n_atoms, {self.scalar_channels}]."
            )
        if vectors.ndim != 3 or vectors.shape[1] != 3 or vectors.shape[-1] != self.vector_channels:
            raise ValueError(
                f"Expected vector features with shape [n_atoms, 3, {self.vector_channels}]."
            )

        for block in self.blocks:
            scalars, vectors = block(scalars, vectors)
        return vectors.squeeze(-1)


class IrrepsForceHead(nn.Module):
    """Predict a polar-vector force from a flattened e3nn feature tensor."""

    def __init__(self, spec: IrrepsFeatureSpec) -> None:
        super().__init__()
        self.feature_key = spec.key
        self.irreps_in = o3.Irreps(spec.irreps)
        self.in_dim = self.irreps_in.dim
        self.irreps_out = o3.Irreps("1x1o")
        if o3.Irrep("1o") not in {irrep for _, irrep in self.irreps_in}:
            raise ValueError(
                "Direct force prediction requires at least one polar-vector (1o) "
                f"input irrep, but received {self.irreps_in}."
            )
        self.linear = Linear(irreps_in=self.irreps_in, irreps_out=self.irreps_out)

    def forward(self, data: properties.Type) -> torch.Tensor:
        features = data[self.feature_key]
        if features.ndim != 2 or features.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected irreps features with shape [n_atoms, {self.in_dim}]."
            )
        return self.linear(features)


class _UnboundDirectForceHead(nn.Module):
    def forward(self, data: properties.Type) -> torch.Tensor:
        raise RuntimeError(
            "DirectForceOutput has not been bound to a representation feature spec."
        )


class DirectForceOutput(ForceOutput):
    """Write direct atomic-force predictions from representation features."""

    def __init__(
        self,
        hidden_channels: Optional[int] = None,
        block_kwargs: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        if hidden_channels is not None and hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive when provided.")
        self.hidden_channels = hidden_channels
        self.block_kwargs = (
            None
            if block_kwargs is None
            else [dict(kwargs) for kwargs in block_kwargs]
        )
        self.model_outputs = [properties.forces]
        self.head = _UnboundDirectForceHead()
        self._is_bound = False

    @property
    @torch.jit.unused
    def is_bound(self) -> bool:
        return self._is_bound

    @torch.jit.unused
    def bind(self, spec) -> None:
        """Create the appropriate readout head for a representation's feature spec."""
        if self._is_bound:
            raise RuntimeError("DirectForceOutput is already bound.")

        # Keep this import local: importing curator.model while curator.layer is
        # initialized would otherwise create a package initialization cycle.
        from curator.model.features import IrrepsFeatureSpec, ScalarVectorFeatureSpec

        if isinstance(spec, ScalarVectorFeatureSpec):
            self.head = ScalarVectorForceHead(
                spec,
                hidden_channels=self.hidden_channels,
                block_kwargs=self.block_kwargs,
            )
        elif isinstance(spec, IrrepsFeatureSpec):
            self.head = IrrepsForceHead(spec)
        else:
            raise TypeError(
                "Unsupported direct-force feature spec "
                f"{type(spec).__name__}; expected ScalarVectorFeatureSpec or IrrepsFeatureSpec."
            )
        self._is_bound = True

    @torch.jit.unused
    def reset_binding(self) -> None:
        """Discard a head tied to an old representation before cloning."""
        self.head = _UnboundDirectForceHead()
        self._is_bound = False

    def forward(self, data: properties.Type) -> properties.Type:
        data[properties.forces] = self.head(data)
        return data


__all__ = [
    "ForceOutput",
    "DirectForceOutput",
    "ScalarVectorForceHead",
    "IrrepsForceHead",
]
