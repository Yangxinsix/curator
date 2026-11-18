"""Typed descriptions of representation features used by output heads."""

from dataclasses import dataclass
from typing import Mapping, Tuple, Union

import torch
from e3nn import o3


@dataclass(frozen=True)
class ScalarVectorFeatureSpec:
    """Describe separate invariant-scalar and equivariant-vector features.

    Vector features are expected to have shape ``[n_atoms, 3, vector_channels]``.
    A scalar-only representation uses ``vector_channels=0`` and may omit its
    vector tensor (``vector_key=""``).
    """

    scalar_key: str
    vector_key: str
    scalar_channels: int
    vector_channels: int

    def __post_init__(self) -> None:
        if not self.scalar_key:
            raise ValueError("scalar_key must be non-empty.")
        if self.vector_channels and not self.vector_key:
            raise ValueError("vector_key must be non-empty.")
        if self.scalar_channels <= 0:
            raise ValueError("scalar_channels must be positive.")
        if self.vector_channels < 0:
            raise ValueError("vector_channels must be nonnegative.")

    @property
    def irreps(self) -> o3.Irreps:
        """Types in the channel-major flattened response tensor."""
        irreps = o3.Irreps([(self.scalar_channels, "0e")])
        if self.vector_channels:
            irreps += o3.Irreps([(self.vector_channels, "1o")])
        return irreps

    def flatten(self, data: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Pack scalar/vector features without dropping channels or gradients."""
        scalar = data[self.scalar_key]
        if scalar.ndim != 2 or scalar.shape[1] != self.scalar_channels:
            raise ValueError("scalar feature shape does not match its specification")
        if not self.vector_channels:
            return scalar
        vector = data[self.vector_key]
        if vector.shape != (scalar.shape[0], 3, self.vector_channels):
            raise ValueError("vector feature shape does not match its specification")
        return torch.cat((scalar, vector.transpose(1, 2).flatten(1)), dim=-1)


@dataclass(frozen=True)
class IrrepsFeatureSpec:
    """Describe a flattened e3nn feature tensor and its irreducible representations."""

    key: str
    irreps: Union[str, o3.Irreps]

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key must be non-empty.")
        irreps = o3.Irreps(self.irreps)
        if irreps.dim == 0:
            raise ValueError("irreps must contain at least one feature.")
        object.__setattr__(self, "irreps", irreps)

    def flatten(self, data: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return all declared irreps in their original order, including l > 1."""
        features = data[self.key]
        if features.ndim != 2 or features.shape[1] != self.irreps.dim:
            raise ValueError("feature shape does not match its irreps specification")
        return features


def scalar_vector_features(
    features: torch.Tensor, irreps: Union[str, o3.Irreps]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract legacy 0e scalars and 1o vectors from a full response tensor.

    This is only the compatibility view for existing heads and functionals;
    new response heads should also receive the original flattened tensor.
    Missing types return zero-width tensors, preserving atom and vector axes.
    """
    irreps = o3.Irreps(irreps)
    if features.ndim != 2 or features.shape[1] != irreps.dim:
        raise ValueError("feature shape does not match its irreps specification")
    scalars, vectors = [], []
    for (multiplicity, irrep), part in zip(irreps, irreps.slices()):
        if irrep == o3.Irrep("0e"):
            scalars.append(features[:, part])
        elif irrep == o3.Irrep("1o"):
            vectors.append(features[:, part].reshape(features.shape[0], multiplicity, 3).transpose(1, 2))
    scalar = torch.cat(scalars, dim=-1) if scalars else features.new_zeros((features.shape[0], 0))
    vector = torch.cat(vectors, dim=-1) if vectors else features.new_zeros((features.shape[0], 3, 0))
    return scalar, vector


__all__ = ["ScalarVectorFeatureSpec", "IrrepsFeatureSpec", "scalar_vector_features"]
