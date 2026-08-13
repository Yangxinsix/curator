"""Typed descriptions of representation features used by output heads."""

from dataclasses import dataclass
from typing import Union

from e3nn import o3


@dataclass(frozen=True)
class ScalarVectorFeatureSpec:
    """Describe separate invariant-scalar and equivariant-vector features.

    Vector features are expected to have shape ``[n_atoms, 3, vector_channels]``.
    """

    scalar_key: str
    vector_key: str
    scalar_channels: int
    vector_channels: int

    def __post_init__(self) -> None:
        if not self.scalar_key:
            raise ValueError("scalar_key must be non-empty.")
        if not self.vector_key:
            raise ValueError("vector_key must be non-empty.")
        if self.scalar_channels <= 0:
            raise ValueError("scalar_channels must be positive.")
        if self.vector_channels <= 0:
            raise ValueError("vector_channels must be positive.")


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


__all__ = ["ScalarVectorFeatureSpec", "IrrepsFeatureSpec"]
