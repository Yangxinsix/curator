"""Euclidean fast attention layers and state-conditioned energy variants."""

from .core import EuclideanFastAttention
from .adapter import EFAAdapter
from .latent import LatentChargeEFAAdapter
from .q_conditioned import QConditionedEFAAdapter
from .explicit_q import QConditionedEFAEnergy
from .convex import ConvexQConditionedEFAEnergy
from .two_scale import TwoScaleQConditionedEFAEnergy
from .proposal import EFAChargeProposal
from .shared import SharedGeometryChargeProposal, SharedTwoScaleQConditionedEFAEnergy
from .blocks import EquivariantEFA
from .readout import InvariantEnergyReadout
from .state import (
    ChargeStateEncoder,
    ScalarChargeEncoder,
    GatedChargeEncoder,
    EquivariantStateFusion,
    StateEFAEnergy,
)

__all__ = [
    "EuclideanFastAttention",
    "EFAAdapter",
    "LatentChargeEFAAdapter",
    "QConditionedEFAAdapter",
    "QConditionedEFAEnergy",
    "ConvexQConditionedEFAEnergy",
    "TwoScaleQConditionedEFAEnergy",
    "EFAChargeProposal",
    "SharedGeometryChargeProposal",
    "SharedTwoScaleQConditionedEFAEnergy",
    "EquivariantEFA",
    "InvariantEnergyReadout",
    "ChargeStateEncoder",
    "ScalarChargeEncoder",
    "GatedChargeEncoder",
    "EquivariantStateFusion",
    "StateEFAEnergy",
]
