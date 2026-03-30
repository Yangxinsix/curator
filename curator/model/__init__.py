from curator.data._uncertainty import UncertaintyModule, collect_uncertainty_outputs

from .painn import Painn
from .nequip import Nequip
from .mace import MACE
from .allegro import AllegroRepresentation
from .esen import ESENRepresentation
from .base import NeuralNetworkPotential, LitNNP
from .ensemble import EnsembleModel
from .conversion import (
    build_mace_from_curator,
    convert_multi_to_single_domain,
    convert_single_to_multi_domain,
    create_model_from_mace,
)
from .adapters import (
    AllegroAdapter,
    ESENAdapter,
    MatGLAdapter,
    is_external_model_spec,
    load_external_model,
)

from . import compat
