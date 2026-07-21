from .._torch_compat import ensure_torch_safe_globals
from curator.data._uncertainty import UncertaintyModule, collect_uncertainty_outputs

ensure_torch_safe_globals()

from .painn import Painn
from .nequip import Nequip
from .mace import MACE
from .base import NeuralNetworkPotential, LitNNP
from .ensemble import EnsembleModel
from .conversion import (
    build_mace_from_curator,
    convert_multi_to_selected_domains,
    convert_multi_to_single_domain,
    convert_single_to_multi_domain,
    create_model_from_mace,
)
from .multi_domain import (
    align_model_domains,
    align_model_domains_from_datamodule,
)
from .external import (
    AllegroRepresentation,
    ESENRepresentation,
    ExternalRepresentation,
    MatGLRepresentation,
)
from .adapter import (
    AllegroAdapter,
    ESENAdapter,
    ExternalModelSpec,
    MatGLAdapter,
    is_external_model_spec,
    load_external_model,
    parse_external_model_spec,
    register_adapter_loader,
)

from . import compat
