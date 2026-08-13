from .._torch_compat import ensure_torch_safe_globals

ensure_torch_safe_globals()

from ._atomic_linear import (
    AtomwiseLinear,
    AtomwiseNonLinear,
    FeatureProjection,
    ProjectedElementEmbedding,
)
from ._atomwise_reduce import AtomwiseReduce
from ._atomwise_nn import AtomwiseNN, MACEAtomwiseNN, MultiDomainAtomwiseNN, MultiDomainMACEAtomwiseNN
from ._convnet import ConvNetLayer
from ._charge_equilibration import ChargeEquilibration
from ._energy_hessian import EnergyHessianOutput
from ._ewald import EwaldSummation
from ._grad_output import EnergyGradientOutput, GradientOutput
from ._force_output import (
    DirectForceOutput,
    ForceOutput,
    IrrepsForceHead,
    ScalarVectorForceHead,
)
from .gated_equivariant import GatedEquivariantBlock
from .initialization import reset_linear
from ._nequip_interaction import InteractionLayer
from ._mace_interaction import (
    EquivariantProductBasisBlock,
    RealAgnosticDensityInteractionBlock,
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from ._mace_readout_adapter import MACEReadoutAdapter
from ._node_embedding import OneHotAtomEncoding
from ._painn_message import PainnMessage
from ._painn_update import PainnUpdate
from ._pairwise_distance import PairwiseDistance, get_pair_distance
from ._pair_repulsion import ZBLBasis, PairRepulsionEnergy
from ._symmetric_contraction import Contraction, SymmetricContraction
from ._rescale import GlobalRescaleShift, PerSpeciesRescaleShift, PerSpeciesScale, MultiDomainRescaleShift
from ._strain import Strain
from ._feature import (
    BaseKMEAggregator,
    FeatureAggregator,
    FeatureCalculator,
    FeatureExtractor,
    FeatureKernel,
    FeatureSpec,
    FeatureStatistics,
    H5Feature,
    IdentityKMEAggregator,
    MeanAggregator,
    RandomFourierKMEAggregator,
    SketchingKMEAggregator,
    SumAggregator,
    DistanceMetrics,
    feature_spec_from_object,
    normalize_kernel,
)
from ._atomwise_nn import Dense, AtomwiseNN, MACEAtomwiseNN, MultiDomainAtomwiseNN, MultiDomainMACEAtomwiseNN
from .cutoff import CosineCutoff, PolynomialCutoff
from .activation import ScaledSiLU
from .nonlinearities import ShiftedSoftPlus
from .norm import safe_norm
from .residual import ResidualAdd
from .variance_scale import VarianceScale
from .wrappers import merge_model_wrappers
from .radial_basis import (
    BesselBasis,
    SineBasis,
    GaussianBasis,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
    AgnesiTransform,
    SoftTransform,
)
from .utils import (
    tp_path_exists, 
    tp_out_irreps_with_instructions,
    linear_out_irreps,
    reshape_irreps,
    find_layer_by_name_recursive,
)

__all__ = [
    AtomwiseLinear,
    AtomwiseNonLinear,
    FeatureProjection,
    ProjectedElementEmbedding,
    AtomwiseReduce,
    AtomwiseNN,
    MACEAtomwiseNN,
    MultiDomainAtomwiseNN,
    MultiDomainMACEAtomwiseNN,
    ConvNetLayer,
    ChargeEquilibration,
    EnergyHessianOutput,
    EwaldSummation,
    InteractionLayer,
    EquivariantProductBasisBlock,
    RealAgnosticDensityInteractionBlock,
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    MACEReadoutAdapter,
    Contraction,
    SymmetricContraction,
    OneHotAtomEncoding,
    PainnMessage,
    PainnUpdate,
    CosineCutoff,
    PolynomialCutoff,
    ScaledSiLU,
    ShiftedSoftPlus,
    safe_norm,
    ResidualAdd,
    VarianceScale,
    BesselBasis,
    SineBasis,
    GaussianBasis,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
    AgnesiTransform,
    SoftTransform,
    tp_path_exists,
    tp_out_irreps_with_instructions,
    linear_out_irreps,
    get_pair_distance,
    reshape_irreps,
    PairwiseDistance,         # input modules (preprocess, calculate pairwise distances)
    Strain,                   # input modules (preprocess, add strain on cell and atom positions)
    ZBLBasis,
    PairRepulsionEnergy,
    GradientOutput,           # output modules (output forces and stress)
    EnergyGradientOutput,
    ForceOutput,
    DirectForceOutput,
    GatedEquivariantBlock,
    reset_linear,
    ScalarVectorForceHead,
    IrrepsForceHead,
    EnergyHessianOutput,
    GlobalRescaleShift,            # output modules (postprocess energy)
    PerSpeciesRescaleShift,
    PerSpeciesScale,
    MultiDomainRescaleShift,
    BaseKMEAggregator,
    FeatureExtractor,
    FeatureCalculator, 
    FeatureAggregator,
    FeatureKernel,
    FeatureSpec,
    FeatureStatistics,
    H5Feature,
    IdentityKMEAggregator,
    MeanAggregator,
    RandomFourierKMEAggregator,
    SketchingKMEAggregator,
    SumAggregator,
    DistanceMetrics,
    feature_spec_from_object,
    normalize_kernel,
    MACEAtomwiseNN,
    AtomwiseNN,
    MultiDomainAtomwiseNN,
    MultiDomainMACEAtomwiseNN,
    Dense,
    merge_model_wrappers,
]
