from .callbacks import DistillLossWeightSchedule, ExponentialMovingAverage, FreezeSchedule
from .losses import (
    HuberLoss,
    SpeciesBalancedLoss,
    StructureBalancedLoss,
    VectorHuberLoss,
    VectorL2MAELoss,
)
from .variational_losses import VariationalJointLoss, VariationalModelOutput
from .amortized_variational import (
    AmortizedTrainingModelOutput,
    AmortizedVariationalLoss,
)
from .model_output import ModelOutput
from .metrics import (
    AtomsMetric,
    PerSpeciesMAE, 
    PerSpeciesRMSE, 
    PerAtomMAE, 
    PerAtomRMSE
)
from .train import train
from .variance_scaling import (
    ensure_variance_scales_fitted,
    find_variance_scales,
    fit_variance_scales,
)
__all__ = [
    ExponentialMovingAverage,
    DistillLossWeightSchedule,
    FreezeSchedule,
    HuberLoss,
    SpeciesBalancedLoss,
    StructureBalancedLoss,
    VectorHuberLoss,
    VectorL2MAELoss,
    VariationalJointLoss,
    VariationalModelOutput,
    AmortizedTrainingModelOutput,
    AmortizedVariationalLoss,
    ModelOutput,
    AtomsMetric,
    PerSpeciesMAE, 
    PerSpeciesRMSE, 
    PerAtomMAE, 
    PerAtomRMSE,
    ensure_variance_scales_fitted,
    find_variance_scales,
    fit_variance_scales,
    train,
]


def __getattr__(name):
    # LitNNP imports curator.train during initialization; defer the task import
    # so the convenient Hydra target does not create an eager circular import.
    if name == "VariationalGradientTask":
        from .variational_optimization import VariationalGradientTask
        return VariationalGradientTask
    raise AttributeError(name)
