from .core.calculator import MLCalculator, EnsembleCalculator
from .logger import MDLogger
try:
    from .engines.torchsim import TorchSimEngine  # optional
    from .callbacks.torchsim_logger import TorchSimThermoLogger
    _HAS_TORCHSIM = True
except ImportError:
    TorchSimEngine = None
    TorchSimThermoLogger = None
    _HAS_TORCHSIM = False
from .uncertainty import BaseUncertainty, EnsembleUncertainty, MahalanobisUncertainty, MCDropoutUncertainty
from .lammps_mliap_interface import (
    LAMMPS_MLIAP, 
    LAMMPS_MLIAP_QEQ, 
    prepare_model_for_qeq_inference,
    CURATORLammpsConfig,
)
from .create_lammps_model import create_lammps_model
from .openmm import CuratorOpenMM, export_curator_to_openmm_torchscript

__all__ = [
    "MLCalculator",
    "EnsembleCalculator",
    "BaseUncertainty",
    "EnsembleUncertainty",
    "MahalanobisUncertainty",
    "MCDropoutUncertainty",
    "MDLogger",
    "TorchSimEngine",
    "TorchSimThermoLogger",
    "CuratorOpenMM",
    "export_curator_to_openmm_torchscript",
    # LAMMPS MLIAP
    "LAMMPS_MLIAP",
    "LAMMPS_MLIAP_QEQ",
    "prepare_model_for_qeq_inference",
    "create_lammps_model",
    "CURATORLammpsConfig",
]

# prune missing optional components
__all__ = [name for name in __all__ if globals().get(name) is not None]
