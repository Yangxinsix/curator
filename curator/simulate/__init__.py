"""Simulation package exports.

Keep import-time work small so CLI startup does not eagerly import optional
backends such as LAMMPS, OpenMM, or torch-sim.
"""

from importlib import import_module

_EXPORTS = {
    "MLCalculator": (".core.calculator", "MLCalculator"),
    "EnsembleCalculator": (".core.calculator", "EnsembleCalculator"),
    "MDLogger": (".logger", "MDLogger"),
    "BaseUncertainty": (".uncertainty", "BaseUncertainty"),
    "EnsembleUncertainty": (".uncertainty", "EnsembleUncertainty"),
    "MahalanobisUncertainty": (".uncertainty", "MahalanobisUncertainty"),
    "MCDropoutUncertainty": (".uncertainty", "MCDropoutUncertainty"),
    "TorchSimEngine": (".engines.torchsim", "TorchSimEngine"),
    "TorchSimThermoLogger": (".callbacks.torchsim_logger", "TorchSimThermoLogger"),
    "LAMMPS_MLIAP": (".lammps_mliap_interface", "LAMMPS_MLIAP"),
    "CURATORLammpsConfig": (".lammps_mliap_interface", "CURATORLammpsConfig"),
    "timer": (".lammps_mliap_interface", "timer"),
    "CuratorOpenMM": (".openmm", "CuratorOpenMM"),
    "export_curator_to_openmm_torchscript": (".openmm", "export_curator_to_openmm_torchscript"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    try:
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
    except ImportError:
        value = None
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
