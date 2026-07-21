from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def thermo_callback(
    *,
    logger_target: str,
    uncertainty_method: str,
    model_like: Any,
    reference_dataset: Optional[str],
    device: str,
    run_dir: Path,
    log_interval: int,
    low_threshold: Any,
    high_threshold: Any,
    uncertain_count: Any,
    uncertainty_kernel: str,
    uncertainty_max_structures: Any,
) -> Dict[str, Any]:
    if uncertainty_method == "none":
        return {
            "_target_": logger_target,
            "variables": "full_thermodynamics",
            "interval": int(log_interval),
            "header": True,
        }

    maha_kwargs: Dict[str, Any] = {
        "kernel": uncertainty_kernel,
        "max_structures": None if uncertainty_max_structures is None else int(uncertainty_max_structures),
    }
    ensemble_kwargs: Dict[str, Any] = {"uncertainty_keys": ["force_sd", "energy_sd"]}
    if low_threshold is not None:
        maha_kwargs["low_threshold"] = float(low_threshold)
        ensemble_kwargs["low_threshold"] = float(low_threshold)
    if high_threshold is not None:
        maha_kwargs["high_threshold"] = float(high_threshold)
        ensemble_kwargs["high_threshold"] = float(high_threshold)

    payload = {
        "_target_": logger_target,
        "variables": "basic_energies",
        "interval": int(log_interval),
        "header": True,
        "monitor": "maha_dist" if uncertainty_method == "mahalanobis" else None,
        "low": low_threshold,
        "high": high_threshold,
        "uncertain_count": uncertain_count,
        "uncertainty_backend": {
            "_target_": "curator.simulate.uncertainty.auto.AutoUncertainty",
            "calculator": model_like,
            "dataset": reference_dataset,
            "maha_kwargs": maha_kwargs,
            "ensemble_kwargs": ensemble_kwargs,
            "device": device,
        },
    }
    if logger_target != "curator.simulate.callbacks.torchsim_logger.TorchSimThermoLogger":
        payload["save_path"] = str(run_dir / "warning_struct.traj")
    return payload
