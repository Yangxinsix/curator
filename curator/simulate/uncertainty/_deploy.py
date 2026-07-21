from __future__ import annotations

"""Compatibility wrapper for legacy deploy-time uncertainty injection.

New code should import :func:`curator.simulate.uncertainty.inject.inject_uncertainty`.
This module only keeps older train/evaluate/deploy paths working while the CLI
surface is cleaned up.
"""

from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from curator.data._uncertainty import collect_uncertainty_outputs
from curator.simulate.uncertainty.inject import inject_uncertainty


def _plain_spec(spec: Optional[Any]) -> Optional[dict[str, Any]]:
    if spec is None:
        return None
    if isinstance(spec, DictConfig):
        spec = OmegaConf.to_container(spec, resolve=False)
    return dict(spec)


def _prepare_legacy_ensemble(model, spec: dict[str, Any]) -> None:
    from curator.model import EnsembleModel

    if not isinstance(model, EnsembleModel) or len(model.models) <= 1:
        raise ValueError("deploy.uncertainty.method=ensemble requires an ensemble model.")

    output_keys = spec.get("output_keys")
    if output_keys is None:
        return
    scalar_keys, per_atom_keys = collect_uncertainty_outputs(model)
    available_keys = set(model.model_outputs) | set(scalar_keys) | set(per_atom_keys)
    missing = [key for key in output_keys if key not in available_keys]
    if missing:
        raise ValueError(f"Requested deploy ensemble uncertainty keys are not present in model outputs: {missing}")


def prepare_deploy_uncertainty(
    model,
    spec: Optional[Any],
    *,
    lammps_mliap: bool = False,
    torchscript: bool = True,
) -> None:
    spec = _plain_spec(spec)
    if spec is None:
        return
    method = str(spec.get("method", "none")).strip().lower()
    if method in ("", "none", "null"):
        return
    if method == "ensemble":
        _prepare_legacy_ensemble(model, spec)
        return
    implementation = "native" if lammps_mliap or not torchscript else "scriptable"
    inject_uncertainty(model, spec, implementation=implementation)
