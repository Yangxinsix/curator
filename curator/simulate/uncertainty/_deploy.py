from __future__ import annotations

"""Unified deploy-time uncertainty entrypoint.

Long-term maintenance rule:
- deploy-time uncertainty extensions must enter through this module
- do not add deploy-only uncertainty implementations elsewhere
"""

from typing import Any, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

from curator.data import properties
from curator.data._uncertainty import collect_uncertainty_outputs


def _as_plain_spec(spec: Optional[Any]) -> Optional[dict[str, Any]]:
    if spec is None:
        return None
    if isinstance(spec, DictConfig):
        spec = OmegaConf.to_container(spec, resolve=False)
    if not isinstance(spec, Mapping):
        raise TypeError(f"deploy.uncertainty must be a mapping, got {type(spec)}")
    plain = dict(spec)
    method = plain.get("method", "none")
    plain["method"] = None if method is None else str(method).strip().lower()
    output_keys = plain.get("output_keys")
    if output_keys is not None:
        plain["output_keys"] = [str(key) for key in output_keys]
    return plain


def _prepare_ensemble(model, spec: dict[str, Any]) -> None:
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


def _prepare_mahalanobis(model, spec: dict[str, Any], *, lammps_mliap: bool) -> None:
    from curator.layer import FeatureCalculator
    from curator.layer._feature import normalize_kernel
    from curator.model import EnsembleModel

    if isinstance(model, EnsembleModel):
        raise ValueError("deploy.uncertainty.method=mahalanobis requires a single model, not an ensemble.")

    dataset = spec.get("dataset")
    if dataset in (None, "", "none", "null"):
        raise ValueError("deploy.uncertainty.method=mahalanobis requires deploy.uncertainty.dataset.")

    output_keys = spec.get("output_keys")

    maha_cfg = spec.get("maha") or {}
    kernel = str(maha_cfg.get("kernel", "local-full-g"))
    normalized_kernel = normalize_kernel(kernel)
    local_kernel = normalized_kernel.startswith("local_")
    max_structures = maha_cfg.get("max_structures", None)
    regularization = float(maha_cfg.get("regularization", 1e-6))
    streaming = bool(maha_cfg.get("streaming", False))
    pair_scriptable = normalized_kernel in {"gnn", "local_gnn"}
    allowed_output_keys = {properties.maha_dist}
    if local_kernel:
        allowed_output_keys.add(properties.maha_dist_per_atom)

    if output_keys is not None:
        invalid_output_keys = [key for key in output_keys if key not in allowed_output_keys]
        if invalid_output_keys:
            raise ValueError(
                f"Mahalanobis deploy output_keys {invalid_output_keys} are not supported for kernel={kernel}."
            )

    if not lammps_mliap and not pair_scriptable:
        raise RuntimeError(
            "pair_style curator Mahalanobis is TorchScript-safe only for kernel=gnn/local-gnn. "
            "Hook-based full-g/local-full-g remains MLIAP-only."
        )

    n_random_features = 0 if pair_scriptable else 500

    for module in model.output_modules:
        if isinstance(module, FeatureCalculator):
            module.dataset = dataset
            module.compute_maha_dist = True
            module.output_features = False
            module.distance_kernel = kernel
            module.update_uncertainty_outputs()
            module.max_dataset_size = max_structures
            module.streaming = streaming
            module.regularization = regularization
            if properties.feature in module.model_outputs:
                module.model_outputs.remove(properties.feature)
            module.kernels = module._build_kernels(
                [(kernel, n_random_features)],
                repr_callback=model,
                target_layer=module.extractor.target_layer,
                target_domain=module.extractor.target_domain,
            )
            model.register_callbacks(module)
            return

    feature_calculator = FeatureCalculator(
        kernels=[(kernel, n_random_features)],
        dataset=dataset,
        compute_maha_dist=True,
        output_features=False,
        distance_kernel=kernel,
        max_dataset_size=max_structures,
        streaming=streaming,
        regularization=regularization,
    )
    model.output_modules.append(feature_calculator)


def prepare_deploy_uncertainty(
    model,
    spec: Optional[Any],
    *,
    lammps_mliap: bool = False,
) -> None:
    spec = _as_plain_spec(spec)
    if spec is None:
        return

    method = spec.get("method", "none")
    if method in ("", "none", None):
        return
    if method == "ensemble":
        _prepare_ensemble(model, spec)
        return
    if method == "mahalanobis":
        _prepare_mahalanobis(model, spec, lammps_mliap=lammps_mliap)
        return
    raise ValueError(f"Unknown deploy uncertainty method '{method}'.")
