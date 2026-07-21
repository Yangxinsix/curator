from __future__ import annotations

"""Model-level uncertainty injection for Curator potentials.

This module intentionally does not know about ASE, TorchSim, LAMMPS, or any
deployment target. It mutates a PyTorch-native NeuralNetworkPotential so the
model's ordinary forward path emits uncertainty outputs.
"""

from typing import Any, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

from curator.data import properties
from curator.model import NeuralNetworkPotential
import logging


logger = logging.getLogger(__name__)


def _feature_spec_for_kernel(kernel: str, n_random_features: int) -> dict[str, Any]:
    from curator.layer._feature import normalize_kernel

    normalized = normalize_kernel(kernel)
    local = normalized.startswith("local_")
    raw_feature = normalized[len("local_") :] if local else normalized
    mapping = "identity" if n_random_features <= 0 else "gaussian_sketch"
    return {
        "name": normalized,
        "raw_feature": raw_feature,
        "mapping": mapping,
        "num_features": max(1, int(n_random_features)),
        "layer_combine": "concat",
        "layer_norm": "none",
        "pooling": "mean" if local else "sum",
        "sigma": 1.0,
        "seed": 0,
    }


def _as_plain_spec(spec: Optional[Any]) -> Optional[dict[str, Any]]:
    if spec is None:
        return None
    if isinstance(spec, DictConfig):
        spec = OmegaConf.to_container(spec, resolve=False)
    if not isinstance(spec, Mapping):
        raise TypeError(f"uncertainty spec must be a mapping, got {type(spec)}")
    plain = dict(spec)
    method = plain.get("method", "none")
    plain["method"] = None if method is None else str(method).strip().lower()
    output_keys = plain.get("output_keys")
    if output_keys is not None:
        plain["output_keys"] = [str(key) for key in output_keys]
    return plain


def _prepare_mahalanobis(model: NeuralNetworkPotential, spec: dict[str, Any], *, implementation: str) -> None:
    from curator.layer import FeatureCalculator
    from curator.layer._feature import normalize_kernel

    dataset = spec.get("dataset")
    if dataset in (None, "", "none", "null"):
        raise ValueError("Mahalanobis uncertainty injection requires a reference dataset.")

    output_keys = spec.get("output_keys")
    maha_cfg = spec.get("maha") or {}
    kernel = str(maha_cfg.get("kernel", "local-full-g"))
    normalized_kernel = normalize_kernel(kernel)
    local_kernel = normalized_kernel.startswith("local_")
    max_structures = maha_cfg.get("max_structures", None)
    regularization = float(maha_cfg.get("regularization", 1e-6))
    streaming = bool(maha_cfg.get("streaming", False))

    allowed_output_keys = {properties.maha_dist}
    if implementation == "scriptable" and local_kernel:
        allowed_output_keys.add(properties.maha_dist_per_atom)

    if output_keys is not None:
        invalid_output_keys = [key for key in output_keys if key not in allowed_output_keys]
        if invalid_output_keys:
            raise ValueError(
                f"Mahalanobis uncertainty output_keys {invalid_output_keys} are not supported "
                f"for implementation={implementation} kernel={kernel}."
            )

    if implementation == "scriptable":
        if normalized_kernel not in {"gnn", "local_gnn"}:
            raise RuntimeError(
                "scriptable Mahalanobis injection supports only gnn/local-gnn kernels. "
                "Use implementation=native for full-g/local-full-g."
            )
        from curator.simulate.uncertainty.node_mahalanobis import fit_node_feature_mahalanobis

        output_per_atom = bool(local_kernel and (output_keys is None or properties.maha_dist_per_atom in output_keys))
        scorer = fit_node_feature_mahalanobis(
            model,
            dataset,
            local=local_kernel,
            output_per_atom=output_per_atom,
            max_structures=max_structures,
            regularization=regularization,
        )
        model.output_modules.append(scorer)
        return

    if implementation != "native":
        raise ValueError("implementation must be one of: native, scriptable.")

    feature_spec = _feature_spec_for_kernel(kernel, n_random_features=500)
    logger.info(
        "Injecting native Mahalanobis uncertainty: dataset=%s kernel=%s max_structures=%s streaming=%s",
        dataset,
        kernel,
        max_structures if max_structures is not None else "all",
        streaming,
    )

    for module in model.output_modules:
        if isinstance(module, FeatureCalculator):
            module.dataset = dataset
            module.compute_maha_dist = True
            module.output_features = False
            module.distance_kernel = kernel
            if hasattr(module, "update_uncertainty_outputs"):
                module.update_uncertainty_outputs()
            elif properties.maha_dist not in module.model_outputs:
                module.model_outputs.append(properties.maha_dist)
            module.max_dataset_size = max_structures
            module.streaming = streaming
            module.regularization = regularization
            if properties.feature in module.model_outputs:
                module.model_outputs.remove(properties.feature)
            module.kernels = module._build_kernels([feature_spec])
            model.register_callbacks(module)
            return

    feature_calculator = FeatureCalculator(
        kernels=[feature_spec],
        dataset=dataset,
        compute_maha_dist=True,
        output_features=False,
        distance_kernel=kernel,
        max_dataset_size=max_structures,
        streaming=streaming,
        regularization=regularization,
    )
    model.output_modules.append(feature_calculator)


def inject_uncertainty(
    model: NeuralNetworkPotential,
    spec: Optional[Any],
    *,
    implementation: str = "native",
) -> NeuralNetworkPotential:
    """Attach uncertainty outputs to a PyTorch-native NeuralNetworkPotential."""

    if not isinstance(model, NeuralNetworkPotential):
        raise TypeError(f"uncertainty injection expects NeuralNetworkPotential, got {type(model)}")

    spec = _as_plain_spec(spec)
    if spec is None:
        return model

    method = spec.get("method", "none")
    if method in ("", "none", None):
        return model
    if method == "mahalanobis":
        _prepare_mahalanobis(model, spec, implementation=str(implementation or "native").strip().lower())
        return model
    if method == "ensemble":
        raise ValueError(
            "ensemble uncertainty is a model-composition strategy, not single NeuralNetworkPotential injection."
        )
    raise ValueError(f"Unknown uncertainty method '{method}'.")
