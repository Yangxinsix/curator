import argparse
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Union

from omegaconf import DictConfig, OmegaConf

from .common import (
    argcomplete,
    ensure_cli_stream_logger,
    ensure_resolvers,
    ensure_torch_safe_globals,
    log,
    prepare_cli_environment,
)


@dataclass(frozen=True)
class DeployOptions:
    model_path: Union[str, list]
    target_path: str = "compiled_model.pt"
    load_weights_only: bool = False
    cfg_path: Optional[str] = None
    return_model: bool = False
    lammps_mliap: bool = False
    python_object: bool = False
    element_types: Optional[List[str]] = None
    uncertainty_spec: Optional[dict] = None


def _as_plain_mapping(spec: Optional[Any]) -> dict:
    if spec is None:
        return {}
    if isinstance(spec, DictConfig):
        spec = OmegaConf.to_container(spec, resolve=False)
    if not isinstance(spec, Mapping):
        raise TypeError(f"deploy.uncertainty must be a mapping, got {type(spec)}")
    return dict(spec)


def _resolve_target_path(target_path: str, *, lammps_mliap: bool, python_object: bool = False) -> str:
    if python_object and target_path == "compiled_model.pt":
        return "python_model.pth"
    if lammps_mliap and target_path == "compiled_model.pt":
        return "mliap_model.pt"
    return target_path


def resolve_uncertainty_spec(
    *,
    base_spec: Optional[Any] = None,
    override_spec: Optional[Any] = None,
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    lammps_mliap: bool = False,
    python_object: bool = False,
    allow_partial: bool = False,
) -> Optional[dict]:
    def merge(base: dict, override: dict) -> dict:
        merged = dict(base)
        for key, value in override.items():
            if value is None:
                continue
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                nested = dict(merged[key])
                for nested_key, nested_value in value.items():
                    if nested_value is not None:
                        nested[nested_key] = nested_value
                merged[key] = nested
            else:
                merged[key] = value
        return merged

    base_plain = _as_plain_mapping(base_spec)
    override_plain = _as_plain_mapping(override_spec)

    if method is not None:
        normalized_method = str(method).strip().lower()
        if normalized_method in ("", "none", "null"):
            override_plain = {"method": "none"}
        elif normalized_method == "ensemble":
            override_plain = merge({"method": "ensemble", "output_keys": None}, override_plain)
        elif normalized_method == "mahalanobis":
            default_kernel = "local-full-g" if lammps_mliap or python_object else "local-gnn"
            override_plain = merge(
                {
                    "method": "mahalanobis",
                    "dataset": None,
                    "output_keys": None,
                    "maha": {
                        "kernel": default_kernel,
                        "max_structures": None,
                        "regularization": 1e-6,
                        "streaming": False,
                    },
                },
                override_plain,
            )
        else:
            raise ValueError(f"Unknown uncertainty preset '{method}'.")
    elif dataset is not None and not allow_partial:
        merged_method = str(
            override_plain.get("method", base_plain.get("method", ""))
        ).strip().lower()
        if merged_method not in {"mahalanobis"}:
            raise ValueError(
                "--dataset requires --uncertainty mahalanobis "
                "or deploy.uncertainty.method=mahalanobis in cfg."
            )

    if dataset is not None:
        override_plain["dataset"] = dataset

    merged = merge(base_plain, override_plain)
    if not merged:
        return None

    normalized_method = str(merged.get("method", "none")).strip().lower()
    if normalized_method in ("", "none", "null"):
        return {"method": "none"}
    merged["method"] = normalized_method

    if normalized_method == "ensemble":
        merged.setdefault("output_keys", None)
        return merged

    if normalized_method == "mahalanobis":
        merged.setdefault("output_keys", None)
        maha_cfg = dict(merged.get("maha") or {})
        default_kernel = "local-full-g" if lammps_mliap or python_object else "local-gnn"
        maha_cfg.setdefault("kernel", default_kernel)
        maha_cfg.setdefault("max_structures", None)
        maha_cfg.setdefault("regularization", 1e-6)
        maha_cfg.setdefault("streaming", False)
        merged["maha"] = maha_cfg

    if normalized_method == "mahalanobis" and merged.get("dataset") in (None, "", "none", "null"):
        raise ValueError(
            "Mahalanobis deploy needs a reference dataset. "
            "Pass --dataset or set deploy.uncertainty.dataset in cfg."
        )
    return merged


def _parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Deploy CURATOR checkpoint(s) to either a TorchScript model for "
            "pair_style curator or a Python-backed model for pair_style mliap unified."
        ),
        epilog=(
            "Examples:\n"
            "  pair_style curator:\n"
            "    curator-deploy model.ckpt --target_path compiled_model.pt\n"
            "\n"
            "  mliap:\n"
            "    curator-deploy model.ckpt --mliap \\\n"
            "      --element-types Fe Li O P --target_path mliap_model.pt\n"
            "\n"
            "  mliap + mahalanobis:\n"
            "    curator-deploy model.ckpt --mliap \\\n"
            "      --element-types Fe Li O P --uncertainty mahalanobis \\\n"
            "      --dataset reference.traj --target_path mliap_model.pt\n"
            "\n"
            "  Python object for curator-simulate + mahalanobis:\n"
            "    curator-deploy model.ckpt --python-object \\\n"
            "      --uncertainty mahalanobis --dataset reference.traj \\\n"
            "      --target_path model_with_maha.pth\n"
            "\n"
            "  ensemble:\n"
            "    curator-deploy ckpt1.ckpt ckpt2.ckpt ckpt3.ckpt \\\n"
            "      --uncertainty ensemble --target_path compiled_ensemble.pt\n"
            "\n"
            "Notes:\n"
            "  - passing multiple INPUT_FILE values creates an EnsembleModel\n"
            "  - --mliap requires --element-types\n"
            "  - --python-object saves a torch.save model object for Python simulation callbacks\n"
            "  - --dataset is only needed for Mahalanobis\n"
            "  - use --cfg_path for advanced deploy.uncertainty settings"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars="+",
    )
    parser.add_argument("model_path", metavar="INPUT_FILE", type=str, nargs="+", help="One or more checkpoint/model paths to export")
    parser.add_argument(
        "--target_path",
        type=str,
        default="compiled_model.pt",
        help="Output path for the exported model; if left unchanged, --mliap rewrites compiled_model.pt to mliap_model.pt",
    )
    parser.add_argument("--load_weights_only", action="store_true", help="Rebuild the model from config/checkpoint metadata and load weights only")
    parser.add_argument("--cfg_path", type=str, help="Optional config file; use deploy.uncertainty there for detailed deploy tuning")
    parser.add_argument("--uncertainty", type=str, choices=["none", "ensemble", "mahalanobis"], default=None, help="Convenience uncertainty preset; keep detailed tuning in cfg_path")
    parser.add_argument("--dataset", type=str, default=None, help="Reference dataset for Mahalanobis fitting; not needed for ensemble")
    parser.add_argument("--mliap", action="store_true", help="Export an mliap unified model instead of TorchScript pair_style curator output")
    parser.add_argument("--python-object", action="store_true", help="Save a Python torch model object instead of TorchScript; useful for curator-simulate uncertainty callbacks")
    parser.add_argument("--element-types", type=str, nargs="+", default=None, help="Element symbols in LAMMPS type order; required when --mliap is set")
    if argcomplete:
        argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)
    if args.mliap and args.python_object:
        parser.error("--python-object cannot be combined with --mliap")
    return args


def _disable_internal_neighborlists(model_obj) -> int:
    from ..layer import PairwiseDistance

    disabled = 0
    for module in model_obj.modules():
        if isinstance(module, PairwiseDistance):
            module.compute_neighbor_list = False
            module.batch_nl = None
            module.compute_distance_from_R = False
            module.compute_forces = True
            disabled += 1
    return disabled


def _prepare_torchscript_model(model) -> None:
    import torch

    readout = getattr(getattr(model, "representation", None), "readout", None)
    if hasattr(readout, "domain_modules"):
        domain_modules = list(readout.domain_modules.values())
        if len(domain_modules) == 1:
            model.representation.readout = domain_modules[0]

    if hasattr(model, "output_modules"):
        for i, module in enumerate(model.output_modules):
            if hasattr(module, "domain_modules"):
                domain_modules = list(module.domain_modules.values())
                if len(domain_modules) == 1:
                    model.output_modules[i] = domain_modules[0]

    for module in model.modules():
        if module.__class__.__name__ == "AtomwiseNN":
            if not hasattr(module, "shared_mlp"):
                module.shared_mlp = torch.nn.Identity()
            if not hasattr(module, "head_modules"):
                module.head_modules = torch.nn.ModuleDict()
            if not hasattr(module, "shared_out_features"):
                module.shared_out_features = getattr(module, "in_features", 0)
        if module.__class__.__name__ == "GlobalRescaleShift" and hasattr(module, "_configure_sync_reduced_outputs"):
            module._configure_sync_reduced_outputs()


def deploy_main(argv: Optional[List[str]] = None):
    prepare_cli_environment()
    ensure_cli_stream_logger(log)
    args = _parse_args(argv)
    options = DeployOptions(
        model_path=args.model_path,
        target_path=_resolve_target_path(args.target_path, lammps_mliap=args.mliap, python_object=args.python_object),
        load_weights_only=args.load_weights_only,
        cfg_path=args.cfg_path,
        return_model=False,
        lammps_mliap=args.mliap,
        python_object=args.python_object,
        element_types=args.element_types,
        uncertainty_spec=resolve_uncertainty_spec(
            method=args.uncertainty,
            dataset=args.dataset,
            lammps_mliap=args.mliap,
            python_object=args.python_object,
            allow_partial=bool(args.cfg_path),
        ),
    )
    model = deploy(**options.__dict__)
    if options.lammps_mliap:
        export_kind = "mliap"
    elif options.python_object:
        export_kind = "python-object"
    else:
        export_kind = "torchscript"
    print(f"Deploy succeeded: type={export_kind} output={options.target_path}")
    return model


def deploy(
    model_path: Union[str, list],
    target_path: str = "compiled_model.pt",
    load_weights_only: bool = False,
    cfg_path: Optional[str] = None,
    return_model: bool = False,
    lammps_mliap: bool = False,
    python_object: bool = False,
    element_types: Optional[List[str]] = None,
    uncertainty_spec: Optional[dict] = None,
):
    prepare_cli_environment()
    ensure_resolvers()
    ensure_torch_safe_globals()
    import torch
    from e3nn.util.jit import script

    from ..config_utils import normalize_config_sequences, read_user_config
    from ..layer.utils import find_layer_by_name_recursive
    from ..model import EnsembleModel
    from ..simulate.uncertainty._deploy import prepare_deploy_uncertainty
    from ..utils import load_models

    if python_object and lammps_mliap:
        raise ValueError("python_object cannot be combined with lammps_mliap.")

    target_path = _resolve_target_path(target_path, lammps_mliap=lammps_mliap, python_object=python_object)

    cfg = None
    cfg_uncertainty_spec = None
    if cfg_path is not None:
        cfg = read_user_config(cfg_path, config_path="configs", config_name="train")
        normalize_config_sequences(cfg)
        cfg_uncertainty_spec = OmegaConf.select(cfg, "deploy.uncertainty", default=None)

    uncertainty_spec = resolve_uncertainty_spec(
        base_spec=cfg_uncertainty_spec,
        override_spec=uncertainty_spec,
        lammps_mliap=lammps_mliap,
        python_object=python_object,
        allow_partial=True,
    )

    models = load_models(
        model_path,
        device=None,
        load_compiled=False,
        load_weights_only=load_weights_only,
        cfg=cfg,
    )
    uncertainty_method = str((uncertainty_spec or {}).get("method", "none")).strip().lower()
    if len(models) > 1:
        model = EnsembleModel(
            models,
            per_atom_uncertainty=bool(lammps_mliap or uncertainty_method == "ensemble"),
        )
    else:
        model = models[0]

    if uncertainty_method not in ("", "none", "null"):
        log.info("Preparing deploy uncertainty via unified registry: %s", uncertainty_method)

    disabled_neighborlist_modules = _disable_internal_neighborlists(model)

    if lammps_mliap:
        if uncertainty_method not in ("", "none", "null"):
            prepare_deploy_uncertainty(model, uncertainty_spec, lammps_mliap=True)
        if not element_types:
            raise ValueError("element_types must be provided when exporting LAMMPS MLIAP models.")
        from ..simulate.lammps_mliap_interface import LAMMPS_MLIAP

        lmp_model = LAMMPS_MLIAP(model, element_types)
        torch.save(lmp_model, target_path)
        if disabled_neighborlist_modules:
            log.info(
                "Disabled internal PairwiseDistance neighbor-list construction in %d module(s) before LAMMPS MLIAP export.",
                disabled_neighborlist_modules,
            )
        return lmp_model

    if python_object:
        prepare_deploy_uncertainty(model, uncertainty_spec, lammps_mliap=False, torchscript=False)
        torch.save(model, target_path)
        log.debug(f"Deploying Python model object at <{target_path}> from <{model_path}>")
        if return_model:
            return model
        return None

    _prepare_torchscript_model(model)
    if disabled_neighborlist_modules:
        log.info(
            "Disabled internal PairwiseDistance neighbor-list construction in %d module(s) before deploy.",
            disabled_neighborlist_modules,
        )

    if uncertainty_method not in ("", "none", "null"):
        prepare_deploy_uncertainty(model, uncertainty_spec, lammps_mliap=False)
    model_compiled = script(model)
    metadata = {"cutoff": str(find_layer_by_name_recursive(model_compiled, "cutoff")).encode("ascii")}
    model_compiled.save(target_path, _extra_files=metadata)
    log.debug(f"Deploying compiled model at <{target_path}> from <{model_path}>")
    if return_model:
        return model_compiled
