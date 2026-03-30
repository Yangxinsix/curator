import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from omegaconf import DictConfig, ListConfig, open_dict

from curator.data import properties


log = logging.getLogger(__name__)
DISTILL_OUTPUT_TARGET = "curator.train.model_output.DistillOutput"

# Get all distillation outputs from the config, skipping defaults and non-distillation outputs.
def iter_distill_outputs(outputs: Optional[DictConfig]) -> Iterable[DictConfig]:
    if outputs is None:
        return
    if isinstance(outputs, DictConfig):
        iterable = outputs.items()
    elif isinstance(outputs, (ListConfig, list)):
        iterable = enumerate(outputs)
    else:
        return
    for name, cfg in iterable:
        if name == "defaults" or not isinstance(cfg, DictConfig):
            continue
        if str(cfg.get("_target_", "")) == DISTILL_OUTPUT_TARGET:
            yield cfg


def distill_output_source_property(cfg: DictConfig) -> str:
    student_property = str(cfg.get("student_property") or cfg.get("name"))
    return str(cfg.get("teacher_output_property") or student_property)


def is_offline_distill_output(cfg: DictConfig) -> bool:
    student_property = str(cfg.get("student_property") or cfg.get("name"))
    teacher_output_property = distill_output_source_property(cfg)
    sampling_enabled = any(
        cfg.get(key) is not None
        for key in ("num_samples", "sample_indices", "sample_index_key", "sample_fn")
    )
    if not sampling_enabled:
        return True
    return teacher_output_property != student_property


def prepare_teacher_model_for_offline_distillation(model, output_columns: Dict[str, str]) -> None:
    if properties.energy_hessian not in output_columns:
        return
    if any(
        not hasattr(model, attr) for attr in ("output_modules", "model_outputs", "register_callbacks")
    ):
        raise TypeError(
            "Offline Hessian distillation requires teacher models with mutable output modules."
        )
    from curator.layer import EnergyHessianOutput, GradientOutput

    gradient_module = None
    gradient_index = None
    for i, module in enumerate(model.output_modules):
        if isinstance(module, GradientOutput) and not getattr(module, "compute_edge_forces_only", False):
            gradient_module = module
            gradient_index = i
            break
    if gradient_module is None:
        insert_at = len(model.output_modules)
        model.output_modules.insert(
            insert_at,
            GradientOutput(
                grad_on_edge_diff=False,
                grad_on_positions=True,
                model_outputs=[properties.forces],
            )
        )
        gradient_index = insert_at
    elif properties.forces not in gradient_module.model_outputs:
        gradient_module.update_model_outputs(properties.forces)

    hessian_module = None
    for module in model.output_modules:
        if isinstance(module, EnergyHessianOutput):
            hessian_module = module
            break
    if hessian_module is None:
        insert_at = len(model.output_modules) if gradient_index is None else gradient_index + 1
        model.output_modules.insert(
            insert_at,
            EnergyHessianOutput(model_outputs=[properties.energy_hessian]),
        )
        hessian_module = model.output_modules[insert_at]
    elif properties.energy_hessian not in hessian_module.model_outputs:
        hessian_module.update_model_outputs(properties.energy_hessian)
    hessian_module.vectorize = False

# Collect the mapping of student properties to teacher properties from the distillation outputs.
def collect_distill_output_columns(outputs: Optional[DictConfig]) -> Dict[str, str]:
    columns: Dict[str, str] = {}
    for cfg in iter_distill_outputs(outputs):
        if not is_offline_distill_output(cfg):
            continue
        source_property = distill_output_source_property(cfg)
        teacher_property = str(cfg.get("teacher_property") or source_property)
        existing = columns.get(source_property)
        if existing is not None and existing != teacher_property:
            raise ValueError(
                f"Distill outputs map '{source_property}' to multiple teacher columns: "
                f"'{existing}' and '{teacher_property}'."
            )
        columns[source_property] = teacher_property
    return columns


def resolve_distill_sqlite_paths(
    data_cfg: DictConfig,
    run_path: str,
) -> Dict[str, Path]:
    has_datapath = getattr(data_cfg, "datapath", None) is not None
    split_keys = [
        key for key in ("train_path", "val_path", "test_path") if getattr(data_cfg, key, None) is not None
    ]
    distill_dir = Path(run_path) / "distill_dataset"

    if not has_datapath and not split_keys:
        raise ValueError("Distillation requires `data.datapath` or explicit `data.train_path` / `val_path` / `test_path`.")
    if has_datapath and split_keys:
        raise ValueError("Use either `data.datapath` or split paths for distillation, not both.")
    if has_datapath:
        return {"datapath": distill_dir / "dataset.sqlite"}
    return {key: distill_dir / f"{key.replace('_path', '')}.sqlite" for key in split_keys}


def resolve_teacher_labels_path(
    data_cfg: DictConfig,
    run_path: str,
    teacher_labels_path: Optional[str],
    logger: logging.Logger,
) -> Dict[str, Path]:
    # Default output lives under run_path/distill_dataset; teacher_labels_path only overrides that location.
    sqlite_paths = resolve_distill_sqlite_paths(data_cfg, run_path)
    if teacher_labels_path is None:
        logger.info("Using default offline distillation sqlite path at %s", next(iter(sqlite_paths.values())).parent)
        return sqlite_paths

    teacher_labels_path = Path(teacher_labels_path).expanduser()
    if not teacher_labels_path.is_absolute():
        teacher_labels_path = Path(run_path) / teacher_labels_path
    if "datapath" in sqlite_paths:
        if teacher_labels_path.suffix != ".sqlite":
            raise ValueError(
                "`task.distill.teacher_labels_path` must be a sqlite file when using `data.datapath`."
            )
        sqlite_paths["datapath"] = teacher_labels_path
        return sqlite_paths
    if teacher_labels_path.suffix == ".sqlite":
        raise ValueError(
            "`task.distill.teacher_labels_path` must be a directory when using split datasets "
            "(`data.train_path` / `val_path` / `test_path`)."
        )
    for key in sqlite_paths:
        sqlite_paths[key] = teacher_labels_path / sqlite_paths[key].name
    return sqlite_paths


def prepare_distillation(config: DictConfig, logger: Optional[logging.Logger] = None) -> None:
    logger = logger or log
    output_columns = collect_distill_output_columns(getattr(config.task, "outputs", None))
    if not output_columns:
        return

    distill_cfg = getattr(config.task, "distill", None)
    if distill_cfg is None:
        raise ValueError("Distill outputs require `task.distill` to be configured.")

    sqlite_paths = resolve_teacher_labels_path(
        config.data,
        config.run_path,
        distill_cfg.get("teacher_labels_path"),
        logger,
    )
    teacher_model_path = distill_cfg.get("teacher_model_path")
    overwrite = bool(distill_cfg.get("overwrite", False))

    if teacher_model_path is not None:
        from curator.evaluate import Evaluator
        from curator.model import EnsembleModel
        from curator.utils import load_models

        device = config.device
        if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA is not available; generating distillation labels on CPU.")
            device = "cpu"

        models = load_models(
            teacher_model_path,
            device=device,
            load_compiled=False,
            load_weights_only=False,
            cfg=distill_cfg.get("teacher_cfg"),
        )
        for teacher_model in models:
            prepare_teacher_model_for_offline_distillation(teacher_model, output_columns)
        model = EnsembleModel(models) if len(models) > 1 else models[0]
        evaluator = Evaluator(
            model=model,
            save_data=False,
            plot_figure=False,
            output_dir=Path(config.run_path) / "distill",
            device=device,
            batch_size=int(getattr(config.data, "batch_size", 8)),
            num_workers=int(getattr(config.data, "num_workers", 0)),
            pin_memory=bool(getattr(config.data, "pin_memory", False)),
        )
        # teacher_model_path is set: missing sqlite files can be generated here; existing ones are reused
        # unless overwrite=true.
        for key, sqlite_path in sqlite_paths.items():
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            if sqlite_path.exists():
                if not overwrite:
                    logger.info("Reusing offline distillation sqlite at %s", sqlite_path)
                    continue
                sqlite_path.unlink()
            logger.info("Generating offline distillation sqlite at %s", sqlite_path)
            evaluator.evaluate(
                getattr(config.data, key),
                sqlite_output=sqlite_path,
                output_columns=output_columns,
            )
    else:
        # teacher_model_path is not set: do not run teacher inference, only reuse existing sqlite labels.
        missing = [str(path) for path in sqlite_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Offline distillation sqlite does not exist. "
                "Set `task.distill.teacher_model_path` to generate it or provide an existing file: "
                + ", ".join(missing)
            )

    with open_dict(config):
        config.data.data_type = "Sqlite3"
        if "datapath" in sqlite_paths:
            config.data.datapath = str(sqlite_paths["datapath"])
            config.data.train_path = None
            config.data.val_path = None
            config.data.test_path = None
        else:
            config.data.datapath = None
            for key in ("train_path", "val_path", "test_path"):
                if key in sqlite_paths:
                    setattr(config.data, key, str(sqlite_paths[key]))
        for output_cfg in iter_distill_outputs(config.task.outputs):
            if is_offline_distill_output(output_cfg):
                output_cfg.teacher_model_path = None
                output_cfg.teacher_cfg = None
