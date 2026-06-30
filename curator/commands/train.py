import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from .common import (
    CONFIGS_PATH,
    configure_cli_logger,
    ensure_resolvers,
    log,
    log_logo,
    prepare_cli_environment,
    prepare_run_path,
    torch_load_compat,
)
from .deploy import deploy

_ALLOWED_FINETUNE = {None, "full", "head_only", "lora"}


@dataclass
class LoadedModel:
    model: Any
    outputs: Any = None
    resume_from: Optional[Path] = None
    wrapper_from_checkpoint: Any = None
    wrapper_transform_applied: bool = False


def _resolve_data_dtype(datamodule: Any) -> Any:
    if hasattr(datamodule, "default_dtype"):
        return datamodule.default_dtype
    if hasattr(datamodule, "domain_modules") and datamodule.domain_modules:
        first_domain = next(iter(datamodule.domain_modules.values()))
        return getattr(first_domain, "default_dtype", None)
    return None


def _load_model(config: DictConfig, *, build_dtype=None) -> LoadedModel:
    import torch
    from hydra.utils import instantiate

    from ..layer.wrappers import get_config_wrapper_config, get_model_wrapper_config
    from ..layer.wrappers.utils import temporary_default_dtype
    from ..model import NeuralNetworkPotential
    from ..model.conversion import (
        apply_model_transforms,
        convert_model_wrapper,
        load_pretrained_weights_from_model,
    )
    from ..model.external import is_external_model_spec, load_external_model, parse_external_model_spec
    from ..utils import find_best_model

    def instantiate_model(model_config: Any) -> Any:
        with temporary_default_dtype(build_dtype):
            return instantiate(model_config, _convert_="all")

    def checkpoint_state_dict(checkpoint_payload: Any, *, context: str) -> Any:
        if isinstance(checkpoint_payload, torch.nn.Module):
            return checkpoint_payload.state_dict()
        if isinstance(checkpoint_payload, dict):
            state_dict = checkpoint_payload.get("state_dict")
            saved_model = checkpoint_payload.get("model")
            if state_dict is None and isinstance(saved_model, torch.nn.Module):
                state_dict = saved_model.state_dict()
            if state_dict is not None:
                return {
                    name.replace("model.", "", 1): value
                    for name, value in state_dict.items()
                }
        raise ValueError(f"Checkpoint is missing a state_dict for {context}.")

    def apply_transforms(model: Any) -> Any:
        if not transforms:
            return model
        return apply_model_transforms(model, transforms, target_dtype=build_dtype)

    if config.model_path is None:
        source_path, source_mode, transforms = None, "model", []
    elif isinstance(config.model_path, (str, Path)):
        source_path, source_mode, transforms = config.model_path, "model", []
    else:
        payload = (
            OmegaConf.to_container(config.model_path, resolve=False)
            if isinstance(config.model_path, DictConfig)
            else config.model_path
        )
        if not isinstance(payload, dict):
            raise TypeError(
                "model_path must be None, a string/path, or a dict with keys "
                "['path', 'mode', 'transform']."
            )
        source_path = payload.get("path")
        source_mode = str(payload.get("mode", "model")).strip().lower()
        transforms = payload.get("transform")
        if transforms is None:
            transforms = []
        elif not isinstance(transforms, list):
            transforms = [transforms]

    if source_mode not in {"weights", "model", "resume"}:
        raise ValueError(
            f"Unknown model_path.mode={source_mode!r}; expected one of "
            f"['weights', 'model', 'resume']."
        )
    wrapper_transform_applied = any(
        transform in {"wrapper", "model_wrapper"}
        if isinstance(transform, str)
        else str(transform.get("type", "")).strip() in {"wrapper", "model_wrapper"}
        for transform in transforms
    )

    if source_path is None:
        return LoadedModel(model=instantiate_model(config.model))

    if isinstance(source_path, str) and is_external_model_spec(source_path):
        parsed = parse_external_model_spec(source_path)
        if parsed is None:
            raise ValueError(f"Invalid external model spec: {source_path}")
        if parsed.scheme not in {"mace", "nequip"}:
            raise ValueError(
                f"Fine-tuning only supports external pretrained specs for 'mace' and 'nequip', got {parsed.scheme!r}."
            )
        checkpoint_mode = str(OmegaConf.select(config, "task.checkpoint_mode", default="model")).strip().lower()
        if checkpoint_mode != "model":
            raise ValueError("External pretrained spec requires task.checkpoint_mode='model'.")
        if source_mode == "resume":
            raise ValueError(
                f"External pretrained spec {source_path!r} does not support model_path.mode='resume'."
            )

        log.debug("Loading external pretrained model from spec %s", source_path)
        with temporary_default_dtype(build_dtype):
            source_model = load_external_model(source_path, device=torch.device("cpu"))
        if not isinstance(source_model, NeuralNetworkPotential):
            raise TypeError(f"Expected NeuralNetworkPotential, got {type(source_model)}")

        loaded = LoadedModel(
            model=apply_transforms(source_model),
            wrapper_from_checkpoint=None,
            wrapper_transform_applied=wrapper_transform_applied,
        )
        loaded.wrapper_from_checkpoint = get_model_wrapper_config(loaded.model)
        if source_mode == "model" or transforms:
            return loaded
        model = instantiate_model(config.model)
        loaded_tensors = load_pretrained_weights_from_model(model, loaded.model)
        log.debug(
            "Loaded %d matching tensors from external pretrained model into config-instantiated target.",
            loaded_tensors,
        )
        return LoadedModel(
            model=model,
            wrapper_from_checkpoint=loaded.wrapper_from_checkpoint,
            wrapper_transform_applied=loaded.wrapper_transform_applied,
        )

    model_path = find_best_model(source_path)[0]
    log.debug("Loading trained model from %s", model_path)
    log.debug("Checkpoint loading mode: %s", source_mode)
    checkpoint_data = torch_load_compat(torch, model_path, weights_only=False)

    saved_model = checkpoint_data.get("model") if isinstance(checkpoint_data, dict) else None
    if isinstance(saved_model, torch.nn.Module):
        wrapper_from_checkpoint = get_model_wrapper_config(saved_model)
    elif isinstance(checkpoint_data, torch.nn.Module):
        wrapper_from_checkpoint = get_model_wrapper_config(checkpoint_data)
    elif isinstance(checkpoint_data, dict):
        wrapper_from_checkpoint = get_config_wrapper_config(checkpoint_data.get("wrapper_config"))
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint_data)}")

    if source_mode == "resume":
        log.debug(
            "Resuming training from checkpoint; optimizer and scheduler states will be restored from %s",
            model_path,
        )
        return LoadedModel(
            model=instantiate_model(config.model),
            resume_from=model_path,
            wrapper_from_checkpoint=wrapper_from_checkpoint,
        )

    if source_mode == "weights":
        model = instantiate_model(config.model)
        if wrapper_from_checkpoint is not None:
            model = convert_model_wrapper(model, wrapper_from_checkpoint, target_dtype=build_dtype)
        model.load_state_dict(
            checkpoint_state_dict(checkpoint_data, context="weights loading"),
            strict=False,
        )
        model = apply_transforms(model)
        return LoadedModel(
            model=model,
            wrapper_from_checkpoint=get_model_wrapper_config(model),
            wrapper_transform_applied=wrapper_transform_applied,
        )

    outputs = checkpoint_data.get("outputs") if isinstance(checkpoint_data, dict) else None
    if isinstance(checkpoint_data, torch.nn.Module):
        model = checkpoint_data
    elif isinstance(saved_model, torch.nn.Module):
        model = saved_model
    else:
        model_config = checkpoint_data.get("model_params") or checkpoint_data.get("model_cfg")
        if model_config is None:
            raise ValueError(
                "Checkpoint does not include a model object or model_params/model_cfg needed "
                "to reconstruct the native model."
            )
        model_payload = (
            OmegaConf.to_container(model_config, resolve=False)
            if isinstance(model_config, DictConfig)
            else model_config
        )
        config_context = OmegaConf.create({"model": model_payload})
        if checkpoint_data.get("data_params") is not None:
            config_context.data = checkpoint_data["data_params"]
        model = instantiate_model(config_context.model)
        if wrapper_from_checkpoint is not None:
            model = convert_model_wrapper(model, wrapper_from_checkpoint, target_dtype=build_dtype)
        model.load_state_dict(
            checkpoint_state_dict(checkpoint_data, context="native model reconstruction"),
            strict=False,
        )

    if not isinstance(model, NeuralNetworkPotential):
        raise TypeError(f"Expected NeuralNetworkPotential, got {type(model)}")
    model = apply_transforms(model)
    return LoadedModel(
        model=model,
        outputs=None if transforms else outputs,
        wrapper_from_checkpoint=get_model_wrapper_config(model),
        wrapper_transform_applied=wrapper_transform_applied,
    )


def _prepare_model(
    loaded: LoadedModel,
    *,
    config: DictConfig,
    datamodule: Any,
    data_dtype=None,
):
    import torch

    from ..layer.wrappers import get_config_wrapper_config, get_model_wrapper_config
    from ..model import align_model_domains_from_datamodule
    from ..model.conversion import convert_model_wrapper

    model = loaded.model
    wrapper_to_apply = loaded.wrapper_from_checkpoint or get_model_wrapper_config(model)
    if loaded.wrapper_transform_applied:
        wrapper_to_apply = None

    requested_payload = None
    if getattr(config, "wrapper", None) is not None:
        requested_payload = OmegaConf.to_container(config.wrapper, resolve=False)
    if isinstance(requested_payload, dict) and wrapper_to_apply is not None:
        requested_payload = {key: value for key, value in requested_payload.items() if value is not None}
        if requested_payload:
            if "backend" not in requested_payload:
                requested_payload["backend"] = wrapper_to_apply.backend
            requested_wrapper = get_config_wrapper_config(requested_payload)
            if requested_wrapper is not None:
                wrapper_to_apply = requested_wrapper

    model = align_model_domains_from_datamodule(model, datamodule, logger=log)
    if wrapper_to_apply is not None:
        model = convert_model_wrapper(model, wrapper_to_apply, target_dtype=data_dtype)

    model.initialize_modules(datamodule)
    log.debug("Initialized model modules from datamodule before task setup.")

    if data_dtype is not None:
        model = model.to(dtype=data_dtype)
        log.debug("Casting model dtype to data dtype %s", data_dtype)
    if config.compile:
        log.debug("Compiling model with torch.compile")
        model = torch.compile(model)
    return model


@hydra.main(config_path=CONFIGS_PATH, config_name="train", version_base=None)
def train(config: DictConfig) -> None:
    prepare_cli_environment()
    ensure_resolvers()
    import torch
    from hydra.utils import instantiate
    from pytorch_lightning import seed_everything
    from pytorch_lightning.loggers import WandbLogger

    from ..model import LitNNP
    from ..train.distill import prepare_distillation
    from ..utils import (
        CustomFormatter,
        find_best_model,
        update_config_from_datamodule,
    )

    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="train")

    normalize_config_sequences(config)
    prune_config_targets(config, logger=log)
    prepare_run_path(config.run_path)
    configure_cli_logger(log, os.path.join(config.run_path, "training.log"), CustomFormatter(), stream=True)
    log_logo(log)

    if hasattr(config, "trainer") and getattr(config.trainer, "accelerator", None) == "cpu":
        try:
            torch.cuda.is_available = lambda: False
            torch.cuda.device_count = lambda: 0
        except Exception:
            pass
    if "seed" in config:
        log.debug(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.debug("Seed randomly...")
    log.debug("Running on host: %s", socket.gethostname())

    prepare_distillation(config, logger=log)
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)

    log.debug(
        "Instantiating datamodule <%s> from dataset %s",
        config.data._target_,
        config.data.datapath or config.data.train_path,
    )
    datamodule = instantiate(config.data)
    datamodule.setup()
    update_config_from_datamodule(config, datamodule, logger=log)
    if hasattr(datamodule, "domain_modules") and OmegaConf.select(config, "task.normalize_domain_loss", default=None) is None:
        config.task.normalize_domain_loss = True
        log.info("Enabled task.normalize_domain_loss=True by default for multi-domain training.")
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    data_dtype = _resolve_data_dtype(datamodule)

    finetune_mode = str(getattr(config, "finetune", "") or "").strip().lower() or None
    if finetune_mode not in _ALLOWED_FINETUNE:
        raise ValueError(
            f"Unknown finetune mode {finetune_mode!r}; expected one of "
            f"[None, 'full', 'head_only', 'lora']."
        )
    log.info("Finetune mode: %s", finetune_mode or "full")

    loaded = _load_model(config, build_dtype=data_dtype)
    model = _prepare_model(
        loaded,
        config=config,
        datamodule=datamodule,
        data_dtype=data_dtype,
    )

    log.debug(f"Instantiating task <{config.task._target_}>")
    task: LitNNP = instantiate(config.task, model=model)
    if loaded.outputs is not None:
        if not isinstance(loaded.outputs, torch.nn.ModuleList):
            loaded.outputs = torch.nn.ModuleList(loaded.outputs)
        task.outputs = loaded.outputs

    log.debug(f"Instantiating model {type(model)} with GNN representation {type(model.representation)}")
    task.save_configuration(config)

    log.debug(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = instantiate(config.trainer)
    if isinstance(trainer.logger, WandbLogger):
        os.makedirs(trainer.logger.save_dir + "/wandb", exist_ok=True)

    trainer.fit(model=task, datamodule=datamodule, ckpt_path=loaded.resume_from)

    if config.deploy_model:
        best_model = find_best_model(run_path=config.run_path + "/model_path")
        if best_model is None:
            if getattr(trainer, "fast_dev_run", False):
                log.warning("Skipping deploy because fast_dev_run does not write checkpoints.")
            else:
                log.warning("Skipping deploy because no checkpoint was written to <%s/model_path>.", config.run_path)
        else:
            model_path, val_loss = best_model
            if val_loss is None:
                log.debug("Deploy trained model from %s", model_path)
            else:
                log.debug("Deploy trained model from %s with validation loss of %.3f", model_path, val_loss)
            deploy(
                model_path,
                f"{config.run_path}/compiled_model.pt",
                uncertainty_spec=OmegaConf.select(config, "deploy.uncertainty", default=None),
            )
            log.debug("Deploying compiled model at <%s/compiled_model.pt>", config.run_path)


@hydra.main(config_path=CONFIGS_PATH, config_name="train", version_base=None)
def tmp_train(config: DictConfig):
    prepare_cli_environment()
    ensure_resolvers()
    import torch
    from hydra.utils import instantiate
    from pytorch_lightning import seed_everything

    from ..layer.wrappers.utils import temporary_default_dtype
    from ..train import train as train_loop
    from ..utils import CustomFormatter, normalize_config_sequences, prune_config_targets, read_user_config

    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="train")

    normalize_config_sequences(config)
    prune_config_targets(config, logger=log)
    prepare_run_path(config.run_path)
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    log.debug("Running on host: " + str(socket.gethostname()))
    if "seed" in config:
        log.debug(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.debug("Seed randomly...")

    configure_cli_logger(log, os.path.join(config.run_path, "training.log"), CustomFormatter(), stream=True)
    log.debug(f"Instantiating datamodule <{config.data._target_}> from dataset {config.data.datapath or config.data.train_path}")
    datamodule = hydra.utils.instantiate(config.data)
    datamodule.setup()
    if hasattr(datamodule, "domain_modules") and OmegaConf.select(config, "task.normalize_domain_loss", default=None) is None:
        config.task.normalize_domain_loss = True
        log.info("Enabled task.normalize_domain_loss=True by default for multi-domain training.")
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    with temporary_default_dtype(_resolve_data_dtype(datamodule)):
        model = instantiate(config.model, _convert_="all")
    model = _prepare_model(
        LoadedModel(model=model),
        config=config,
        datamodule=datamodule,
        data_dtype=_resolve_data_dtype(datamodule),
    )
    outputs = instantiate(config.task.outputs)
    optimizer = instantiate(config.task.optimizer)(model.parameters())
    scheduler = instantiate(config.task.scheduler)(optimizer=optimizer)

    model = train_loop(
        model=model,
        outputs=outputs,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config.device,
        num_epochs=config.trainer.max_epochs,
        log_frequency=config.trainer.log_every_n_steps,
    )

    if config.deploy_model:
        model_path = [str(f) for f in Path(f"{config.run_path}").rglob("best_model.pth*")]
        if not model_path:
            log.warning("Skipping deploy because no best_model.pth checkpoint was written to <%s>.", config.run_path)
            return
        if len(model_path) > 1:
            log.warning("Multiple best models found, using the last one.")
        model_path = model_path[-1]
        deploy(
            model_path,
            f"{config.run_path}/compiled_model.pt",
            uncertainty_spec=OmegaConf.select(config, "deploy.uncertainty", default=None),
        )
        log.debug(f"Deploying compiled model at <{config.run_path}/compiled_model.pt>")


tmptrain = tmp_train
