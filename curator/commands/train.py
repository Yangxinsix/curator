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


def _resolve_data_dtype(datamodule: Any) -> Any:
    if hasattr(datamodule, "default_dtype"):
        return datamodule.default_dtype
    if hasattr(datamodule, "domain_modules") and datamodule.domain_modules:
        first_domain = next(iter(datamodule.domain_modules.values()))
        return getattr(first_domain, "default_dtype", None)
    return None


def _load_model(config: DictConfig, *, build_dtype=None) -> LoadedModel:
    import torch
    from collections import OrderedDict
    from hydra.utils import instantiate

    from ..layer.wrappers.utils import temporary_default_dtype
    from ..layer.wrappers import get_config_wrapper_config, get_model_wrapper_config
    from ..model.external import is_external_model_spec, load_external_model, parse_external_model_spec
    from ..model import NeuralNetworkPotential
    from ..utils import find_best_model

    if config.model_path is None:
        with temporary_default_dtype(build_dtype):
            return LoadedModel(model=instantiate(config.model))

    if isinstance(config.model_path, str) and is_external_model_spec(config.model_path):
        checkpoint_mode = str(config.task.checkpoint_mode).strip().lower()
        spec = parse_external_model_spec(config.model_path)
        if spec is None:
            raise ValueError(f"Invalid external model spec: {config.model_path}")
        if spec.scheme not in {"mace", "nequip"}:
            raise ValueError(
                f"Fine-tuning only supports external pretrained specs for 'mace' and 'nequip', got {spec.scheme!r}."
            )
        if checkpoint_mode != "model":
            raise ValueError(
                f"External pretrained spec {config.model_path!r} requires task.checkpoint_mode='model'."
            )
        log.debug("Loading external pretrained model from spec %s", config.model_path)
        with temporary_default_dtype(build_dtype):
            model = load_external_model(config.model_path, device=torch.device("cpu"))
        if not isinstance(model, NeuralNetworkPotential):
            raise TypeError(f"Expected NeuralNetworkPotential, got {type(model)}")
        return LoadedModel(model=model)

    model_path = find_best_model(config.model_path)[0]
    config.model_path = model_path
    checkpoint_mode = str(config.task.checkpoint_mode).strip().lower()
    if checkpoint_mode not in {"weights", "model", "resume"}:
        raise ValueError(
            f"Unknown task.checkpoint_mode={checkpoint_mode!r}; expected one of "
            f"['weights', 'model', 'resume']."
        )
    log.debug("Loading trained model from %s", model_path)
    log.debug("Checkpoint loading mode: %s", checkpoint_mode)

    if checkpoint_mode == "resume":
        log.debug(
            "Resuming training from checkpoint; optimizer and scheduler states will be restored from %s",
            model_path,
        )
        with temporary_default_dtype(build_dtype):
            model = instantiate(config.model)
        return LoadedModel(
            model=model,
            resume_from=model_path,
        )

    checkpoint_data = torch_load_compat(torch, model_path, weights_only=False)
    if isinstance(checkpoint_data, torch.nn.Module):
        wrapper_from_checkpoint = get_model_wrapper_config(checkpoint_data)
    elif isinstance(checkpoint_data, dict):
        wrapper_from_checkpoint = get_config_wrapper_config(checkpoint_data.get("wrapper_config"))
    else:
        wrapper_from_checkpoint = None

    if checkpoint_mode == "weights":
        state_dict = None
        if isinstance(checkpoint_data, torch.nn.Module):
            state_dict = checkpoint_data.state_dict()
        elif isinstance(checkpoint_data, dict):
            saved_model = checkpoint_data.get("model")
            state_dict = checkpoint_data.get("state_dict")
            if state_dict is None and isinstance(saved_model, torch.nn.Module):
                state_dict = saved_model.state_dict()
        if state_dict is None:
            raise ValueError("Checkpoint is missing a state_dict for weights loading.")
        with temporary_default_dtype(build_dtype):
            model = instantiate(config.model)
        model.load_state_dict(
            OrderedDict(
                (name.replace("model.", "", 1), value)
                for name, value in state_dict.items()
            ),
            strict=False,
        )
        return LoadedModel(
            model=model,
            wrapper_from_checkpoint=wrapper_from_checkpoint,
        )

    if isinstance(checkpoint_data, torch.nn.Module):
        model = checkpoint_data
        outputs = getattr(checkpoint_data, "outputs", None)
    elif isinstance(checkpoint_data, dict):
        outputs = checkpoint_data.get("outputs")
        saved_model = checkpoint_data.get("model")
        if isinstance(saved_model, torch.nn.Module):
            model = saved_model
        else:
            model_config = checkpoint_data.get("model_params") or checkpoint_data.get("model_cfg")
            state_dict = checkpoint_data.get("state_dict")
            if model_config is None:
                raise ValueError(
                    "Checkpoint does not include a model object or model_params/model_cfg needed "
                    "to reconstruct the native model."
                )
            if state_dict is None:
                raise ValueError(
                    "Checkpoint is missing state_dict needed to reconstruct the native model."
                )
            with temporary_default_dtype(build_dtype):
                model = instantiate(model_config, _convert_="all")
            model.load_state_dict(
                OrderedDict(
                    (name.replace("model.", "", 1), value)
                    for name, value in state_dict.items()
                ),
                strict=False,
            )
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint_data)}")

    if not isinstance(model, NeuralNetworkPotential):
        raise TypeError(f"Expected NeuralNetworkPotential, got {type(model)}")
    return LoadedModel(
        model=model,
        outputs=outputs,
        wrapper_from_checkpoint=wrapper_from_checkpoint,
    )


def _prepare_model(
    loaded: LoadedModel,
    *,
    config: DictConfig,
    datamodule: Any,
    finetune_mode: Optional[str],
):
    import torch

    from ..layer.wrappers import get_config_wrapper_config, get_model_wrapper_config, resolve_wrapper_config
    from ..model import align_model_domains_from_datamodule
    from ..model.conversion import convert_model_wrapper

    model = loaded.model
    checkpoint_wrapper = loaded.wrapper_from_checkpoint
    data_dtype = _resolve_data_dtype(datamodule)

    requested_wrapper = get_config_wrapper_config(config)
    wrapper_to_apply = requested_wrapper or checkpoint_wrapper

    model = align_model_domains_from_datamodule(model, datamodule, logger=log)

    if finetune_mode == "lora":
        source_wrapper = wrapper_to_apply or get_model_wrapper_config(model)
        wrapper_to_apply = resolve_wrapper_config(
            backend=source_wrapper.backend,
            adapter="lora",
            lora_rank=int(getattr(config, "lora_rank", 16)),
            lora_alpha=float(getattr(config, "lora_alpha", 16.0)),
            lora_freeze_base=bool(getattr(config, "lora_freeze_base", False)),
        )

    if wrapper_to_apply is None:
        current_wrapper = get_model_wrapper_config(model)
        if current_wrapper.backend == "cueq" and data_dtype is not None:
            wrapper_to_apply = current_wrapper

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
        normalize_config_sequences,
        prune_config_targets,
        read_user_config,
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
        finetune_mode=finetune_mode,
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
    from ..utils import CustomFormatter, normalize_config_sequences, read_user_config

    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="train")

    normalize_config_sequences(config)
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
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    with temporary_default_dtype(_resolve_data_dtype(datamodule)):
        model = instantiate(config.model)
    model.initialize_modules(datamodule)
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
