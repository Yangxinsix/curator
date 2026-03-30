import os
import socket
from pathlib import Path

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


def _resolve_checkpoint_mode(task_cfg: DictConfig) -> str:
    mode = str(task_cfg.checkpoint_mode).strip().lower()
    if mode not in {"weights", "model", "resume"}:
        raise ValueError(
            f"Unknown task.checkpoint_mode={mode!r}; expected one of "
            f"['weights', 'model', 'resume']."
        )
    return mode


@hydra.main(config_path=CONFIGS_PATH, config_name="train", version_base=None)
def train(config: DictConfig) -> None:
    prepare_cli_environment()
    ensure_resolvers()
    import torch
    from hydra.utils import instantiate
    from pytorch_lightning import seed_everything
    from pytorch_lightning.loggers import WandbLogger

    from ..finetune import prepare_multi_domain_finetune
    from ..layer.wrappers import apply_wrappers
    from ..model import LitNNP, NeuralNetworkPotential
    from ..train.distill import prepare_distillation
    from ..utils import (
        CustomFormatter,
        find_best_model,
        normalize_config_sequences,
        prune_config_targets,
        read_user_config,
        resolve_wrapper_config_payload,
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

    finetune = str(getattr(config, "finetune", "") or "").strip().lower() or None
    use_multi_domain_finetune = finetune == "multi_domain"
    if finetune not in {None, "full", "head_only", "multi_domain", "elora"}:
        raise ValueError(
            f"Unknown finetune mode {finetune!r}; expected one of "
            f"[None, 'full', 'head_only', 'multi_domain', 'elora']."
        )

    resume_ckpt = None
    checkpoint_outputs = None
    checkpoint_wrapper_cfg = None
    runtime_wrapper_cfg = resolve_wrapper_config_payload(getattr(config, "addon", None))

    if config.model_path is not None:
        config.model_path = find_best_model(config.model_path)[0]
        checkpoint_mode = _resolve_checkpoint_mode(config.task)
        log.debug("Loading trained model from %s", config.model_path)
        log.debug("Checkpoint loading mode: %s", checkpoint_mode)
        checkpoint_obj = torch_load_compat(torch, config.model_path, weights_only=False)
        if isinstance(checkpoint_obj, torch.nn.Module):
            checkpoint_wrapper_cfg = resolve_wrapper_config_payload(checkpoint_obj)
        elif isinstance(checkpoint_obj, dict):
            checkpoint_wrapper_cfg = resolve_wrapper_config_payload(
                checkpoint_obj.get("wrapper_params"),
                checkpoint_obj.get("model"),
                checkpoint_obj.get("model_params") or checkpoint_obj.get("model_cfg"),
            )
        runtime_wrapper_cfg = resolve_wrapper_config_payload(
            getattr(config, "addon", None),
            checkpoint_wrapper_cfg,
        )

        load_native_model = (
            checkpoint_mode == "model"
            or use_multi_domain_finetune
            or runtime_wrapper_cfg is not None
        )
        if checkpoint_mode == "resume":
            model = instantiate(config.model)
            resume_ckpt = config.model_path
            log.debug(
                "Resuming training from checkpoint; optimizer and scheduler states will be restored from %s",
                resume_ckpt,
            )
        elif load_native_model:
            if isinstance(checkpoint_obj, torch.nn.Module):
                model = checkpoint_obj
                checkpoint_outputs = getattr(checkpoint_obj, "outputs", None)
            elif isinstance(checkpoint_obj, dict):
                checkpoint_outputs = checkpoint_obj.get("outputs")
                checkpoint_model = checkpoint_obj.get("model")
                if isinstance(checkpoint_model, torch.nn.Module):
                    model = checkpoint_model
                else:
                    from collections import OrderedDict

                    checkpoint_model_cfg = checkpoint_obj.get("model_params") or checkpoint_obj.get("model_cfg")
                    raw_state_dict = checkpoint_obj.get("state_dict")
                    if checkpoint_model_cfg is None:
                        raise ValueError(
                            "Checkpoint does not include a model object or model_params/model_cfg needed "
                            "to reconstruct the native model."
                        )
                    if raw_state_dict is None:
                        raise ValueError(
                            "Checkpoint is missing state_dict needed to reconstruct the native model."
                        )
                    model = instantiate(checkpoint_model_cfg, _convert_="all")
                    if checkpoint_wrapper_cfg is not None:
                        model = apply_wrappers(model, checkpoint_wrapper_cfg)
                    stripped_state_dict = OrderedDict(
                        (key.replace("model.", "", 1), value) for key, value in raw_state_dict.items()
                    )
                    model.load_state_dict(stripped_state_dict, strict=False)
            else:
                raise TypeError(f"Unsupported checkpoint type: {type(checkpoint_obj)}")
            if not isinstance(model, NeuralNetworkPotential):
                raise TypeError(f"Expected NeuralNetworkPotential, got {type(model)}")
        else:
            from collections import OrderedDict

            raw_state_dict = None
            checkpoint_model = None
            if isinstance(checkpoint_obj, torch.nn.Module):
                checkpoint_model = checkpoint_obj
                raw_state_dict = checkpoint_obj.state_dict()
            else:
                checkpoint_model = checkpoint_obj.get("model")
                raw_state_dict = checkpoint_obj.get("state_dict")
                if raw_state_dict is None and isinstance(checkpoint_model, torch.nn.Module):
                    raw_state_dict = checkpoint_model.state_dict()
            if raw_state_dict is None:
                raise ValueError("Checkpoint is missing a state_dict for weights loading.")

            stripped_state_dict = OrderedDict(
                (key.replace("model.", "", 1), value) for key, value in raw_state_dict.items()
            )
            model = instantiate(config.model)
            model.load_state_dict(stripped_state_dict, strict=False)
    else:
        model = instantiate(config.model)

    if use_multi_domain_finetune:
        model = prepare_multi_domain_finetune(config, datamodule, model, logger=log)

    if runtime_wrapper_cfg is not None:
        model = apply_wrappers(model, runtime_wrapper_cfg)

    if not getattr(model, "_initialized", False):
        model.initialize_modules(datamodule)
        log.debug("Initialized model modules from datamodule before task setup.")

    target_dtype = None
    if hasattr(datamodule, "default_dtype"):
        target_dtype = datamodule.default_dtype
    elif hasattr(datamodule, "domain_modules") and datamodule.domain_modules:
        first_dm = next(iter(datamodule.domain_modules.values()))
        target_dtype = getattr(first_dm, "default_dtype", None)
    if target_dtype is not None:
        model = model.to(dtype=target_dtype)
        log.debug("Casting model dtype to data dtype %s", target_dtype)

    if config.compile:
        log.debug("Compiling model with torch.compile")
        model = torch.compile(model)

    log.debug(f"Instantiating task <{config.task._target_}>")
    task: LitNNP = instantiate(config.task, model=model)
    if checkpoint_outputs is not None:
        if not isinstance(checkpoint_outputs, torch.nn.ModuleList):
            checkpoint_outputs = torch.nn.ModuleList(checkpoint_outputs)
        task.outputs = checkpoint_outputs

    log.debug(f"Instantiating model {type(model)} with GNN representation {type(model.representation)}")
    task.save_configuration(config)

    log.debug(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = instantiate(config.trainer)
    if isinstance(trainer.logger, WandbLogger):
        os.makedirs(trainer.logger.save_dir + "/wandb", exist_ok=True)

    trainer.fit(model=task, datamodule=datamodule, ckpt_path=resume_ckpt)

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
