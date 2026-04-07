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


@hydra.main(config_path=CONFIGS_PATH, config_name="train", version_base=None)
def train(config: DictConfig) -> None:
    prepare_cli_environment()
    ensure_resolvers()
    import torch
    from hydra.utils import instantiate
    from pytorch_lightning import seed_everything
    from pytorch_lightning.loggers import WandbLogger

    from ..config_utils import normalize_config_sequences, prune_config_targets, read_user_config
    from ..model import LitNNP, NeuralNetworkPotential
    from ..utils import (
        CustomFormatter,
        find_best_model,
        update_config_from_datamodule,
        update_model_domains,
    )

    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="train")

    normalize_config_sequences(config)
    prune_config_targets(config, logger=log)
    prepare_run_path(config.run_path)
    configure_cli_logger(log, os.path.join(config.run_path, "training.log"), CustomFormatter(), stream=True)
    log_logo(log)
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    log.debug("Running on host: " + str(socket.gethostname()))

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

    log.debug(f"Instantiating datamodule <{config.data._target_}> from dataset {config.data.datapath or config.data.train_path}")
    datamodule = instantiate(config.data)
    datamodule.setup()
    update_config_from_datamodule(config, datamodule, logger=log)

    domain_mode = getattr(config.task, "domain_mode", None)
    new_domains = getattr(config.task, "new_domains", None)
    if domain_mode is None and config.model_path is not None:
        if hasattr(datamodule, "domain_modules") and len(datamodule.domain_modules) > 1:
            domain_mode = "extend"
            config.task.domain_mode = domain_mode
            log.debug("Auto-set domain_mode=extend for multi-domain fine-tune.")
    if domain_mode in ("extend", "replace") and new_domains is None:
        if hasattr(datamodule, "domain_modules") and hasattr(datamodule, "domain_to_id"):
            inferred = []
            for name in datamodule.domain_modules.keys():
                if str(name).lower().startswith("replay"):
                    continue
                dom_id = datamodule.domain_to_id.get(name)
                if dom_id is not None:
                    inferred.append(str(dom_id))
            if inferred:
                new_domains = inferred
                config.task.new_domains = inferred
                log.debug("Inferred new_domains from datapath (excluding replay): %s", inferred)

    resume_ckpt = None
    checkpoint_outputs = None

    if config.model_path is not None:
        config.model_path = find_best_model(config.model_path)[0]
        log.debug(f"Loading trained model from {config.model_path}")
        if config.task.load_entire_model:
            state_dict = torch_load_compat(torch, config.model_path, weights_only=False)
            if isinstance(state_dict, torch.nn.Module):
                model = state_dict
                checkpoint_outputs = getattr(state_dict, "outputs", None)
            else:
                model = state_dict["model"]
                checkpoint_outputs = state_dict.get("outputs")
            if not isinstance(model, NeuralNetworkPotential):
                raise TypeError(f"Expected NeuralNetworkPotential, got {type(model)}")
        else:
            model = instantiate(config.model)
            if config.task.load_weights_only:
                from collections import OrderedDict

                state_dict = torch_load_compat(torch, config.model_path, weights_only=False)
                if isinstance(state_dict, torch.nn.Module):
                    state_dict = {"state_dict": state_dict.state_dict()}
                new_state_dict = OrderedDict((key.replace("model.", ""), value) for key, value in state_dict["state_dict"].items())
                model.load_state_dict(new_state_dict, strict=False)
            else:
                resume_ckpt = config.model_path
                log.debug(
                    "Resuming training from checkpoint; optimizer and scheduler states will be restored from %s",
                    resume_ckpt,
                )
    else:
        model = instantiate(config.model)

    init_from = getattr(config.task, "init_new_domains_from", None)
    if domain_mode in ("extend", "replace") and new_domains is not None:
        init_strategy = "copy" if init_from is not None else "random"
        updated = update_model_domains(
            model,
            new_domains,
            mode=domain_mode,
            template_domain=init_from or "0",
            init_strategy=init_strategy,
            logger=log,
        )
        log.debug("Updated model domains: mode=%s new_domains=%s updated=%s", domain_mode, new_domains, updated)

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
                log.debug(f"Deploy trained model from {model_path}")
            else:
                log.debug(f"Deploy trained model from {model_path} with validation loss of {val_loss:.3f}")
            deploy(model_path, f"{config.run_path}/compiled_model.pt", uncertainty_spec=OmegaConf.select(config, "deploy.uncertainty", default=None))
            log.debug(f"Deploying compiled model at <{config.run_path}/compiled_model.pt>")


@hydra.main(config_path=CONFIGS_PATH, config_name="train", version_base=None)
def tmp_train(config: DictConfig):
    prepare_cli_environment()
    ensure_resolvers()
    import torch
    from hydra.utils import instantiate
    from pytorch_lightning import seed_everything

    from ..train import train as train_loop
    from ..config_utils import normalize_config_sequences, read_user_config
    from ..utils import CustomFormatter

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
        deploy(model_path, f"{config.run_path}/compiled_model.pt", uncertainty_spec=OmegaConf.select(config, "deploy.uncertainty", default=None))
        log.debug(f"Deploying compiled model at <{config.run_path}/compiled_model.pt>")


tmptrain = tmp_train
