import logging
import os
import socket

from omegaconf import DictConfig, OmegaConf

from .common import configure_cli_logger, ensure_resolvers, log, log_logo, prepare_cli_environment, prepare_run_path


def run_select_config(config: DictConfig):
    prepare_cli_environment()
    ensure_resolvers()
    from hydra.utils import instantiate
    from pytorch_lightning import seed_everything

    from ..config_utils import read_user_config
    from ..select import GeneralActiveLearning
    from ..utils import load_models

    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="select")
    prepare_run_path(config.run_path)

    configure_cli_logger(
        log,
        os.path.join(config.run_path, "selection.log"),
        logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"),
        stream=True,
    )
    log_logo(log)
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    log.debug("Running on host: " + str(socket.gethostname()))
    if "seed" in config:
        log.debug(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.debug("Seed randomly...")

    log.debug("Using model from <{}>".format(config.model_path))
    models = load_models(config.model_path, config.device, load_compiled=False)
    cutoff = models[0].representation.cutoff

    transforms = instantiate(config.transforms) if config.transforms else []
    pool_source = None
    if config.data_url is not None:
        if isinstance(config.data_url, str):
            pool_source = config.data_url
        else:
            data_url = dict(config.data_url)
            if "url" not in data_url:
                raise RuntimeError("data_url must include a 'url' field.")
            pool_source = data_url["url"]
            if len(data_url) > 1:
                log.warning("data_url options are ignored; use pool_set with a URL string instead.")
    elif config.pool_set:
        pool_source = config.pool_set
    if pool_source is None:
        raise RuntimeError("pool_set or data_url is required for selection.")

    select_batch_size = OmegaConf.select(config, "select_batch_size") or OmegaConf.select(config, "batch_size") or 100
    data_batch_size = OmegaConf.select(config, "data_batch_size") or select_batch_size
    save_features = None
    if config.save_features:
        save_features = config.save_features if isinstance(config.save_features, str) else os.path.join(config.run_path, "features.h5")
    save_selected_features = None
    if getattr(config, "save_selected_features", None):
        save_selected_features = (
            config.save_selected_features
            if isinstance(config.save_selected_features, str)
            else os.path.join(config.run_path, "selected_features.h5")
        )
    save_images = None
    if config.save_images:
        save_images = config.save_images if isinstance(config.save_images, str) else os.path.join(config.run_path, "selected.traj")

    feature_specs = OmegaConf.to_container(
        OmegaConf.select(config, "feature_specs", default=[]),
        resolve=True,
    )
    if not feature_specs:
        kernel = OmegaConf.select(config, "kernel", default="full-g")
        n_random_features = OmegaConf.select(config, "n_random_features", default=None)
        legacy_presets = {
            "full-g": "fg-sketch",
            "full-gradient": "fg-sketch",
            "ll-g": "llg-sketch",
            "ll-gradient": "llg-sketch",
            "gnn": "gnn-sketch",
            "local-full-g": "local_fg-sketch",
            "local_full-g": "local_fg-sketch",
            "local-full-gradient": "local_fg-sketch",
            "local_full-gradient": "local_fg-sketch",
        }
        spec = {"preset": legacy_presets.get(str(kernel), str(kernel))}
        if n_random_features:
            spec["num_features"] = int(n_random_features)
        feature_specs = [spec]

    al = GeneralActiveLearning(
        models=models,
        selection=config.method,
        feature_specs=feature_specs,
        selection_feature=OmegaConf.select(config, "selection_feature"),
        target_layer=OmegaConf.select(config, "target_layer", default="readout"),
        num_layers=OmegaConf.select(config, "num_layers"),
        invariants_only=OmegaConf.select(config, "invariants_only", default=True),
        batch_size=data_batch_size,
        device=config.device,
        dataset_cutoff=cutoff,
        transforms=transforms,
        save_features=save_features,
        target_domain=OmegaConf.select(config, "target_domain"),
        selection_kwargs=OmegaConf.to_container(
            OmegaConf.select(config, "selection_kwargs", default={}),
            resolve=True,
        ),
    )
    save_json = os.path.join(config.run_path, "selected.json")
    indices = al.select(
        pool_set=pool_source,
        train_set=config.train_set,
        select_batch_size=select_batch_size,
        save_json=save_json,
        save_images=save_images,
        save_selected_features=save_selected_features,
        normalize_features=OmegaConf.select(config, "export_normalized_features", default=True),
        compute_features_only=bool(
            OmegaConf.select(config, "compute_features_only", default=False)
        ),
    )

    log.debug(
        "Active learning selection completed! Check %s for %d selected structures!",
        os.path.abspath(save_json),
        len(indices),
    )


select = run_select_config
