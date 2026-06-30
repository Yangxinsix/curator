import torch
import json
from ._torch_compat import ensure_torch_safe_globals

ensure_torch_safe_globals()

from e3nn.util.jit import script
from omegaconf import open_dict, OmegaConf, DictConfig, ListConfig
from hydra import compose, initialize, initialize_config_dir
import hydra
from hydra.utils import instantiate, get_class
import inspect
from collections import abc
import logging
import re
from ase import units
from pathlib import Path, PosixPath
from typing import Any, List, Optional, Tuple, Union, Dict, Literal
import numpy as np

from curator.data import properties
def write_json(path: Union[str, Path], payload: Any, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=indent)
    return path


def save_npz(path: Union[str, Path], **payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path

def register_resolvers():
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y, replace=True)
    OmegaConf.register_new_resolver("multiply_fs", lambda x: x * units.fs, replace=True)
    OmegaConf.register_new_resolver("divide_by_fs", lambda x: x / units.fs, replace=True)

def split_list(lst, chunk_or_num, by_chunk_size=False):
    if by_chunk_size:
        num_chunks, remainder = divmod(len(lst), chunk_or_num)
    else:
        chunk_or_num, remainder = divmod(len(lst), chunk_or_num)
    if by_chunk_size:
        return [
            lst[i * chunk_or_num + min(i, remainder):(i + 1) * chunk_or_num + min(i + 1, remainder)]
            for i in range(num_chunks)
        ]
    else:
        return [
            lst[i * (chunk_or_num + (1 if i < remainder else 0)):(i + 1) * (chunk_or_num + (1 if i < remainder else 0))]
            for i in range(chunk_or_num)
        ]

def _copy_config(config_like: Optional[Union[DictConfig, dict]]) -> Optional[DictConfig]:
    """Return a mutable DictConfig copy for easier manipulation."""

    if config_like is None:
        return None
    if isinstance(config_like, DictConfig):
        return OmegaConf.create(OmegaConf.to_container(config_like, resolve=False))
    if isinstance(config_like, dict):
        return OmegaConf.create(config_like)
    return None


def _listify_config_field(config: Optional[DictConfig], field: str) -> None:
    if config is None or field not in config:
        return
    value = config[field]
    if isinstance(value, DictConfig):
        config[field] = [value[k] for k in value.keys()]
    elif isinstance(value, dict):
        config[field] = list(value.values())
    elif isinstance(value, ListConfig):
        config[field] = list(value)


def load_trained_model(
    model_file: Union[str, Path],
    device = None,
    load_compiled: bool = True,
    load_weights_only: bool = False,
    cfg: Optional[DictConfig] = None,
) -> torch.nn.Module:
    """Load a trained model or checkpoint and return a torch.nn.Module."""
    from curator.layer.wrappers import get_config_wrapper_config, get_model_wrapper_config
    from curator.model.checkpoint_upgrade import _register_legacy_outputspec, _upgrade_legacy_checkpoint_model
    from curator.model.conversion import convert_model_wrapper

    model_file = Path(model_file)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_container(config_like):
        if isinstance(config_like, DictConfig):
            return OmegaConf.to_container(config_like, resolve=False)
        return config_like

    def _compose_model_config(model_config, *, data_config=None, task_config=None):
        bundle = OmegaConf.create({"model": _to_container(model_config)})
        if data_config is not None:
            bundle.data = _to_container(data_config)
        if task_config is not None:
            bundle.task = _to_container(task_config)
        return bundle.model

    def _torch_load(model_file: Path, device, load_weights_only: bool):
        try:
            return torch.load(model_file, map_location=torch.device(device), weights_only=load_weights_only)
        except TypeError:
            return torch.load(model_file, map_location=torch.device(device))

    # TorchScript
    if model_file.suffix == '.pt' and load_compiled:
        try:
            model = torch.jit.load(model_file, map_location=torch.device(device))
        except RuntimeError as exc:
            if "cuda" in str(exc).lower() and str(device).startswith("cuda"):
                device = torch.device("cpu")
                model = torch.jit.load(model_file, map_location=device)
            else:
                raise
        try:
            model.to(device)
        except Exception:
            pass
        return model

    _register_legacy_outputspec()
    try:
        obj = _torch_load(model_file, device, load_weights_only)
    except RuntimeError as exc:
        if "cuda" in str(exc).lower() and str(device).startswith("cuda"):
            device = torch.device("cpu")
            obj = _torch_load(model_file, device, load_weights_only)
        else:
            raise

    cfg_copy = _copy_config(cfg)
    if cfg_copy is not None:
        normalize_config_sequences(cfg_copy)

    if isinstance(obj, torch.nn.Module):
        obj = _upgrade_legacy_checkpoint_model(obj)
        obj.to(device)
        return obj

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint format at {model_file}.")

    stored_model = obj.get("model")
    if isinstance(stored_model, torch.nn.Module):
        stored_model = _upgrade_legacy_checkpoint_model(stored_model)
        obj["model"] = stored_model
        stored_wrapper = get_model_wrapper_config(stored_model)
    else:
        stored_wrapper = get_config_wrapper_config(obj.get("wrapper_config"))

    if not load_weights_only and isinstance(stored_model, torch.nn.Module):
        stored_model.to(device)
        return stored_model

    model_cfg = cfg_copy.model if cfg_copy is not None else obj.get("model_params") or obj.get("model_cfg")
    model_cfg = _copy_config(model_cfg)
    if model_cfg is None:
        raise ValueError("Checkpoint does not contain model parameters to instantiate.")
    model_cfg = _compose_model_config(
        model_cfg,
        data_config=(cfg_copy.data if cfg_copy is not None else obj.get("data_params")),
        task_config=(cfg_copy.task if cfg_copy is not None and "task" in cfg_copy else None),
    )
    _listify_config_field(model_cfg, "input_modules")
    _listify_config_field(model_cfg, "output_modules")
    model = instantiate(model_cfg, _convert_="all")

    if stored_wrapper is not None:
        model = convert_model_wrapper(model, stored_wrapper)

    data_cfg = cfg_copy.data if cfg_copy is not None else obj.get('data_params') if isinstance(obj, dict) else None
    data_cfg = _copy_config(data_cfg)
    if data_cfg is not None:
        datamodule = instantiate(data_cfg, _convert_="all")
        if hasattr(datamodule, 'setup'):
            datamodule.setup()
        if hasattr(model, 'initialize_modules'):
            model.initialize_modules(datamodule)

    state_dict = obj.get("state_dict")
    if state_dict is None and isinstance(stored_model, torch.nn.Module):
        state_dict = stored_model.state_dict()
    if state_dict is None:
        raise ValueError("Checkpoint is missing a state_dict.")
    model.load_state_dict(
        {name.replace("model.", "", 1): value for name, value in state_dict.items()},
        strict=False,
    )
    model.to(device)
    return model


def load_model(
    model_file: Union[str, Path],
    device = None,
    load_compiled: bool = True,
    load_weights_only: bool = False,
    cfg: Optional[DictConfig] = None,
) -> torch.nn.Module:
    if isinstance(model_file, str):
        from curator.model.adapter import is_external_model_spec, load_external_model

        if is_external_model_spec(model_file):
            return load_external_model(model_file, device=device)
    return load_trained_model(
        model_file,
        device=device,
        load_compiled=load_compiled,
        load_weights_only=load_weights_only,
        cfg=cfg,
    )

def load_models(
    model_like: Union[str, Path, torch.nn.Module, List[Union[str, Path, torch.nn.Module]]],
    device = None,
    load_compiled: bool = True,
    load_weights_only: bool = False,
    cfg: Optional[DictConfig] = None,
) -> List[torch.nn.Module]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize OmegaConf list/tuple containers into plain Python lists for type checks.
    if isinstance(model_like, ListConfig):
        model_like = list(model_like)

    # for single model passed with str
    if not isinstance(model_like, (list, tuple)):
        model_like = [model_like]
    
    # for list of models passed with nn.Module
    if all(isinstance(m, torch.nn.Module) for m in model_like):
        models = list(model_like)
        for m in models:
            try:
                m.to(device)
            except Exception:
                pass
        return models
    
    # paths or run dirs
    models: List[torch.nn.Module] = []
    for m in model_like:
        if isinstance(m, (str, Path)):
            if isinstance(m, str):
                from curator.model.adapter import is_external_model_spec, load_external_model

                if is_external_model_spec(m):
                    models.append(load_external_model(m, device=device))
                    continue
            p = Path(m)
            if p.is_file():
                models.append(
                    load_model(
                        p,
                        device,
                        load_compiled,
                        load_weights_only=load_weights_only,
                        cfg=cfg,
                    )
                )
                continue
            best_info = find_best_model(p)
            if best_info is None:
                raise FileNotFoundError(
                    f"Could not find a model file in '{p}'. Expected a file path or a run directory."
                )
            best, _ = best_info
            models.append(
                load_model(
                    best,
                    device,
                    load_compiled,
                    load_weights_only=load_weights_only,
                    cfg=cfg,
                )
            )
        else:
            raise TypeError("List elements must be all nn.Module or all str/Path.")
    
    return models

def ensure_dict(value: Any, prefix: str = "item"):
    """Convert list-style Hydra nodes to dictionaries."""

    if isinstance(value, DictConfig):
        return value
    if isinstance(value, (ListConfig, list)):
        items = {}
        for idx, entry in enumerate(value):
            key = _infer_sequence_key(entry, idx, prefix)
            if key in items:
                key = f"{key}_{idx}"
            items[key] = entry
        return OmegaConf.create(items)
    return value

def _camel_to_snake(name: str) -> str:
    import re
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

def _infer_sequence_key(entry: Any, idx: int, prefix: str) -> str:
    if isinstance(entry, (DictConfig, dict)):
        name = entry.get("name")
        if isinstance(name, str) and name:
            return name
        target = entry.get("_target_")
        if isinstance(target, str) and target:
            return _camel_to_snake(target.split(".")[-1])
    return f"{prefix}_{idx}"

def find_best_model(run_path: Union[str, Path]) -> Tuple[Path, Optional[float]]:
    """Return an existing file path directly or the best ckpt path under a run directory."""

    run_path = Path(run_path)
    if run_path.is_file():
        return run_path, None

    cands = list(run_path.glob("best_model_*.ckpt"))
    if cands:
        best_p, best_v = None, float('inf')
        for p in cands:
            try:
                v = float(str(p).split('=')[-1].rstrip('.ckpt'))
            except Exception:
                continue
            if v < best_v:
                best_v, best_p = v, p
        if best_p is not None:
            return best_p, best_v
    
    # return newest .ckpt if no best_model_*.ckpt is there
    all_ckpts = sorted(run_path.glob("*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if all_ckpts:
        return all_ckpts[0], None

class CustomFormatter(logging.Formatter):
    format = "%(asctime)s: %(message)s"
    time_format = "%Y-%m-%d %H:%M:%S"
     
    FORMATS = {
        logging.DEBUG: format,
        logging.INFO: "%(message)s",
        logging.WARNING: format,
        logging.ERROR: format,
        logging.CRITICAL: format
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, self.time_format)
        return formatter.format(record)

_LOGO_LOGGED = False


def log_logo(logger: Optional[logging.Logger] = None) -> None:
    global _LOGO_LOGGED
    if _LOGO_LOGGED:
        return
    _LOGO_LOGGED = True
    log = logger or logging.getLogger("curator")
    logo = [
        """
            █████████  █████  █████ ███████████     █████████   ███████████    ███████    ███████████  
           ███░░░░░███░░███  ░░███ ░░███░░░░░███   ███░░░░░███ ░█░░░███░░░█  ███░░░░░███ ░░███░░░░░███ 
          ███     ░░░  ░███   ░███  ░███    ░███  ░███    ░███ ░   ░███  ░  ███     ░░███ ░███    ░███ 
         ░███          ░███   ░███  ░██████████   ░███████████     ░███    ░███      ░███ ░██████████  
         ░███          ░███   ░███  ░███░░░░░███  ░███░░░░░███     ░███    ░███      ░███ ░███░░░░░███ 
         ░░███     ███ ░███   ░███  ░███    ░███  ░███    ░███     ░███    ░░███     ███  ░███    ░███ 
          ░░█████████  ░░████████   █████   █████ █████   █████    █████    ░░░███████░   █████   █████
           ░░░░░░░░░    ░░░░░░░░   ░░░░░   ░░░░░ ░░░░░   ░░░░░    ░░░░░       ░░░░░░░    ░░░░░   ░░░░░

                           Active learning for machine learning interatomic potentials
        """,
    ]
    display_lines = [line.replace("\\\\", "\\") for line in logo]
    width = max(max(len(line) for line in display_lines), 80)
    for line in display_lines:
        log.info(line.center(width))

# Auxiliary function for parsing config file 
def get_all_pairs(d, keys=()):
    if isinstance(d, abc.Mapping):
         for k in d:
            for rv in get_all_pairs(d[k], keys + (k, )):
                yield rv
    else:
        yield (keys, d)

def _dictify_field(
    container: Optional[DictConfig],
    key: str,
    prefix: str,
    path: str,
    converted: set,
) -> None:
    if container is None or key not in container or container[key] is None:
        return

    new_value = ensure_dict(container[key], prefix)
    if new_value is container[key]:
        return

    if isinstance(container, DictConfig):
        with open_dict(container):
            container[key] = new_value
    else:
        container[key] = new_value

    converted.add(path)


def _dictify_sequence_nodes(config: Optional[DictConfig]) -> set:
    converted = set()
    if config is None:
        return converted

    if "trainer" in config:
        _dictify_field(config.trainer, "callbacks", "callback", "trainer.callbacks", converted)

    if "model" in config:
        _dictify_field(config.model, "input_modules", "input_module", "model.input_modules", converted)
        _dictify_field(config.model, "output_modules", "output_module", "model.output_modules", converted)

    if "task" in config:
        _dictify_field(config.task, "outputs", "output", "task.outputs", converted)

    if "simulator" in config:
        _dictify_field(config.simulator, "callbacks", "callback", "simulator.callbacks", converted)

    return converted

def normalize_config_sequences(config: Optional[DictConfig]) -> None:
    """Convert configurable sequence fields to list form for easier consumption."""
    if config is None:
        return

    if "trainer" in config:
        _listify_config_field(config.trainer, "callbacks")

    if "model" in config:
        _listify_config_field(config.model, "input_modules")
        _listify_config_field(config.model, "output_modules")

    if "task" in config:
        _listify_config_field(config.task, "outputs")

    if "simulator" in config:
        _listify_config_field(config.simulator, "callbacks")


def prune_config_targets(config: Optional[DictConfig], logger: Optional[logging.Logger] = None) -> None:
    """
    Remove keys from config nodes that specify a _target_ but include arguments
    not accepted by the target's signature (unless it has **kwargs).
    Helps prevent stale parameters from other defaults (e.g., switching models/engines).
    """
    if config is None:
        return

    log = logger or logging.getLogger("curator")
    special_keys = {"_target_", "_partial_", "_recursive_", "_convert_"}

    def _prune(node: DictConfig, path: str = ""):
        if not isinstance(node, DictConfig):
            return

        target = node.get("_target_")
        if target:
            try:
                obj = get_class(str(target))
            except Exception:
                obj = None

            if obj is not None:
                sig = inspect.signature(obj.__init__ if inspect.isclass(obj) else obj)
                params = sig.parameters
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                    allowed = None
                else:
                    allowed = {name for name in params if name != "self"}

                if allowed is not None:
                    allowed.update(special_keys)
                    unknown = [k for k in node.keys() if k not in allowed]
                    if unknown:
                        with open_dict(node):
                            for k in unknown:
                                del node[k]
                        log.debug(f"Pruned keys {unknown} from config node '{path or '<root>'}' for target {target}")

        for k, v in node.items():
            if isinstance(v, DictConfig):
                _prune(v, f"{path}.{k}" if path else k)

    _prune(config)


def update_config_from_datamodule(
    config: DictConfig,
    datamodule: Any,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Update config based on datamodule contents.

    This function intentionally stays on the generic config-update path:
    infer species and default heads, but do not mutate the model structure here.
    Domain-aware model preparation happens later in the training flow.
    """
    def _is_auto(value) -> bool:
        return value is None or (isinstance(value, str) and value.lower() == "auto")

    def _ensure_list(value, default=None) -> List:
        if value is None:
            return list(default or [])
        if isinstance(value, str):
            return [value]
        return list(value)

    def _unique_preserve_order(values: List) -> List:
        unique: List = []
        for value in values:
            if value not in unique:
                unique.append(value)
        return unique

    def _update_heads_cfg(cfg: DictConfig, keypath: str, heads: List) -> None:
        heads_cfg = OmegaConf.select(cfg, keypath)
        if isinstance(heads_cfg, DictConfig) and "_target_" in heads_cfg:
            OmegaConf.update(cfg, f"{keypath}.heads", heads, force_add=True)
        else:
            OmegaConf.update(cfg, keypath, heads, force_add=True)

    def _has_configured_representation_readout(cfg: DictConfig) -> bool:
        readout_cfg = OmegaConf.select(cfg, "model.representation.readout")
        return isinstance(readout_cfg, DictConfig) and "_target_" in readout_cfg

    def _readout_has_explicit_outputs(cfg: DictConfig) -> bool:
        if not _has_configured_representation_readout(cfg):
            return False

        readout_heads = OmegaConf.select(cfg, "model.representation.readout.heads")
        if isinstance(readout_heads, (ListConfig, list)) and len(readout_heads) > 0:
            return True
        if isinstance(readout_heads, str) and readout_heads.lower() != "auto":
            return True

        readout_heads_by_domain = OmegaConf.select(cfg, "model.representation.readout.heads_by_domain")
        if isinstance(readout_heads_by_domain, (DictConfig, dict)) and len(readout_heads_by_domain) > 0:
            return True

        return False

    def _update_representation_heads(cfg: DictConfig, heads: List) -> None:
        if _has_configured_representation_readout(cfg):
            OmegaConf.update(cfg, "model.representation.readout.heads", heads, force_add=True)
        else:
            OmegaConf.update(cfg, "model.representation.heads", heads, force_add=True)

    def _update_representation_heads_by_domain(
        cfg: DictConfig,
        heads_by_domain: Dict[str, List],
    ) -> None:
        if _has_configured_representation_readout(cfg):
            OmegaConf.update(cfg, "model.representation.readout.heads_by_domain", heads_by_domain, force_add=True)
            OmegaConf.update(cfg, "model.representation.readout.domains", list(heads_by_domain.keys()), force_add=True)
            return

        merged_heads: List = []
        for domain_heads in heads_by_domain.values():
            for head in domain_heads:
                if head not in merged_heads:
                    merged_heads.append(head)
        OmegaConf.update(cfg, "model.representation.heads", merged_heads or ["energy"], force_add=True)
        if logger is not None and len(heads_by_domain) > 1:
            logger.warning(
                "Per-domain heads requested but model.representation.readout is not configurable in the "
                "config snapshot; storing merged representation heads %s in config while domain-specific "
                "readout heads will be applied when the model is promoted to multi-domain.",
                merged_heads or ["energy"],
            )

    def _update_rescale_heads(cfg: DictConfig, heads: List) -> None:
        output_modules = OmegaConf.select(cfg, "model.output_modules")
        if isinstance(output_modules, (DictConfig, dict)) and "global_rescale_shift" in output_modules:
            _update_heads_cfg(cfg, "model.output_modules.global_rescale_shift.heads", heads)
            return
        if isinstance(output_modules, (ListConfig, list)):
            for idx, item in enumerate(output_modules):
                if not isinstance(item, (DictConfig, dict)):
                    continue
                target = item.get("_target_")
                if target and "RescaleShift" in str(target):
                    _update_heads_cfg(cfg, f"model.output_modules.{idx}.heads", heads)
                    return

    # Update config.data.species from datamodule or contexts.
    if hasattr(datamodule, "species") and _is_auto(getattr(datamodule, "species", None)):
        inferred = datamodule._get_species()
        config.data.species = inferred
    elif not hasattr(datamodule, "species") and hasattr(datamodule, "build_contexts"):
        try:
            ctxs = datamodule.build_contexts([])
            if "global" in ctxs and ctxs["global"].species:
                inferred = ctxs["global"].species
                config.data.species = inferred
        except Exception:
            pass

    # Update heads for the generic model path.
    data_heads = OmegaConf.select(config, "data.heads")
    if _is_auto(data_heads) or data_heads is None:
        data_heads = ["energy"]
    data_heads = _ensure_list(data_heads, default=["energy"])
    shared_rescale_shift_heads = OmegaConf.select(config, "data.rescale_shift_heads")
    shared_rescale_shift_heads = _ensure_list(shared_rescale_shift_heads, default=[])

    readout_heads = OmegaConf.select(config, "model.representation.readout.heads")
    should_update_heads = not _readout_has_explicit_outputs(config) and (
        _is_auto(readout_heads)
        or (
            isinstance(readout_heads, (ListConfig, list))
            and list(readout_heads) == ["energy"]
        )
    )
    domain_modules = getattr(datamodule, "domain_modules", None)
    if domain_modules:
        heads_by_domain: Dict[str, List] = {}
        rescale_heads = []
        domain_to_id = getattr(datamodule, "domain_to_id", {}) or {}
        for domain_name, domain_dm in domain_modules.items():
            domain_key = str(domain_to_id.get(domain_name, domain_name))
            domain_heads = _ensure_list(getattr(domain_dm, "heads", None), default=data_heads)
            heads_by_domain[domain_key] = _unique_preserve_order(domain_heads)

            domain_rescale = [properties.energy]
            for key in _ensure_list(
                getattr(domain_dm, "rescale_shift_heads", None),
                default=shared_rescale_shift_heads,
            ):
                if key not in domain_rescale:
                    domain_rescale.append(key)

            for key in domain_rescale:
                rescale_heads.append({"key": key, "domains": [domain_key]})

        if should_update_heads:
            merged_heads: List = []
            for domain_heads in heads_by_domain.values():
                for head in domain_heads:
                    if head not in merged_heads:
                        merged_heads.append(head)
            _update_heads_cfg(config, "model.heads", merged_heads or ["energy"])
            _update_representation_heads_by_domain(config, heads_by_domain)

        _update_rescale_heads(
            config,
            rescale_heads or [{"key": "energy", "domains": list(heads_by_domain.keys())}],
        )
    else:
        rescale_heads = [properties.energy]
        for key in shared_rescale_shift_heads:
            if key not in rescale_heads:
                rescale_heads.append(key)

        if should_update_heads:
            _update_heads_cfg(config, "model.heads", data_heads)
            _update_representation_heads(config, data_heads)

        _update_rescale_heads(config, rescale_heads or [properties.energy])

    # If we are not using multi-domain loaders, strip dataloader_idx suffixes.
    if not hasattr(datamodule, "domain_modules"):
        def _strip_idx(value: Optional[str]) -> Optional[str]:
            if isinstance(value, str) and value.endswith("/dataloader_idx_0"):
                return value.replace("/dataloader_idx_0", "")
            return value

        sched_monitor = OmegaConf.select(config, "task.scheduler_monitor")
        sched_monitor = _strip_idx(sched_monitor)
        if sched_monitor is not None:
            OmegaConf.update(config, "task.scheduler_monitor", sched_monitor, force_add=True)

        callbacks = OmegaConf.select(config, "trainer.callbacks")
        if isinstance(callbacks, (ListConfig, list)):
            for idx, cb in enumerate(callbacks):
                if not isinstance(cb, (DictConfig, dict)):
                    continue
                monitor = cb.get("monitor", None)
                monitor = _strip_idx(monitor)
                if monitor is not None:
                    OmegaConf.update(config, f"trainer.callbacks.{idx}.monitor", monitor, force_add=True)

    if hasattr(datamodule, "log_summary"):
        summary = datamodule.log_summary()
        if summary:
            (logger or logging.getLogger(__name__)).info("%s", summary)

# Ugly workaround for specifying config files outside of the package
def read_user_config(
    cfg: Union[DictConfig, PosixPath, str, None]=None,
    config_path="configs",
    config_name="train.yaml",
    overrides: Optional[Union[str, List[str]]] = None,
):
    # load cfg
    if isinstance(cfg, DictConfig):
        user_cfg = cfg.copy()
    elif isinstance(cfg, (PosixPath, str)):
        user_cfg = OmegaConf.load(cfg)
    else:
        user_cfg = OmegaConf.create()

    converted_fields = set()
    if isinstance(user_cfg, DictConfig):
        converted_fields = _dictify_sequence_nodes(user_cfg)

    config_path_obj = Path(config_path)
    use_config_dir = config_path_obj.is_absolute()
    if not use_config_dir:
        pkg_base = Path(__file__).resolve().parent
        candidate = (pkg_base / config_path_obj).resolve()
        if candidate.exists():
            config_path_obj = candidate
            use_config_dir = True
        else:
            candidate = (Path.cwd() / config_path_obj).resolve()
            if candidate.exists():
                config_path_obj = candidate
                use_config_dir = True
    config_path = str(config_path_obj)

    override_list = []
    if "defaults" in user_cfg:
        default_list = user_cfg.pop("defaults")
        for d in default_list:
            if isinstance(d, (dict, DictConfig)):
                for k, v in d.items():
                    override_list.append(f"{k}={v}")
    
    for path in sorted(converted_fields):
        override_list.append(f"~{path}")

    deferred_updates = []
    for k, v in get_all_pairs(user_cfg):
        key = ".".join(k)
        if isinstance(v, (DictConfig, ListConfig, dict, list)):
            deferred_updates.append((key, v))
            continue
        # process value
        value = str(escape_all(v)).replace("'", "")
        if value == 'None':
            value = 'null'
        override_list.append(f'++{key}={value}')
    
    # command line overrides
    try:
        cli_overrides = hydra.core.hydra_config.HydraConfig.get().overrides.task
    except:
        cli_overrides = []
    finally:
        override_list.extend(cli_overrides)

    if overrides is not None:
        if isinstance(overrides, str):
            overrides = [overrides]
        override_list.extend(overrides)

    # reload hyperparameters         
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    if use_config_dir:
        context = initialize_config_dir(version_base=None, config_dir=config_path)
    else:
        context = initialize(version_base=None, config_path=config_path)
    with context:
        composed_cfg = compose(config_name=config_name, overrides=override_list)

    # Allow write access to unknown fields
    OmegaConf.set_struct(composed_cfg, False)

    for key, value in deferred_updates:
        OmegaConf.update(composed_cfg, key, value, merge=True)

    normalize_config_sequences(composed_cfg)
    prune_config_targets(composed_cfg)
        
    return composed_cfg

def escape_special_characters(value: str) -> str:
    special_characters = r"\()[]{}:=,&"
    for char in special_characters:
        if char in value:
            value = f'"{value}"'
            break
    return value

def escape_all(data):
    if isinstance(data, str):
        return escape_special_characters(data)
    elif isinstance(data, (dict, DictConfig)):
        return {k: escape_all(v) for k, v in data.items()}
    elif isinstance(data, (list, ListConfig)):
        return [escape_all(item) for item in data]
    else:
        return data

def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Sums all values from the `src` tensor into `out` at the indices specified in the `index` tensor
    along the dimension `dim`. If `out` is not provided, it will be automatically created with the correct size.

    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices of elements to scatter.
            Must have the same size as `src` at dimension `dim` or be broadcastable to that size.
        dim (int): The axis along which to index. Negative values wrap around.
        out (torch.Tensor, optional): The destination tensor.

    Returns:
        torch.Tensor: The resulting tensor with the summed values scattered at the specified indices.
    """
    index = _broadcast(index, src, dim)
    if out is None:
        # Determine size of output tensor along dimension `dim`
        output_size = list(src.size())
        output_size[dim] = int(index.max()) + 1  # Size along dim is max index + 1
        out = torch.zeros(output_size, dtype=src.dtype, device=src.device)

    # Perform scatter add
    out.scatter_add_(dim, index, src)

    return out

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the mean of all values from the `src` tensor into `out` at the indices specified in the `index` tensor
    along the dimension `dim`. If `out` is not provided, it will be automatically created to have the correct size.

    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices of elements to scatter. Must have the same size as `src` at dimension `dim`.
        dim (int): The axis along which to index. Negative values wrap around.
        out (torch.Tensor, optional): The destination tensor.

    Returns:
        torch.Tensor: The resulting tensor with the mean values scattered at the specified indices.
    """
    index = _broadcast(index, src, dim)

    if out is None:
        # Determine size of output tensor along dimension `dim`
        output_size = list(src.size())
        output_size[dim] = int(index.max()) + 1  # Size along dim is max index + 1
        out = torch.zeros(output_size, dtype=src.dtype, device=src.device)
        out_count = torch.zeros_like(out)
    else:
        out_count = torch.zeros_like(out)

    # Compute sum of values
    out.scatter_add_(dim, index, src)

    # Count number of occurrences at each index
    ones = torch.ones_like(src, dtype=src.dtype)
    out_count.scatter_add_(dim, index, ones)

    # Avoid division by zero
    zero_mask = out_count == 0
    out_count[zero_mask] = 1

    # Compute mean
    out = out / out_count

    return out

def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the maximum of all values from the `src` tensor into `out` at the indices specified in the `index` tensor
    along the dimension `dim`.

    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices of elements to scatter.
            Must have the same size as `src` at dimension `dim` or be broadcastable to that size.
        dim (int): The axis along which to index. Negative values wrap around.
        out (torch.Tensor, optional): The destination tensor. If None, a new tensor is created.

    Returns:
        torch.Tensor: The resulting tensor with the maximum values scattered at the specified indices.
    """
    index = _broadcast(index, src, dim)

    # Determine size of output tensor along dimension `dim`
    output_size = list(src.size())
    output_size[dim] = int(index.max()) + 1  # Size along dim is max index + 1

    # Initialize out tensor with minimum possible values
    if out is None:
        out = torch.full(output_size, torch.finfo(src.dtype).min, dtype=src.dtype, device=src.device)
    else:
        out.fill_(torch.finfo(src.dtype).min)

    # Compute maximum values
    out.scatter_(dim, index, torch.max(out.gather(dim, index), src))

    return out

def scatter_min(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the minimum of all values from the `src` tensor into `out` at the indices specified in the `index` tensor
    along the dimension `dim`.

    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices of elements to scatter.
            Must have the same size as `src` at dimension `dim` or be broadcastable to that size.
        dim (int): The axis along which to index. Negative values wrap around.
        out (torch.Tensor, optional): The destination tensor. If None, a new tensor is created.

    Returns:
        torch.Tensor: The resulting tensor with the minimum values scattered at the specified indices.
    """
    index = _broadcast(index, src, dim)

    # Determine size of output tensor along dimension `dim`
    output_size = list(src.size())
    output_size[dim] = int(index.max()) + 1

    # Initialize out tensor with maximum possible values
    if out is None:
        out = torch.full(output_size, torch.finfo(src.dtype).max, dtype=src.dtype, device=src.device)
    else:
        out.fill_(torch.finfo(src.dtype).max)

    # Compute minimum values
    out.scatter_(dim, index, torch.min(out.gather(dim, index), src))

    return out


def camel_to_snake(name: str) -> str:
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return s1.lower()


def ensure_list(value: Any):
    if isinstance(value, DictConfig):
        return [value[k] for k in value.keys()]
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, ListConfig):
        return list(value)
    return value


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, best_loss):
        if val_loss - best_loss > self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def deploy_model(model, file_path: str):
    compiled_model = script(model)
    compiled_model.save(file_path)


def scatter_reduce(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor = None,
    reduce: Literal["sum", "mean", "max", "min"] = "sum",
    include_self: bool = False,
) -> torch.Tensor:
    if reduce not in ["sum", "mean", "max", "min"]:
        raise ValueError(
            f"Invalid reduce operation '{reduce}'. Supported operations: 'sum', 'mean', 'max', 'min'."
        )

    index = _broadcast(index, src, dim)
    output_size = list(src.size())
    output_size[dim] = int(index.max()) + 1

    if out is None:
        if reduce in ["sum", "mean"]:
            out = torch.zeros(output_size, dtype=src.dtype, device=src.device)
        elif reduce == "max":
            out = torch.full(
                output_size, torch.finfo(src.dtype).min, dtype=src.dtype, device=src.device
            )
        else:
            out = torch.full(
                output_size, torch.finfo(src.dtype).max, dtype=src.dtype, device=src.device
            )
    elif not include_self:
        if reduce in ["sum", "mean"]:
            out.zero_()
        elif reduce == "max":
            out.fill_(torch.finfo(src.dtype).min)
        else:
            out.fill_(torch.finfo(src.dtype).max)

    if reduce == "sum":
        out.scatter_add_(dim, index, src)
    elif reduce == "mean":
        out.scatter_add_(dim, index, src)
        count = torch.zeros_like(out)
        ones = torch.ones_like(src, dtype=src.dtype)
        count.scatter_add_(dim, index, ones)
        zero_mask = count == 0
        count[zero_mask] = 1
        out = out / count
    elif reduce == "max":
        out.scatter_(dim, index, torch.max(out.gather(dim, index), src))
    else:
        out.scatter_(dim, index, torch.min(out.gather(dim, index), src))

    return out


def is_upper_triangular(cell):
    return np.allclose(np.tril(cell, -1), 0)


def upper_triangular_cell(atoms, verbose=False):
    if not is_upper_triangular(atoms.get_cell()):
        a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
        angles = np.radians((alpha, beta, gamma))
        sin_a, sin_b, sin_g = np.sin(angles)
        cos_a, cos_b, cos_g = np.cos(angles)
        cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
        cos_p = np.clip(cos_p, -1, 1)
        sin_p = np.sqrt(1 - cos_p**2)
        new_basis = [
            (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
            (0, b * sin_a, b * cos_a),
            (0, 0, c),
        ]
        atoms.set_cell(new_basis, scale_atoms=True)
        if verbose:
            print("Transformed to upper triangular unit cell.", flush=True)
    return atoms


def get_representation_config(model):
    rep = model.representation
    export_fn = getattr(rep, "export_init_kwargs", None)
    if callable(export_fn):
        try:
            exported = export_fn()
        except NotImplementedError:
            exported = None
        if isinstance(exported, abc.Mapping):
            return dict(exported)
    raise TypeError(
        f"{rep.__class__.__name__} does not expose export_init_kwargs() for wrapper rebuilds."
    )


def get_kmax_pairs(
    max_L: int, correlation: int, num_layers: int
) -> List[Tuple[int, int]]:
    if correlation == 2:
        raise NotImplementedError("Correlation 2 not supported yet")
    if correlation == 3:
        kmax_pairs = [[i, max_L] for i in range(num_layers - 1)]
        kmax_pairs = kmax_pairs + [[num_layers - 1, 0]]
        return kmax_pairs
    raise NotImplementedError(f"Correlation {correlation} not supported")


def transfer_symmetric_contractions(
    source_dict: Dict[str, torch.Tensor],
    target_dict: Dict[str, torch.Tensor],
    max_L: int,
    correlation: int,
    num_layers: int,
):
    kmax_pairs = get_kmax_pairs(max_L, correlation, num_layers)

    for i, kmax in kmax_pairs:
        wm = torch.concatenate(
            [
                source_dict[
                    f"products.{i}.symmetric_contractions.contractions.{k}.weights{j}"
                ]
                for k in range(kmax + 1)
                for j in ["_max", ".0", ".1"]
            ],
            dim=1,
        )
        target_dict[f"products.{i}.symmetric_contractions.sc.weight"] = wm


def _squeeze_if_compatible(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if src.shape != target_shape and src.dim() == len(target_shape) + 1 and src.shape[0] == 1:
        return src.squeeze(0)
    return src


def _expand_if_compatible(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if src.shape != target_shape and src.dim() + 1 == len(target_shape) and target_shape[0] == 1:
        return src.unsqueeze(0)
    return src


def transfer_symmetric_contractions_back(
    source_dict: Dict[str, torch.Tensor],
    target_dict: Dict[str, torch.Tensor],
    max_L: int,
    correlation: int,
    num_layers: int,
):
    kmax_pairs = get_kmax_pairs(max_L, correlation, num_layers)

    for i, kmax in kmax_pairs:
        key = f"products.{i}.symmetric_contractions.sc.weight"
        if key not in source_dict:
            continue
        weight = source_dict[key]
        offset = 0
        for k in range(kmax + 1):
            for suffix in ["_max", ".0", ".1"]:
                tgt_key = f"products.{i}.symmetric_contractions.contractions.{k}.weights{suffix}"
                if tgt_key not in target_dict:
                    continue
                width = target_dict[tgt_key].shape[1]
                target_dict[tgt_key] = weight[:, offset : offset + width]
                offset += width


def load_e3nn_weights(source_model, target_model):
    source_dict = source_model.representation.state_dict()
    target_dict = target_model.representation.state_dict()
    target_shapes = {k: v.shape for k, v in target_dict.items()}

    num_layers = len(source_model.representation.interactions)
    transfer_keys = sorted(
        key
        for key in set(source_dict.keys()) & set(target_dict.keys())
        if key.startswith(("embeddings.", "interactions.", "products.", "readout."))
        and "symmetric_contraction" not in key
    )
    for key in transfer_keys:
        target_shape = target_shapes.get(key)
        if target_shape is None:
            continue
        if key in source_dict:
            target_dict[key] = _expand_if_compatible(source_dict[key], target_shape)
        else:
            logging.warning("Key %s not found in source model", key)

    use_cueq = any(k.endswith("symmetric_contractions.sc.weight") for k in target_shapes)
    if use_cueq:
        lmax = source_model.representation.lmax
        try:
            correlation = (
                len(source_model.representation.products[0].symmetric_contractions.contractions[0].weights) + 1
            )
        except AttributeError:
            correlation = source_model.representation.products[0].symmetric_contractions.sc.contraction_degree
        transfer_symmetric_contractions(source_dict, target_dict, lmax, correlation, num_layers)

    transferred_keys = set(transfer_keys)
    remaining_keys = set(source_dict.keys()) & set(target_dict.keys()) - transferred_keys
    if use_cueq:
        remaining_keys = {k for k in remaining_keys if "symmetric_contraction" not in k}
    for key in remaining_keys:
        src_val = _expand_if_compatible(source_dict[key], target_shapes[key])
        if src_val.shape == target_shapes[key]:
            target_dict[key] = src_val
        else:
            logging.warning(
                "Shape mismatch for key %s: source %s vs target %s",
                key,
                source_dict[key].shape,
                target_shapes[key],
            )

    target_model.representation.load_state_dict(target_dict)


def load_cueq_weights(source_model, target_model):
    source_dict = source_model.representation.state_dict()
    target_dict = target_model.representation.state_dict()
    target_shapes = {k: v.shape for k, v in target_dict.items()}

    num_layers = len(target_model.representation.interactions)
    transfer_keys = sorted(
        key
        for key in set(source_dict.keys()) & set(target_dict.keys())
        if key.startswith(("embeddings.", "interactions.", "products.", "readout."))
        and "symmetric_contraction" not in key
    )
    for key in transfer_keys:
        target_shape = target_shapes.get(key)
        if target_shape is None:
            continue
        if key in source_dict:
            target_dict[key] = _squeeze_if_compatible(source_dict[key], target_shape)
        else:
            logging.warning("Key %s not found in source cueq model", key)

    lmax = getattr(source_model.representation, "lmax", None)
    try:
        correlation = (
            len(source_model.representation.products[0].symmetric_contractions.contractions[0].weights) + 1
        )
    except Exception:
        correlation = source_model.representation.products[0].symmetric_contractions.sc.contraction_degree
    if lmax is not None:
        transfer_symmetric_contractions_back(source_dict, target_dict, lmax, correlation, num_layers)

    transferred_keys = set(transfer_keys)
    remaining_keys = set(source_dict.keys()) & set(target_dict.keys()) - transferred_keys
    remaining_keys = {k for k in remaining_keys if "symmetric_contraction" not in k}
    for key in remaining_keys:
        src_val = _squeeze_if_compatible(source_dict[key], target_shapes[key])
        if src_val.shape == target_shapes[key]:
            target_dict[key] = src_val
    target_model.representation.load_state_dict(target_dict)
