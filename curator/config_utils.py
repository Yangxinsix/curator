"""Lightweight config helpers that must not pull in torch at import time."""

import inspect
import logging
from collections import abc
from pathlib import Path, PosixPath
from typing import Any, List, Optional, Union

import hydra
from ase import units
from hydra import compose, initialize, initialize_config_dir
from hydra.utils import get_class
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict


def register_resolvers() -> None:
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y, replace=True)
    OmegaConf.register_new_resolver("multiply_fs", lambda x: x * units.fs, replace=True)
    OmegaConf.register_new_resolver("divide_by_fs", lambda x: x / units.fs, replace=True)


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


def ensure_dict(value: Any, prefix: str = "item"):
    """Convert legacy list-style Hydra nodes to dictionaries."""

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


def get_all_pairs(d, keys=()):
    if isinstance(d, abc.Mapping):
        for k in d:
            for rv in get_all_pairs(d[k], keys + (k,)):
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
                        log.debug(
                            "Pruned keys %s from config node '%s' for target %s",
                            unknown,
                            path or "<root>",
                            target,
                        )

        for k, v in node.items():
            if isinstance(v, DictConfig):
                _prune(v, f"{path}.{k}" if path else k)

    _prune(config)


def read_user_config(
    cfg: Union[DictConfig, PosixPath, str, None] = None,
    config_path: str = "configs",
    config_name: str = "train.yaml",
    overrides: Optional[Union[str, List[str]]] = None,
):
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
        value = str(escape_all(v)).replace("'", "")
        if value == "None":
            value = "null"
        override_list.append(f"++{key}={value}")

    try:
        cli_overrides = hydra.core.hydra_config.HydraConfig.get().overrides.task
    except Exception:
        cli_overrides = []
    finally:
        override_list.extend(cli_overrides)

    if overrides is not None:
        if isinstance(overrides, str):
            overrides = [overrides]
        override_list.extend(overrides)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    if use_config_dir:
        context = initialize_config_dir(version_base=None, config_dir=config_path)
    else:
        context = initialize(version_base=None, config_path=config_path)
    with context:
        composed_cfg = compose(config_name=config_name, overrides=override_list)

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
    if isinstance(data, (dict, DictConfig)):
        return {k: escape_all(v) for k, v in data.items()}
    if isinstance(data, (list, ListConfig)):
        return [escape_all(item) for item in data]
    return data
