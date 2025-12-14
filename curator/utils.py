import torch
from e3nn.util.jit import script
from omegaconf import open_dict, OmegaConf, DictConfig, ListConfig
from hydra import compose, initialize, initialize_config_dir
import hydra
from hydra.utils import instantiate, get_class
import inspect
from collections import abc
import logging
from ase import units
from pathlib import Path, PosixPath
from typing import Any, List, Optional, Tuple, Union, Dict
import numpy as np
import torch.serialization as torch_serialization

from ase.data import chemical_symbols
import torch, re

def register_resolvers():
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y, replace=True)
    OmegaConf.register_new_resolver("multiply_fs", lambda x: x * units.fs, replace=True)
    OmegaConf.register_new_resolver("divide_by_fs", lambda x: x / units.fs, replace=True)

def create_model_from_mace(mace_model):
    from curator.layer import GlobalRescaleShift, Strain, AtomwiseReduce, PairwiseDistance, GradientOutput, RealAgnosticInteractionBlock, RealAgnosticResidualInteractionBlock
    from curator.model import NeuralNetworkPotential, MACE
    interaction_map = {
        'RealAgnosticInteractionBlock': RealAgnosticInteractionBlock,
        'RealAgnosticResidualInteractionBlock': RealAgnosticResidualInteractionBlock,
    }

    input_modules = [Strain(), PairwiseDistance(compute_distance_from_R=True)]
    num_heads = len(getattr(mace_model, "heads", ["Default"])) if hasattr(mace_model, "heads") else 1
    interaction_cls_first = interaction_map.get(mace_model.interactions[0].__class__.__name__, RealAgnosticInteractionBlock)
    interaction_cls = interaction_map.get(mace_model.interactions[0].__class__.__name__, RealAgnosticResidualInteractionBlock)
    curator_mace = MACE(
        cutoff=float(mace_model.r_max),
        num_interactions=len(mace_model.interactions),
        correlation=[contraction.correlation for contraction in mace_model.products[0].symmetric_contractions.contractions],
        species=[chemical_symbols[i] for i in mace_model.atomic_numbers],
        hidden_irreps=mace_model.interactions[0].hidden_irreps,
        edge_sh_irreps=mace_model.spherical_harmonics.irreps_out,
        avg_num_neighbors=mace_model.interactions[0].avg_num_neighbors,
        MLP_irreps=mace_model.readouts[-1].hidden_irreps,
        num_basis=len(mace_model.radial_embedding.bessel_fn.bessel_weights),
        power=float(mace_model.radial_embedding.cutoff_fn.p),
        interaction_cls=interaction_cls,
        interaction_cls_first=interaction_cls_first,
        num_heads=num_heads,
    )

    output_modules = [
        AtomwiseReduce(output_key='energy'),
        GlobalRescaleShift(
            scale_by=float(mace_model.scale_shift.scale),
            shift_by=float(mace_model.scale_shift.shift),
            atomic_energies={
                int(idx): float(e)
                for idx, e in zip(mace_model.atomic_numbers, mace_model.atomic_energies_fn.atomic_energies.squeeze())
            },
        ),
        GradientOutput(model_outputs=['energy', 'forces'], grad_on_edge_diff=False, grad_on_positions=True),
    ]
    curator_mace.embeddings.radial_basis.basis.load_state_dict(mace_model.radial_embedding.bessel_fn.state_dict(), strict=False)
    curator_mace.embeddings.chemical_embedding.linear.load_state_dict(mace_model.node_embedding.linear.state_dict(), strict=False)
    for i in range(len(mace_model.interactions)):
        curator_mace.interactions[i].avg_num_neighbors = torch.tensor(mace_model.interactions[i].avg_num_neighbors)
        curator_mace.interactions[i].load_state_dict(mace_model.interactions[i].state_dict(), strict=False)
        curator_mace.products[i].load_state_dict(mace_model.products[i].state_dict())
        if i < len(mace_model.readouts) - 1 and hasattr(mace_model.readouts[i], "linear"):
            curator_mace.readout_mlp[i].load_state_dict(mace_model.readouts[i].linear.state_dict(), strict=False)
        elif i < len(mace_model.readouts):
            curator_mace.readout_mlp[i].load_state_dict(mace_model.readouts[i].state_dict(), strict=False)
    nnp = NeuralNetworkPotential(
        input_modules=input_modules,
        representation=curator_mace,
        output_modules=output_modules,
    )
    
    return nnp


def convert_mace_to_curator(mace_path: Union[str, Path], output_path: Union[str, Path], foundation: bool = False, device: Optional[torch.device] = None) -> Path:
    """Load a mace model checkpoint and save a curator-style model."""
    torch_serialization.add_safe_globals([slice])
    mace_path = Path(mace_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(mace_path, map_location=device)
    mace_model = None
    if isinstance(obj, torch.nn.Module):
        mace_model = obj
    elif isinstance(obj, dict):
        mace_model = obj.get("model")
        if mace_model is None and "state_dict" in obj:
            raise TypeError("MACE checkpoint does not include an instantiated model; please provide a TorchScript or full model checkpoint.")
    if mace_model is None:
        raise TypeError(f"Unsupported MACE checkpoint format at {mace_path}")
    curator_model = create_model_from_mace(mace_model, foundation=foundation)
    output_path = Path(output_path)
    torch.save(curator_model, output_path)
    return output_path

def _build_mace_from_curator(curator_model):
    from curator.layer import GlobalRescaleShift
    from curator.model import NeuralNetworkPotential, MACE
    """Best-effort recreation of a mace.modules.models.ScaleShiftMACE from a Curator MACE model."""
    from mace.modules import models as mace_models
    from mace.modules import blocks as mace_blocks

    if isinstance(curator_model, NeuralNetworkPotential):
        repr_model = curator_model.representation
        output_modules = list(curator_model.output_modules)
    else:
        repr_model = curator_model
        output_modules = []

    if not isinstance(repr_model, MACE):
        raise TypeError("Provided model is not a Curator MACE representation.")

    # species / atomic numbers
    atomic_numbers = list(range(repr_model.embeddings.onehot_embedding.num_elements))
    mapper = getattr(repr_model.embeddings.onehot_embedding, "type_mapper", None)
    if mapper is not None:
        atomic_numbers = [int(z) for z in mapper.index_to_Z.cpu().tolist()]

    heads = [f"head_{i}" for i in range(repr_model.num_heads)]
    correlation = [
        contraction.correlation for contraction in repr_model.products[0].symmetric_contractions.contractions
    ]
    num_basis = len(repr_model.embeddings.radial_basis.basis.bessel_weights)
    num_polynomial_cutoff = int(repr_model.embeddings.radial_basis.cutoff_fn.power)
    avg_num_neighbors = float(repr_model.interactions[0].avg_num_neighbors.squeeze())

    scale, shift = 1.0, 0.0
    atomic_energies = torch.zeros(len(atomic_numbers))
    for m in output_modules:
        if isinstance(m, GlobalRescaleShift):
            scale = float(m.scale_by)
            shift = float(m.shift_by)
            atomic_energies = m.atomic_energies[atomic_numbers]
            break

    mace_model = mace_models.ScaleShiftMACE(
        atomic_inter_scale=scale,
        atomic_inter_shift=shift,
        r_max=float(repr_model.cutoff),
        num_bessel=num_basis,
        num_polynomial_cutoff=num_polynomial_cutoff,
        max_ell=repr_model.lmax,
        interaction_cls=mace_blocks.InteractionBlock,
        interaction_cls_first=mace_blocks.InteractionBlock,
        num_interactions=len(repr_model.interactions),
        num_elements=repr_model.embeddings.onehot_embedding.num_elements,
        hidden_irreps=repr_model.hidden_irreps,
        MLP_irreps=repr_model.MLP_irreps,
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=atomic_numbers,
        correlation=correlation,
        gate=torch.nn.functional.silu,
        heads=heads,
    )

    # load weights (best effort, shapes match original MACE layout)
    mace_model.radial_embedding.bessel_fn.load_state_dict(
        repr_model.embeddings.radial_basis.basis.state_dict(), strict=False
    )
    mace_model.radial_embedding.cutoff_fn.load_state_dict(
        repr_model.embeddings.radial_basis.cutoff_fn.state_dict(), strict=False
    )
    mace_model.node_embedding.linear.load_state_dict(
        repr_model.embeddings.chemical_embedding.linear.state_dict(), strict=False
    )
    for i in range(len(repr_model.interactions)):
        mace_model.interactions[i].load_state_dict(repr_model.interactions[i].state_dict(), strict=False)
        mace_model.products[i].load_state_dict(repr_model.products[i].state_dict(), strict=False)
        if i < len(mace_model.readouts) - 1 and hasattr(mace_model.readouts[i], "linear"):
            mace_model.readouts[i].linear.load_state_dict(repr_model.readout_mlp[i].state_dict(), strict=False)
        elif i < len(mace_model.readouts):
            mace_model.readouts[i].load_state_dict(repr_model.readout_mlp[i].state_dict(), strict=False)
    return mace_model


def convert_curator_to_mace(curator_path: Union[str, Path], output_path: Union[str, Path], device: Optional[torch.device] = None) -> Path:
    """Convert a saved Curator MACE model back to a mace.modules.models.ScaleShiftMACE checkpoint."""
    curator_path = Path(curator_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(curator_path, device=device, load_compiled=False, load_weights_only=False)
    mace_model = _build_mace_from_curator(model)
    output_path = Path(output_path)
    torch.save(mace_model, output_path)
    return output_path

def dummy_load(*args, **kwargs):
    original_torch_jit_load = torch.jit.load
    def torch_jit_load_cpu(*args, **kwargs):
        if not torch.cuda.is_available():
            kwargs['map_location'] = torch.device('cpu')
        return original_torch_jit_load(*args, **kwargs)
    torch.jit.load = torch_jit_load_cpu

def camel_to_snake(name: str) -> str:
    s1 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return s1.lower()

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

    model_file = Path(model_file)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TorchScript
    if model_file.suffix == '.pt' and load_compiled:
        model = torch.jit.load(model_file, map_location=torch.device(device))
        try:
            model.to(device)
        except Exception:
            pass
        return model

    obj = torch.load(model_file, map_location=torch.device(device))

    if isinstance(obj, torch.nn.Module):
        obj.to(device)
        return obj

    if isinstance(obj, dict):
        stored_model = obj.get('model')
        if not load_weights_only and isinstance(stored_model, torch.nn.Module):
            stored_model.to(device)
            return stored_model

        model_cfg = cfg.model if cfg is not None else obj.get('model_params')
        model_cfg = _copy_config(model_cfg)
        if model_cfg is None:
            raise ValueError("Checkpoint does not contain model parameters to instantiate.")
        _listify_config_field(model_cfg, "input_modules")
        _listify_config_field(model_cfg, "output_modules")
        model = instantiate(model_cfg, _convert_="all")

        data_cfg = cfg.data if cfg is not None else obj.get('data_params')
        data_cfg = _copy_config(data_cfg)
        if data_cfg is not None:
            datamodule = instantiate(data_cfg, _convert_="all")
            if hasattr(datamodule, 'setup'):
                datamodule.setup()
            if hasattr(model, 'initialize_modules'):
                model.initialize_modules(datamodule)

        sd = obj.get('state_dict')
        if sd is None:
            raise ValueError("Checkpoint is missing a state_dict.")
        stripped = {k.replace("model.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(stripped, strict=False)
        model.to(device)
        return model

    raise TypeError(f"Unsupported checkpoint format at {model_file}.")


def load_model(
    model_file: Union[str, Path],
    device = None,
    load_compiled: bool = True,
    load_weights_only: bool = False,
    cfg: Optional[DictConfig] = None,
) -> torch.nn.Module:
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
            p = Path(m)
            if p.is_file() and p.suffix in {'.pt', '.pth', '.ckpt'}:
                models.append(
                    load_model(
                        p,
                        device,
                        load_compiled,
                        load_weights_only=load_weights_only,
                        cfg=cfg,
                    )
                )
            else:
                best, _ = find_best_model(p)
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

def ensure_list(value: Any):
    """Convert dictionary-like Hydra nodes to list values."""

    if isinstance(value, DictConfig):
        return [value[k] for k in value.keys()]
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, ListConfig):
        return list(value)
    return value

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

def find_best_model(run_path: Union[str, Path]) -> Tuple[Path, Optional[float]]:
    """Return best ckpt path under a run directory or the path itself if it is a .ckpt."""

    run_path = Path(run_path)
    if run_path.suffix == '.ckpt':
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

# Set up Early stopping for pytorch training 
class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, best_loss):
        if val_loss - best_loss > self.min_delta:
            self.counter +=1
            if self.counter >= self.patience:  
                self.early_stop = True
        return self.early_stop

def deploy_model(model, file_path: str):
    compiled_model = script(model)
    compiled_model.save()

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

# Ugly workaround for specifying config files outside of the package
def read_user_config(cfg: Union[DictConfig, PosixPath, str, None]=None, config_path="configs", config_name="train.yaml"):
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

    override_list = []
    if "defaults" in user_cfg:
        default_list = user_cfg.pop("defaults")
        for d in default_list:
            if isinstance(d, (dict, DictConfig)):
                for k, v in d.items():
                    override_list.append(f"{k}={v}")
    
    for path in sorted(converted_fields):
        override_list.append(f"~{path}")

    for k, v in get_all_pairs(user_cfg):
        key = ".".join(k)
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

    # reload hyperparameters         
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path=config_path):
        composed_cfg = compose(config_name=config_name, overrides=override_list)
    
    # Allow write access to unknown fields
    OmegaConf.set_struct(composed_cfg, False)

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

def scatter_reduce(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: torch.Tensor = None,
                   reduce: str = 'sum', include_self: bool = False) -> torch.Tensor:
    """
    Reduces all values from the `src` tensor into `out` at the indices specified in the `index` tensor
    along the dimension `dim` using the specified reduction ('sum', 'mean', 'max', 'min').

    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices of elements to scatter.
            Must have the same size as `src` at dimension `dim` or be broadcastable to that size.
        dim (int): The axis along which to index.
        out (torch.Tensor, optional): The destination tensor. If None, a new tensor is created.
        reduce (str): The reduction operation to apply ('sum', 'mean', 'max', 'min').
        include_self (bool): Whether to include existing values in `out` during reduction.

    Returns:
        torch.Tensor: The resulting tensor with the reduced values scattered at the specified indices.
    """
    # Validate reduce operation
    if reduce not in ['sum', 'mean', 'max', 'min']:
        raise ValueError(f"Invalid reduce operation '{reduce}'. Supported operations: 'sum', 'mean', 'max', 'min'.")

    # Ensure index has the same number of dimensions as src
    index = _broadcast(index, src, dim)

    # Determine size of output tensor along dimension `dim`
    output_size = list(src.size())
    output_size[dim] = int(index.max()) + 1

    # Initialize out tensor
    if out is None:
        if reduce in ['sum', 'mean']:
            out = torch.zeros(output_size, dtype=src.dtype, device=src.device)
        elif reduce == 'max':
            out = torch.full(output_size, torch.finfo(src.dtype).min, dtype=src.dtype, device=src.device)
        elif reduce == 'min':
            out = torch.full(output_size, torch.finfo(src.dtype).max, dtype=src.dtype, device=src.device)
    else:
        if not include_self:
            if reduce in ['sum', 'mean']:
                out.zero_()
            elif reduce == 'max':
                out.fill_(torch.finfo(src.dtype).min)
            elif reduce == 'min':
                out.fill_(torch.finfo(src.dtype).max)

    if reduce == 'sum':
        out.scatter_add_(dim, index, src)
    elif reduce == 'mean':
        out.scatter_add_(dim, index, src)
        # Count occurrences for mean calculation
        count = torch.zeros_like(out)
        ones = torch.ones_like(src, dtype=src.dtype)
        count.scatter_add_(dim, index, ones)
        zero_mask = count == 0
        count[zero_mask] = 1
        out = out / count
    elif reduce == 'max':
        out.scatter_(dim, index, torch.max(out.gather(dim, index), src))
    elif reduce == 'min':
        out.scatter_(dim, index, torch.min(out.gather(dim, index), src))

    return out

# Function to check if cell is upper-triangular
def is_upper_triangular(cell):
    return np.allclose(np.tril(cell, -1), 0)

# transform lower-triangular cell to upper-triangular cell
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
    """Extract configurations of a model, which can then be used to instantiate a new one."""
    rep = model.representation
    if model.representation.__class__.__name__ == 'MACE':
        species = list(rep.embeddings.onehot_embedding.type_mapper.symbol_to_type.keys())
        try:
            correlation = (
                len(rep.products[0].symmetric_contractions.contractions[0].weights) + 1
            )
        except AttributeError:
            correlation = rep.products[0].symmetric_contractions.contraction_degree

        try:
            gate = rep.readout.readout_mlp[0].activation.acts[0].f
        except:
            gate = rep.readout_mlp[-1].non_linearity.acts[0].f
        rep_config = {
            "cutoff": rep.cutoff,
            "num_interactions": len(rep.interactions),
            "correlation": correlation,
            "interaction_cls": rep.interactions[-1].__class__,
            "interaction_cls_first": rep.interactions[0].__class__,
            "radial_MLP": rep.interactions[0].conv_tp_weights.hs[1:-1],
            "species": list(rep.embeddings.onehot_embedding.type_mapper.symbol_to_type.keys()),
            "num_elements": len(species),
            "hidden_irreps": rep.hidden_irreps,
            "edge_sh_irreps": rep.edge_sh_irreps,
            "node_irreps": rep.node_irreps,
            "MLP_irreps": rep.MLP_irreps,
            "avg_num_neighbors": float(rep.interactions[0].avg_num_neighbors),
            "num_basis": rep.embeddings.radial_basis.basis.num_basis,
            "power": rep.embeddings.radial_basis.cutoff_fn.p,
            "gate": gate,
        }
    elif model.representation.__class__.__name__ == 'Nequip':
        species = list(rep.embeddings.onehot_embedding.type_mapper.symbol_to_type.keys())

        rep_config = {
            "cutoff": rep.cutoff,
            "num_interactions": len(rep.interactions),
            "species": list(rep.embeddings.onehot_embedding.type_mapper.symbol_to_type.keys()),
            "num_elements": len(species),
            "hidden_irreps": rep.hidden_irreps,
            "edge_sh_irreps": rep.edge_sh_irreps,
            "node_irreps": rep.node_irreps,
            "num_basis": rep.embeddings.radial_basis.basis.num_basis,
            "power": rep.embeddings.radial_basis.cutoff_fn.p,
            "resnet": rep.interactions[0].resnet,
            "nonlinearity_type": rep.nonlinearity_type,
            "nonlinearity_scalars": rep.nonlinearity_scalars,
            "nonlinearity_gates": rep.nonlinearity_gates,
            "convolution_kwargs": rep.convolution_kwargs,
        }
    elif model.representation.__class__.__name__ == 'Painn':
        rep_config = {
            "cutoff": rep.cutoff,
            "num_interactions": rep.num_interactions,
            "num_features": rep.num_features,
            "num_basis": rep.num_basis,
        }
    return rep_config

def get_kmax_pairs(
    max_L: int, correlation: int, num_layers: int
) -> List[Tuple[int, int]]:
    """Determine kmax pairs based on max_L and correlation"""
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
    """Transfer symmetric contraction weights"""
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
        target_dict[f"products.{i}.symmetric_contractions.weight"] = wm

def get_transfer_keys(num_layers: int) -> List[str]:
    """Get list of keys that need to be transferred"""
    return [
        "embeddings.chemical_embedding.linear.weight",
        *[f"readout.readouts.{j}.linear.weight" for j in range(num_layers - 1)],
        *[f"readout.readout_mlp.{i}.linear.weight" for i in range(2)],
        *[f"readout.readouts.{num_layers - 1}.{i}.linear.weight" for i in range(2)],
    ] + [
        s
        for j in range(num_layers)
        for s in [
            f"interactions.{j}.linear_up.weight",
            *[f"interactions.{j}.conv_tp_weights.layer{i}.weight" for i in range(4)],
            f"interactions.{j}.linear.weight",
            f"interactions.{j}.skip_tp.weight",
            f"products.{j}.linear.weight",
        ]
    ]

def load_e3nn_weights(source_model, target_model):
    """Load weights from an e3nn model to cuequivariance model"""
    source_dict = source_model.representation.state_dict()
    target_dict = target_model.representation.state_dict()

    # Transfer main weights
    num_layers = len(source_model.representation.interactions)
    transfer_keys = get_transfer_keys(num_layers)
    for key in transfer_keys:
        if key in source_dict:  # Check if key exists
            target_dict[key] = source_dict[key]
        else:
            logging.warning(f"Key {key} not found in source model")

    # unsqueeze linear and skip_tp weights
    for key in source_dict.keys():
        if any(x in key for x in ["linear", "skip_tp"]) and "weight" in key:
            target_dict[key] = target_dict[key].unsqueeze(0)
    
    # transfer symmetric contractions
    lmax = source_model.representation.lmax
    try:
        correlation = (
            len(source_model.representation.products[0].symmetric_contractions.contractions[0].weights) + 1
        )
    except AttributeError:
        correlation = source_model.representation.products[0].symmetric_contractions.contraction_degree
    transfer_symmetric_contractions(source_dict, target_dict, lmax, correlation, num_layers)

    transferred_keys = set(transfer_keys)
    remaining_keys = (
        set(source_dict.keys()) & set(target_dict.keys()) - transferred_keys
    )
    remaining_keys = {k for k in remaining_keys if "symmetric_contraction" not in k}
    if remaining_keys:
        for key in remaining_keys:
            if source_dict[key].shape == target_dict[key].shape:
                logging.debug(f"Transferring additional key: {key}")
                target_dict[key] = source_dict[key]
            else:
                logging.warning(
                    f"Shape mismatch for key {key}: "
                    f"source {source_dict[key].shape} vs target {target_dict[key].shape}"
                )

    target_model.representation.load_state_dict(target_dict)

def convert_e3nn_to_cueq(model):
    rep_config = get_representation_config(model)
    rep_config["use_cueq"] = True
    cueq_rep = model.representation.__class__(**rep_config)

    cueq_model = model.__class__(
        input_modules=list(model.input_modules),
        output_modules=list(model.output_modules),
        representation=cueq_rep,
        model_outputs=model.model_outputs,
    )

    load_e3nn_weights(model, cueq_model)

    return cueq_model

def update_model(model):
    import warnings
    rep_config = get_representation_config(model)
    new_rep = model.representation.__class__(**rep_config)

    old_state_dict = model.representation.state_dict()

    # replace readout weight name
    try:
        if new_rep.__class__.__name__ == 'MACE':
            for i in range(len(model.representation.readout_mlp)):
                # replace normal layers
                if i != len(model.representation.readout_mlp) - 1:
                    for name in ['weight', 'bias', 'output_mask']:
                        old_state_dict[f'readout.readouts.{i}.linear.{name}'] = old_state_dict.pop(f'readout_mlp.{i}.{name}')
                    warnings.warn("Rename weights in deprecated readouts.")
                else:
                # replace readout_mlp layer
                    for j in range(len(new_rep.readout.readout_mlp)):
                        for name in ['weight', 'bias', 'output_mask']:
                            old_state_dict[f'readout.readouts.{i}.{j}.linear.{name}'] = old_state_dict.pop(f'readout_mlp.{i}.linear_{j+1}.{name}')
                            old_state_dict[f'readout.readout_mlp.{j}.linear.{name}'] = old_state_dict[f'readout.readouts.{i}.{j}.linear.{name}']
                    warnings.warn("Rename weights in deprecated readout_mlp.")
        elif new_rep.__class__.__name__ == 'Painn':
            for i in range(len(new_rep.readout.readout_mlp)):
                for name in ['weight', 'bias']:
                    old_state_dict[f'readout.readout_mlp.{i}.linear.{name}'] = old_state_dict.pop(f'readout_mlp.{2*i}.{name}')
            warnings.warn("Rename weights in deprecated readout_mlp.")

    except KeyError:
        pass

    try:
        new_rep.load_state_dict(old_state_dict)
    except:
        warnings.warn("Loading weights from old model failed!")

    output_modules = model.output_modules
    # fix output modules
    try:
        # modify modules in-place
        for i, m in enumerate(output_modules):
            if m.__class__.__name__ == 'GradientOutput':
                output_modules[i] = m.__class__(
                    grad_on_edge_diff = m.grad_on_edge_diff,
                    grad_on_positions = m.grad_on_positions,
                    compute_edge_forces = getattr(m, 'compute_edge_forces', False),
                    compute_edge_forces_only = getattr(m, 'compute_edge_forces_only', False),
                    model_outputs = m.model_outputs,
                )
                warnings.warn('Replace GradientOutput module.')
            if m.__class__.__name__ == 'GlobalRescaleShift':
                scale_by = m.scale_by.detach().clone().cpu().squeeze().item()
                shift_by = m.shift_by.detach().clone().cpu().squeeze().item()
                output_modules[i] = m.__class__(
                    scale_by=scale_by,
                    shift_by=shift_by,
                    scale_trainable=isinstance(getattr(m, "scale_by", None), torch.nn.Parameter),
                    shift_trainable=isinstance(getattr(m, "shift_by", None), torch.nn.Parameter),
                    scale_keys=list(m.scale_keys),
                    shift_keys=list(m.shift_keys),
                    atomwise_shift=bool(getattr(m, "atomwise_shift", False)),
                    atomwise_normalization=bool(m.atomwise_normalization),
                    output_keys=list(m.output_keys),
                    atomic_energies=(
                        m.atomic_energies.detach().clone()
                        if getattr(m, "shift_by_E0", torch.tensor(False)).item()
                        else None
                    )
                )
                warnings.warn('Replace GlobalRescaleShift module.')
        # remove module
        for i, m in enumerate(output_modules):
            if m.__class__.__name__ == 'AtomwiseReduce':
                output_modules.pop(i)
                warnings.warn('Remove AtomwiseReduce module in output modules.')
    except:
        pass

    new_model = model.__class__(
        input_modules=list(model.input_modules),        # almost no update in input_modules and output_modules
        output_modules=list(output_modules),
        representation=new_rep,
        model_outputs=model.model_outputs,
    )

    return new_model

def upgrade_checkpoint(
    ckpt_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Path:
    """Upgrade an older Curator checkpoint by rebuilding its stored model.

    Loads the checkpoint on CPU by default (so conversion works without GPUs),
    rebuilds the model via ``update_model``, and writes a new checkpoint.
    """
    import curator.model.compat  # registers legacy class aliases for torch.load
    from collections import OrderedDict

    ckpt_path = Path(ckpt_path)
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if output_path is None:
        output_path = ckpt_path.with_name(f"{ckpt_path.stem}_converted{ckpt_path.suffix}")
    output_path = Path(output_path)

    obj = torch.load(ckpt_path, map_location=device)

    if isinstance(obj, torch.nn.Module):
        upgraded_model = update_model(obj)
        torch.save(upgraded_model, output_path)
        return output_path

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    if "model" not in obj:
        raise KeyError("Checkpoint is missing 'model' entry to upgrade.")

    upgraded_model = update_model(obj["model"])
    obj["model"] = upgraded_model
    if "state_dict" in obj:
        state_dict = upgraded_model.state_dict()
        new_state_dict = OrderedDict()
        for k in state_dict.keys():
            new_state_dict['model.' + k] = state_dict[k]
        obj["state_dict"] = new_state_dict
    torch.save(obj, output_path)
    return output_path
