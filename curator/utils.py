import torch
from e3nn.util.jit import script
from omegaconf import open_dict, OmegaConf, DictConfig, ListConfig
from hydra import compose, initialize
import hydra
from hydra.utils import instantiate
from collections import abc
import logging
from ase import units
from pathlib import Path, PosixPath
from typing import Optional, Union
import numpy as np

def register_resolvers():
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y, replace=True)
    OmegaConf.register_new_resolver("multiply_fs", lambda x: x * units.fs, replace=True)
    OmegaConf.register_new_resolver("divide_by_fs", lambda x: x / units.fs, replace=True)

def dummy_load(*args, **kwargs):
    original_torch_jit_load = torch.jit.load
    def torch_jit_load_cpu(*args, **kwargs):
        if not torch.cuda.is_available():
            kwargs['map_location'] = torch.device('cpu')
        return original_torch_jit_load(*args, **kwargs)
    torch.jit.load = torch_jit_load_cpu

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

def load_model(model_file, device, load_compiled: bool=True):
    if model_file.suffix == '.pt' and load_compiled:
        model = torch.jit.load(model_file, map_location=torch.device(device))
    else:
        model_dict = torch.load(model_file, map_location=torch.device(device))
        if 'model' in model_dict:
            model = model_dict['model']
        else:
            datamodule = instantiate(model_dict['data_params'])
            datamodule.setup()
            model = instantiate(model_dict['model_params'])
            model.initialize_modules(datamodule)
            model.load_state_dict(model_dict['model'])
    model.to(device)

    return model

def load_models(model_paths, device, load_compiled: bool=True):
    if isinstance(model_paths, str):
        model_paths = [model_paths]
    
    models = []
    for model_path in model_paths:
        path = Path(model_path)
        if path.is_file() and (path.suffix == '.pt' or path.suffix == '.pth' or path.suffix == '.ckpt'):
            models.append(load_model(path, device, load_compiled))
        else:
            model_path, _ = find_best_model(run_path=model_path)
            models.append(load_model(model_path, device, load_compiled))
    
    return models

def find_best_model(run_path):
    # return best model path if a path is provided, else checkpoint itself
    if Path(run_path).suffix == '.ckpt':
        return run_path, None
    else:
        model_path = [f for f in Path(run_path).glob("best_model_*.ckpt")]
        val_loss = float('inf')
        index = 0
        for i, p in enumerate(model_path):
            loss = float(str(p).split('=')[-1].rstrip('.ckpt'))
            if loss < val_loss:
                val_loss = loss
                index = i
    return model_path[index], val_loss

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

# Ugly workaround for specifying config files outside of the package
def read_user_config(cfg: Union[DictConfig, PosixPath, str, None]=None, config_path="configs", config_name="train.yaml"):
    # load cfg
    if isinstance(cfg, DictConfig):
        user_cfg = cfg
    elif isinstance(cfg, (PosixPath, str)):
        user_cfg = OmegaConf.load(cfg)
    else:
        user_cfg = OmegaConf.create()

    override_list = []
    if "defaults" in user_cfg:
        default_list = user_cfg.pop("defaults")
        for d in default_list:
            if isinstance(d, (dict, DictConfig)):
                for k, v in d.items():
                    override_list.append(f"{k}={v}")
    
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
        
    return composed_cfg

def escape_special_characters(value: str) -> str:
    special_characters = r"\()[]{}:=,"
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