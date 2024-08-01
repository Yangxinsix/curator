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
        model_path = [f for f in Path(run_path).glob("best_model*.ckpt")]
        val_loss = float('inf')
        index = 0
        for i, p in enumerate(model_path):
            loss = float(p.split('=')[-1].rstrip('.ckpt'))
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
        cfg = compose(config_name=config_name, overrides=override_list)
    
    # Allow write access to unknown fields
    OmegaConf.set_struct(cfg, False)
        
    return cfg

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