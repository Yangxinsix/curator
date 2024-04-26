import torch
from e3nn.util.jit import script
from omegaconf import open_dict, OmegaConf, DictConfig
from hydra import compose, initialize
import hydra
from collections import abc
import logging

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
def read_user_config(filepath, config_path="configs", config_name="train.yaml"):
    # load user defined config file
    user_cfg = OmegaConf.load(filepath)
    
    # get override list
    override_list = []
    if "defaults" in user_cfg:
        default_list = user_cfg.pop("defaults")
        for d in default_list:
            for k, v in d.items():
                override_list.append(f"{k}={v}")
    
    for k, v in get_all_pairs(user_cfg):
        key = ".".join(k)
        value = str(v).replace("'", "")
        if value == 'None':
            value = 'null'
        override_list.append(f'++{key}={value}')
    
    # reload hyperparameters         
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name, overrides=override_list)
    
    # Allow write access to unknown fields
    OmegaConf.set_struct(cfg, False)
        
    return cfg
