import hydra
from omegaconf import DictConfig

from .common import CONFIGS_PATH


@hydra.main(config_path=CONFIGS_PATH, config_name="train", version_base=None)
def train(config: DictConfig) -> None:
    from .train import run_train_config

    return run_train_config(config)


@hydra.main(config_path=CONFIGS_PATH, config_name="train", version_base=None)
def tmptrain(config: DictConfig):
    from .train import run_tmp_train_config

    return run_tmp_train_config(config)


tmp_train = tmptrain


@hydra.main(config_path=CONFIGS_PATH, config_name="simulate", version_base=None)
def simulate(config: DictConfig):
    from .simulate import run_simulate_config

    return run_simulate_config(config)


@hydra.main(config_path=CONFIGS_PATH, config_name="select", version_base=None)
def select(config: DictConfig):
    from .select import run_select_config

    return run_select_config(config)


@hydra.main(config_path=CONFIGS_PATH, config_name="label", version_base=None)
def label(config: DictConfig):
    from .label import run_label_config

    return run_label_config(config)


@hydra.main(config_path=CONFIGS_PATH, config_name="evaluate", version_base=None)
def evaluate(config: DictConfig):
    from .evaluate import run_evaluate_config

    return run_evaluate_config(config)
