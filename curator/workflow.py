#!/usr/bin/env python3
from myqueue.workflow import run
from myqueue.task import Task
from typing import Optional, Dict, Tuple, Final, Union
from omegaconf import DictConfig, ListConfig, OmegaConf
from curator.utils import read_user_config, register_resolvers
from pathlib import Path
import warnings

register_resolvers()

# TODO: manage restart all job

# paths for jobs
train_path: Final = 'train'
simulate_path: Final = 'simulate'
select_path: Final = 'select'
label_path: Final = 'label'

# keys inferred in workflow: model_path, init_traj, pool_set, al_info, start_index
# user specified arguments that can be different from worflow defaults: read_traj (init_traj), image_index (start_index)

path_set = set([
    'run_path', 
    'datapath',
    'train_path',
    'val_path',
    'test_path',
    'model_path',
    'load_model',       # used to load specific model but not model trained in workflow
    'init_traj',
    'read_traj',        # used to load specific initial trajectory but not trajectory generated in workflow
])

def resolve_paths(config: Union[DictConfig, dict], base_dir='.', path_set=path_set):
    # resolve absolute paths inside path_set
    for key, value in config.items():
        if key in path_set:
            if isinstance(value, str):
                abs_path = Path(base_dir) / value
                config[key] = str(abs_path.resolve())
            elif isinstance(value, (list, ListConfig)):
                abs_paths = [str((Path(base_dir) / item).resolve()) for item in value]
                config[key] = abs_paths
        elif isinstance(value, (dict, DictConfig)):
            resolve_paths(value, base_dir, path_set)


def _deprecated_output(path: str, replacement: str):
    warnings.warn(
        f"Simulation workflow hand-off via '{path}' is deprecated; declare '{replacement}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _select_simulation_output(cfg: DictConfig, candidates: list[tuple[str, bool]], fallback=None):
    for path, deprecated in candidates:
        value = OmegaConf.select(cfg, path, default=None)
        if value is not None:
            if deprecated:
                replacement = path.replace("simulator.out_traj", "outputs.pool_set")
                replacement = replacement.replace("simulator.uncertain_traj", "outputs.uncertain_set")
                replacement = replacement.replace("simulator.uncertainty.save_uncertain_atoms", "outputs.uncertain_set")
                replacement = replacement.replace("callbacks.thermo.save_path", "outputs.uncertain_set")
                _deprecated_output(path, replacement)
            return value
    return fallback


def _select_callback_output(cfg: DictConfig, *, target_suffixes: tuple[str, ...], field: str):
    callbacks = OmegaConf.select(cfg, 'simulator.callbacks', default=None)
    if not isinstance(callbacks, (list, tuple, ListConfig)):
        return None
    for cb in callbacks:
        if not isinstance(cb, (dict, DictConfig)):
            continue
        target = cb.get('_target_', '')
        if any(str(target).endswith(suffix) for suffix in target_suffixes):
            value = cb.get(field)
            if value is None:
                continue
            if field == 'path' and isinstance(value, str) and '{' in value:
                continue
            _deprecated_output(f"simulator.callbacks[*].{field}", f"outputs.{'pool_set' if field == 'path' else 'uncertain_set'}")
            return value
    return None


def _resolve_iteration_reference(value, iteration: int):
    if isinstance(value, str):
        return value.replace(f'iter_{iteration}', f'iter_{iteration-1}')
    if isinstance(value, (list, tuple, ListConfig)):
        return [_resolve_iteration_reference(item, iteration) for item in value]
    return value


def _as_path_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _simulation_outputs(cfg: DictConfig) -> tuple[Union[str, list[str]], Optional[Union[str, list[str]]], Union[str, list[str]]]:
    pool_set = _select_simulation_output(
        cfg,
        [
            ('outputs.pool_set', False),
            ('simulator.out_traj', True),
        ],
        fallback=_select_callback_output(
            cfg,
            target_suffixes=('TrajectoryWriter',),
            field='path',
        ),
    )
    if pool_set is None:
        raise ValueError("Simulation config must declare outputs.pool_set or legacy simulator.out_traj.")

    uncertain_set = _select_simulation_output(
        cfg,
        [
            ('outputs.uncertain_set', False),
            ('simulator.uncertain_traj', True),
            ('simulator.uncertainty.save_uncertain_atoms', True),
        ],
        fallback=_select_callback_output(
            cfg,
            target_suffixes=('ThermoWithUncertainty', 'MDLogger', 'TorchSimThermoLogger'),
            field='save_path',
        ),
    )
    restart_source = _select_simulation_output(
        cfg,
        [
            ('outputs.restart_source', False),
            ('outputs.pool_set', False),
            ('simulator.out_traj', True),
        ],
        fallback=pool_set,
    )
    return pool_set, uncertain_set, restart_source

def train(
    deps: list[Task],
    config: DictConfig,
    iteration: Optional[int] = 0,
) -> Tuple[list[str], list[Task]]:
    ''' 
    Runs a train task for each model in the ensemble.
    This is done through the three steps for each model:
        - Create a new directory 
        - Save user_cfg file
        - Run task
    '''
    tasks = []
    model_paths = []              # collect models for simulation and active learning
    arguments = ['cfg=train.yaml']
    config = config.copy()

    # get general keys
    general = config.pop('general')

    # load multiple models
    for i, (name, job_config) in enumerate(config.items()):
        # define start_iteration:
        start_iteration = job_config.pop('start_iteration', 0)
        if iteration >= start_iteration:
            # load parameters, create run directory, and save user_cfg file
            job_config = OmegaConf.merge(general, job_config)
            cfg = read_user_config(job_config, config_name='train.yaml')
            if "defaults" in job_config:
                cfg.defaults = job_config.defaults   # here cfg has no defaults list
            run_path = Path(train_path) / f'iter_{iteration}' / name
            run_path.mkdir(parents=True, exist_ok=True)
            cfg.run_path = str(run_path.resolve())
            model_paths.append(cfg.run_path + '/model_path')
            # parse node resources
            job_resources = cfg.pop('resources')

            # TODO: load old model
            if iteration > start_iteration:
                load_model = cfg.pop('load_model', True)           # load_specific model
                if isinstance(load_model, bool):
                    if load_model:
                        model_path = Path(train_path) / f'iter_{iteration-1}' / name / 'model_path'
                        model_path = str(model_path.resolve())
                    else:
                        model_path = None
                elif load_model is None:
                    model_path = None
                elif isinstance(load_model, str):
                    model_path = load_model
                else:
                    raise ValueError("Invalid value for load_model!")
                
                cfg.model_path = model_path

            # save config file
            OmegaConf.save(cfg, run_path / 'train.yaml', resolve=False)

            tasks.append(run(
                shell='curator-train',
                deps=deps,
                args=arguments,
                folder=run_path,
                name='train',
                **job_resources,
            ))
    return tasks, model_paths

def simulate(
    deps: list[Task],
    model_path: list[str],
    config: DictConfig,
    iteration: Optional[int] = 0,
) -> Dict[str, Task]:
    ''' 
    Runs a simulate task for each model in the ensemble.
    This is done through the three steps for each model:
        - Create a new directory 
        - Save user_cfg file
        - Run task
    '''
    tasks = {}
    pool_path = {}                       # collect pool data set for active learning selection
    arguments = ['cfg=simulate.yaml']
    config = config.copy()

    # get general keys
    general = config.pop('general')

    # run multiple simulations
    for name, job_config in config.items():
        start_iteration = job_config.pop('start_iteration', 0)
        if iteration >= start_iteration:
            # load parameters, create run directory, and save user_cfg file
            job_config = OmegaConf.merge(general, job_config)
            cfg = read_user_config(job_config, config_name='simulate.yaml')
            if "defaults" in job_config:
                cfg.defaults = job_config.defaults   # here cfg has no defaults list
            run_path = Path(simulate_path) / f'iter_{iteration}' / name
            run_path.mkdir(parents=True, exist_ok=True)
            cfg.run_path = str(run_path.resolve())

            # parse node resources
            job_resources = cfg.pop('resources')

            # TODO: load old model, init_traj, load compiled model
            cfg.model_path = model_path
            pool_output, uncertain_output, restart_source = _simulation_outputs(cfg)
            # load user specified arguments: read_traj, image_index
            if iteration > start_iteration:
                default_restart = _resolve_iteration_reference(restart_source, iteration)
                init_traj = cfg.simulator.pop('read_traj', default_restart)   # use restart_source from last iteration if no override is specified
                start_index = cfg.simulator.pop('image_index', -1)  # use last image if not specified
                cfg.simulator.init_traj = init_traj
                cfg.simulator.start_index = start_index

            pool_path[name] = _as_path_list(pool_output)
            pool_path[name].extend(_as_path_list(uncertain_output))

            OmegaConf.save(cfg, run_path / 'simulate.yaml', resolve=False)

            tasks[name] = run(
                shell='curator-simulate',
                deps=deps,
                args=arguments,
                folder=run_path,
                name=name,
                **job_resources,
            )
    return tasks, pool_path

def select(
    deps: Dict[str, Task],
    model_path: list[str],
    pool_path: Dict[str, list[str]],
    config: DictConfig,
    iteration: Optional[int] = 0,
) -> Tuple[Dict[str, str], Dict[str, Task]]:
    ''' 
    Runs a select task for each model in the ensemble.
    This is done through the three steps for each model:
        - Create a new directory 
        - Save user_cfg file
        - Run task
    '''
    tasks = {}
    al_info = {}
    arguments = ['cfg=select.yaml']
    config = config.copy()

    # get general keys
    general = config.pop('general')

    # selection for multiple systems
    for name, job_config in config.items():
        start_iteration = job_config.pop('start_iteration', 0)
        if iteration >= start_iteration:
            # load parameters, create run directory, and save user_cfg file
            job_config = OmegaConf.merge(general, job_config)
            cfg = read_user_config(job_config, config_name='select.yaml')
            if "defaults" in job_config:
                cfg.defaults = job_config.defaults   # here cfg has no defaults list
            run_path = Path(select_path) / f'iter_{iteration}' / name
            run_path.mkdir(parents=True, exist_ok=True)
            cfg.run_path = str(run_path.resolve())

            # parse node resources
            job_resources = cfg.pop('resources')

            # TODO: load old model and get pool_set and al_info
            cfg.model_path = model_path
            cfg.pool_set = pool_path[name]
            al_info[name] = cfg.run_path + '/selected.json'

            OmegaConf.save(cfg, run_path / 'select.yaml', resolve=False)

            tasks[name] = run(
                shell='curator-select',
                deps=[deps[name]],
                args=arguments,
                folder=run_path,
                name=name,
                **job_resources,
            )
    return tasks, al_info

def label(
    deps: Dict[str, Task],
    pool_path: Dict[str, list],
    al_info: Dict[str, str],
    config: DictConfig,
    iteration: Optional[int] = 0,
) -> list[Task]:
    ''' 
    Runs a label task for each model in the ensemble.
    This is done through the three steps for each model:
        - Create a new directory 
        - Save user_cfg file
        - Run task
    '''
    tasks = []
    arguments = ['cfg=label.yaml']
    config = config.copy()

    # get general keys
    general = config.pop('general')

    # selection for multiple systems
    for name, job_config in config.items():
        start_iteration = job_config.pop('start_iteration', 0)
        if iteration >= start_iteration:
            # load parameters, create run directory, and save user_cfg file
            job_config = OmegaConf.merge(general, job_config)
            cfg = read_user_config(job_config, config_name='label.yaml')
            if "defaults" in job_config:
                cfg.defaults = job_config.defaults   # here cfg has no defaults list

            # parse node resources
            job_resources = cfg.pop('resources')

            # TODO: get atoms that need to be labelled, possibly overall datapath in training
            cfg.pool_set = pool_path[name]
            cfg.al_info = al_info[name]

            # split jobs if needed
            if cfg.split_jobs is not None:
                for i in range(cfg.split_jobs):
                    run_path = Path(label_path) / f'iter_{iteration}' / name / f'{i}'
                    run_path.mkdir(parents=True, exist_ok=True)
                    cfg.job_order = i
                    cfg.run_path = str(run_path.resolve())
                    OmegaConf.save(cfg, run_path / 'label.yaml', resolve=False)
                    tasks.append(run(
                        shell='curator-label',
                        deps=[deps[name]],
                        args=arguments,
                        folder=run_path,
                        name=name,
                        **job_resources,
                    ))
            else:
                run_path = Path(label_path) / f'iter_{iteration}' / name
                run_path.mkdir(parents=True, exist_ok=True)
                cfg.run_path = str(run_path.resolve())
                OmegaConf.save(cfg, run_path / 'label.yaml', resolve=False)
                tasks.append(run(
                    shell='curator-label',
                    deps=[deps[name]],
                    args=arguments,
                    folder=run_path,
                    name=name,
                    **job_resources,
                ))
    return tasks


def workflow(cfg='user_cfg.yaml'):
    cfg = OmegaConf.load(cfg)
    resolve_paths(cfg, base_dir=cfg.get('run_path', '.'))
    label_tasks = []

    for iteration in range(10):
        train_tasks, model_path = train(deps=label_tasks, config=cfg.train, iteration=iteration)

        simulate_tasks, pool_path = simulate(deps=train_tasks, model_path=model_path, config=cfg.simulate, iteration=iteration)

        select_tasks, al_info = select(deps=simulate_tasks, model_path=model_path, pool_path=pool_path, config=cfg.select, iteration=iteration)

        label_tasks = label(deps=select_tasks, pool_path=pool_path, al_info=al_info, config=cfg.label, iteration=iteration)
