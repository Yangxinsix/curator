from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from curator.data import properties
from .callbacks import CalculatorAssign, ThermoWithUncertainty, TrajectoryWriter
from .callbacks.thermo import MDThermoLogger
from .core.simulator import Simulator
from .core.context import load_atoms_or_traj
from .engines.ase_md import MDEngine
from .engines.neb import NEBEngine
from .uncertainty import MahalanobisUncertainty


class MDSimulator:
    """Hydra-friendly wrapper that wires engine + callbacks for ASE MD runs."""

    def __init__(
        self,
        init_traj: str,
        *,
        calculator: Any,
        dynamics: Any,
        logger: Optional[Any] = None,
        uncertainty: Optional[Any] = None,
        variables: Optional[Sequence[str]] = None,
        monitor: str = properties.maha_dist,
        out_traj: str = "./MD.traj",
        start_index: int = -1,
        print_step: int = 1,
        dump_step: int = 100,
        max_steps: int = 1000,
        initialize_velocities: bool = True,
        temperature: float = 298.15,
        logger_instance: Optional[logging.Logger] = None,
        **_: Any,
    ):
        self.init_traj = init_traj
        self.out_traj = out_traj
        self.start_index = start_index
        self.print_step = max(1, int(print_step))
        self.dump_step = max(1, int(dump_step))
        self.max_steps = max(1, int(max_steps))
        self.initialize_velocities = initialize_velocities
        self.temperature = temperature
        self.monitor = monitor
        self.variables = variables
        self._logger = logger_instance or logging.getLogger("Simulator")

        self._calculator_cfg = calculator
        self._dynamics_cfg = dynamics
        self._logger_cfg = logger
        self._uncertainty_cfg = uncertainty

    def _instantiate(self, maybe_cfg: Any, **kwargs):
        if isinstance(maybe_cfg, DictConfig):
            return instantiate(maybe_cfg, **kwargs)
        return maybe_cfg

    def _build_uncertainty(self, calculator):
        backend_cfg = self._uncertainty_cfg
        if isinstance(backend_cfg, MahalanobisUncertainty):
            if backend_cfg.calc is None:
                backend_cfg.calc = calculator
            return backend_cfg

        backend = self._instantiate(backend_cfg, calculator=calculator)
        if isinstance(backend, MahalanobisUncertainty) and backend.calc is None:
            backend.calc = calculator
        return backend

    def _build_logger(self, backend):
        if isinstance(self._logger_cfg, ThermoWithUncertainty):
            logger = self._logger_cfg
            logger.interval = self.print_step
            if self.variables is not None:
                logger.variables = list(self.variables)
            if backend is not None:
                logger._unc_backend = backend
                include_keys = getattr(backend, "uncertainty_keys", ()) or (self.monitor,)
                logger._include = tuple(include_keys)
                for k in logger._include:
                    if k not in logger.variables:
                        logger.variables.append(k)
            return logger

        if self._logger_cfg is not None:
            return self._instantiate(
                self._logger_cfg,
                uncertainty_backend=backend,
                monitor=self.monitor,
                interval=self.print_step,
                variables=self.variables,
            )
        return ThermoWithUncertainty(
            variables=self.variables,
            interval=self.print_step,
            logger=self._logger,
            uncertainty_backend=backend,
            monitor=self.monitor,
        )

    def _build_callbacks(self, calculator, backend) -> List[Any]:
        callbacks: List[Any] = [
            CalculatorAssign(calculator=calculator, warmup=True, require_forces=True),
            self._build_logger(backend),
        ]
        if self.out_traj:
            callbacks.append(TrajectoryWriter(self.out_traj, interval=self.dump_step, mode="w"))
        return callbacks

    def run(self):
        atoms = load_atoms_or_traj(self.init_traj, self.start_index)
        if self.initialize_velocities:
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)

        calc = self._instantiate(self._calculator_cfg)
        backend = self._build_uncertainty(calc)
        if self._logger:
            self._logger.info(f"Uncertainty backend: {backend}")
        callbacks = self._build_callbacks(calc, backend)

        dyn = self._instantiate(self._dynamics_cfg)
        engine = MDEngine(dynamics_cls=dyn)

        sim = Simulator(atoms, engine, callbacks=callbacks, start_index=None, logger=self._logger)
        sim.run(steps=self.max_steps)

class NEBSimulator:
    """Hydra-friendly wrapper to run ASE NEB optimizations."""

    def __init__(
        self,
        *,
        initial_image: str,
        final_image: str,
        num_images: int = 5,
        calculator: Any,
        optimizer: Any,
        interpolator: Optional[Any] = None,
        neb_kwargs: Optional[DictConfig] = None,
        opt_kwargs: Optional[DictConfig] = None,
        fmax: float = 0.05,
        max_steps: int = 200,
        print_step: int = 1,
        start_index: int = 0,
        logger: Optional[Any] = None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        self.initial_image = initial_image
        self.final_image = final_image
        self.num_images = max(1, int(num_images))
        self.fmax = fmax
        self.max_steps = max(1, int(max_steps))
        self.print_step = max(1, int(print_step))
        self.start_index = start_index

        self._calculator_cfg = calculator
        self._optimizer_cfg = optimizer
        self._interpolator_cfg = interpolator
        self._neb_kwargs_cfg = neb_kwargs or {}
        self._opt_kwargs_cfg = opt_kwargs or {}
        self._logger_cfg = logger
        self._logger = logger_instance or logging.getLogger("Simulator")

    def _instantiate(self, maybe_cfg: Any, **kwargs):
        if isinstance(maybe_cfg, DictConfig):
            return instantiate(maybe_cfg, **kwargs)
        return maybe_cfg

    def _build_images(self) -> List[Atoms]:
        if self.initial_image is None or self.final_image is None:
            raise ValueError("Both initial_image and final_image must be provided for NEB runs.")

        first = load_atoms_or_traj(self.initial_image, self.start_index)
        last = load_atoms_or_traj(self.final_image, self.start_index)
        images = [first]
        for _ in range(self.num_images):
            images.append(first.copy())
        images.append(last)
        return images

    def _build_logger(self) -> MDThermoLogger:
        if self._logger_cfg is not None:
            return self._instantiate(self._logger_cfg)
        return MDThermoLogger(variables=["step", "epot"], interval=self.print_step, logger=self._logger)

    def _build_callbacks(self, calculator) -> List[Any]:
        return [
            CalculatorAssign(calculator=calculator, warmup=True, require_forces=True, apply_to_neb_images=True),
            self._build_logger(),
        ]

    def run(self):
        images = self._build_images()

        calc = self._instantiate(self._calculator_cfg)
        optimizer_cls = self._instantiate(self._optimizer_cfg)
        interpolator = self._instantiate(self._interpolator_cfg) if self._interpolator_cfg is not None else None

        if not callable(optimizer_cls):
            raise TypeError("optimizer must be a class or callable that creates an ASE optimizer")

        neb_kwargs = OmegaConf.to_container(self._neb_kwargs_cfg, resolve=True) if isinstance(self._neb_kwargs_cfg, DictConfig) else dict(self._neb_kwargs_cfg)
        opt_kwargs = OmegaConf.to_container(self._opt_kwargs_cfg, resolve=True) if isinstance(self._opt_kwargs_cfg, DictConfig) else dict(self._opt_kwargs_cfg)

        engine = NEBEngine(
            images=images,
            optimizer_cls=optimizer_cls,
            interpolator=interpolator,
            neb_kwargs=neb_kwargs,
            opt_kwargs=opt_kwargs,
        )

        callbacks = self._build_callbacks(calc)
        sim = Simulator(images[0], engine, callbacks=callbacks, start_index=None, logger=self._logger)
        sim.run(fmax=self.fmax, steps=self.max_steps)

