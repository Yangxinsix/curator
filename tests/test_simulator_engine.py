import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from curator.simulate.core.simulator import Simulator
from curator.simulate.engines.ase_md import MDEngine
from curator.simulate.engines.optimizer import OptimizerEngine


class DummyCalc(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, force_value: float = 0.1):
        super().__init__()
        self.force_value = float(force_value)
        self.results = {}

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if atoms is not None:
            self.atoms = atoms.copy()
        n = len(self.atoms)
        self.results = {
            "energy": 1.0,
            "forces": np.full((n, 3), self.force_value, dtype=float),
            "stress": np.zeros(6, dtype=float),
        }


class DummyDynamics:
    def __init__(self, atoms, **_):
        self.atoms = atoms
        self._hooks = []

    def attach(self, fn, interval: int = 1):
        self._hooks.append((fn, interval))

    def run(self, steps: int):
        for step in range(1, steps + 1):
            for fn, interval in self._hooks:
                if step % interval == 0:
                    fn()


class DummyOptimizer:
    def __init__(self, atoms, **_):
        self.atoms = atoms
        self._hooks = []

    def attach(self, fn, interval: int = 1):
        self._hooks.append((fn, interval))

    def run(self, fmax: float = 0.02, steps: int | None = None):
        n = steps or 1
        for step in range(1, n + 1):
            for fn, interval in self._hooks:
                if step % interval == 0:
                    fn()


class TestSimulatorEngines(unittest.TestCase):
    def test_simulator_md_engine(self):
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]])
        atoms.calc = DummyCalc(force_value=0.0)
        engine = MDEngine(DummyDynamics)
        events = {"start": 0, "setup": 0, "end": 0, "steps": 0}

        def on_sim_start(ctx):
            events["start"] += 1

        def on_engine_setup(ctx):
            events["setup"] += 1

        def on_step(ctx):
            events["steps"] += 1

        def on_sim_end(ctx):
            events["end"] += 1

        cb = SimpleNamespace(
            on_sim_start=on_sim_start,
            on_engine_setup=on_engine_setup,
            on_step=on_step,
            on_sim_end=on_sim_end,
            interval=1,
        )
        sim = Simulator(init_traj=atoms, engine=engine, callbacks=[cb])
        sim.run(steps=2)
        self.assertEqual(events["start"], 1)
        self.assertEqual(events["setup"], 1)
        self.assertEqual(events["end"], 1)
        self.assertEqual(events["steps"], 2)

    def test_simulator_optimizer_engine(self):
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]])
        atoms.calc = DummyCalc(force_value=0.1)
        engine = OptimizerEngine(DummyOptimizer)
        events = {"start": 0, "setup": 0, "end": 0, "steps": 0}

        def on_sim_start(ctx):
            events["start"] += 1

        def on_engine_setup(ctx):
            events["setup"] += 1

        def on_step(ctx):
            events["steps"] += 1

        def on_sim_end(ctx):
            events["end"] += 1

        cb = SimpleNamespace(
            on_sim_start=on_sim_start,
            on_engine_setup=on_engine_setup,
            on_step=on_step,
            on_sim_end=on_sim_end,
            interval=1,
        )
        sim = Simulator(init_traj=atoms, engine=engine, callbacks=[cb])
        sim.run(fmax=0.05, steps=1)
        self.assertEqual(events["start"], 1)
        self.assertEqual(events["setup"], 1)
        self.assertEqual(events["end"], 1)
        self.assertGreaterEqual(events["steps"], 1)


if __name__ == "__main__":
    unittest.main()
