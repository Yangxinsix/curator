import copy
import os
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read

from curator.data import properties
from curator.simulate.core.context import SimContext
from curator.simulate.uncertainty import (
    AutoUncertainty,
    EnsembleUncertainty,
    MahalanobisUncertainty,
    MCDropoutUncertainty,
)
from curator.simulate.callbacks.uncertainty_monitor import UncertaintyMonitor
from curator.simulate.callbacks.thermo_uncertainty import ThermoWithUncertainty
from curator.simulate.core.calculator import MLCalculator
from curator.utils import load_model
from curator.data import AseDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(os.environ.get("CURATOR_TEST_DATASET", PROJECT_ROOT / "example/LiFePO4.traj"))
MODEL_PATH = Path(os.environ.get("CURATOR_TEST_CKPT", PROJECT_ROOT / "example/best_model.ckpt"))


class DummyCalc(Calculator):
    implemented_properties = ["energy", "forces", "stress", properties.f_sd, properties.f_var]

    def __init__(self):
        super().__init__()
        self.results = {}

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if atoms is not None:
            self.atoms = atoms.copy()
        n = len(self.atoms)
        self.results = {
            "energy": 1.0,
            "forces": np.zeros((n, 3), dtype=float),
            "stress": np.zeros(6, dtype=float),
            properties.f_sd: 0.2,
            properties.f_var: 0.04,
        }


class TestUncertainty(unittest.TestCase):
    def test_ensemble_uncertainty(self):
        if not MODEL_PATH.exists():
            self.skipTest(f"Model checkpoint not found at {MODEL_PATH}")
        atoms = read(DATASET_PATH)
        model = load_model(MODEL_PATH, device=torch.device("cpu"), load_compiled=False)
        model.eval()
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        atoms.calc = MLCalculator(model=[model, model_copy], device="cpu")
        unc = EnsembleUncertainty(uncertainty_keys=(properties.f_sd, properties.f_var))
        out = unc(atoms)
        self.assertIn(properties.f_sd, out)
        self.assertIn(properties.f_var, out)
        self.assertIn("is_warning", out)
        self.assertIn("is_outlier", out)

    def test_mc_dropout_uncertainty(self):
        if not DATASET_PATH.exists():
            self.skipTest(f"Dataset not found at {DATASET_PATH}")
        atoms = read(str(DATASET_PATH), index=0)
        predictor = lambda _: 0.5
        unc = MCDropoutUncertainty(predictor=predictor, n_samples=5)
        out = unc(atoms)
        self.assertIn(properties.f_sd, out)
        self.assertIn(properties.f_var, out)

    def test_auto_uncertainty_no_backend(self):
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        unc = AutoUncertainty(calculator=None, dataset=None)
        out = unc(atoms)
        self.assertEqual(out, {})

    def test_mahalanobis_uncertainty(self):
        if not DATASET_PATH.exists():
            self.skipTest(f"Dataset not found at {DATASET_PATH}")
        if not MODEL_PATH.exists():
            self.skipTest(f"Model checkpoint not found at {MODEL_PATH}")
        model = load_model(MODEL_PATH, device=torch.device("cpu"), load_compiled=False)
        model.eval()
        atoms = read(str(DATASET_PATH), index=0)
        atoms.calc = MLCalculator(model=model, device="cpu")
        dataset = AseDataset(str(DATASET_PATH))
        subset = torch.utils.data.Subset(dataset, [0, 1])
        unc = MahalanobisUncertainty(
            calculator=atoms.calc,
            dataset=subset,
            kernel="full-g",
            max_structures=1,
            n_random_features=4,
        )
        out = unc(atoms)
        self.assertIn(properties.maha_dist, out)


class TestCallbacks(unittest.TestCase):
    def test_uncertainty_monitor_flags(self):
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        ctx = SimContext(atoms=atoms, step=0)

        def backend(_):
            return {"sigma": 0.6, "is_outlier": True}

        cb = UncertaintyMonitor(backend=backend, monitor="sigma", low=0.1, high=0.5)
        cb.on_sim_start(ctx)
        cb.on_step(ctx)
        self.assertIn("uncertainty", ctx.state)
        self.assertIn("early_stop_reason", ctx.state)

    def test_thermo_with_uncertainty_backend(self):
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        atoms.calc = DummyCalc()
        ctx = SimContext(atoms=atoms, step=0)

        def backend(_):
            return {properties.maha_dist: 0.2}

        backend.uncertainty_keys = (properties.maha_dist,)
        backend.threshold_key = properties.maha_dist

        cb = ThermoWithUncertainty(
            variables="basic_energies",
            interval=1,
            monitor=properties.maha_dist,
            uncertainty_backend=backend,
            save_path=None,
            low=None,
            high=None,
        )
        cb.on_sim_start(ctx)
        cb.on_step(ctx)
        self.assertIn("uncertainty", ctx.state)
        self.assertIn(properties.maha_dist, cb.variables)


if __name__ == "__main__":
    unittest.main()
