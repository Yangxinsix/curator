import numpy as np
import torch

from curator.evaluate.evaluator import Evaluator


def test_evaluator_reports_per_atom_energy_and_throughput_metrics():
    evaluator = Evaluator(torch.nn.Identity())
    labels = {
        "num_structures": 2,
        "energy_true": [2.0, 8.0],
        "energy_pred": [3.0, 4.0],
        "forces_true": [],
        "forces_pred": [],
        "atomic_numbers": [],
        "n_atoms": [1, 4],
        "throughput": {
            "warmup_batches": 1,
            "timed_batches": 2,
            "timed_structures": 10,
            "seconds": 2.0,
        },
    }

    metrics = evaluator.calculate_metrics(labels)

    assert np.isclose(metrics["energy"]["mae_per_atom"], 1.0)
    assert np.isclose(metrics["energy"]["rmse_per_atom"], 1.0)
    assert np.isclose(metrics["throughput"]["samples_per_second"], 5.0)
    assert np.isclose(metrics["throughput"]["ns_per_day_at_1fs"], 0.432)
