"""The public entrypoint trains and resumes one joint GNN optimizer."""

import json
import os
from pathlib import Path
import subprocess
import sys

import torch


class MixedCheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3))
        self.register_buffer("statistics", torch.zeros(2, dtype=torch.float64))


def _run(tmp_path, name, overrides=(), *, evaluate=False):
    repository = Path(__file__).resolve().parents[1]
    entrypoint = "evaluate" if evaluate else "train"
    config_name = "acvr_joint_evaluate" if evaluate else "acvr_joint"
    axis = "datapath" if evaluate else "data"
    dimensions = [f"{axis}.num_train=4", f"{axis}.num_val=2", f"{axis}.num_test=2",
                  f"{axis}.num_conditions=2"]
    if not evaluate:
        dimensions += ["data.batch_size=2", "model.representation.num_features=4",
                       "model.representation.num_interactions=1", "model.representation.num_basis=4",
                       "model.output_modules.0.hidden_features=4"]
    command = [sys.executable, "-c",
               f"from curator.commands.hydra_entrypoints import {entrypoint}; {entrypoint}()",
               "--config-name", config_name, f"run_path={tmp_path / name}", *dimensions, *overrides]
    result = subprocess.run(command, cwd=repository, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=180,
                            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    (tmp_path / (name + ".log")).write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-15000:]
    if not evaluate:
        return torch.load(tmp_path / name / "model_path/last.ckpt", map_location="cpu", weights_only=False)


def _assert_same_state(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
        else:
            assert actual[name] == value


def test_batch_neighbor_list_handles_explicit_nonperiodic_zero_cell():
    from curator.data import BatchNeighborList, properties

    batch = {
        properties.positions: torch.tensor([[0., 0., 0.], [1., 0., 0.],
                                           [0., 0., 0.], [6., 0., 0.]]),
        properties.n_atoms: torch.tensor([2, 2]),
        properties.cell: torch.zeros(6, 3),
        properties.pbc: torch.zeros(2, 3, dtype=torch.bool),
    }
    result = BatchNeighborList(cutoff=2.0)(batch)
    assert result[properties.edge_idx].tolist() == [[0, 1], [1, 0]]
    torch.testing.assert_close(result[properties.edge_diff], torch.tensor([[1., 0., 0.], [-1., 0., 0.]]))


def test_state_dict_reconstruction_preserves_parameter_and_buffer_dtypes(tmp_path):
    from omegaconf import OmegaConf
    from curator.utils import load_model

    checkpoint = {
        "model_params": OmegaConf.create({"_target_": __name__ + ".MixedCheckpointModel"}),
        "state_dict": {
            "model.weight": torch.tensor([1.000000000000001, 2.123456789012345, 3.], dtype=torch.float64),
            "model.statistics": torch.tensor([0.2, 0.3], dtype=torch.float32),
        },
    }
    path = tmp_path / "mixed.ckpt"
    torch.save(checkpoint, path)
    default_dtype = torch.get_default_dtype()
    loaded = load_model(path, device="cpu")
    assert torch.get_default_dtype() == default_dtype
    assert loaded.weight.dtype == torch.float64
    assert loaded.statistics.dtype == torch.float32
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, checkpoint["state_dict"]["model." + name], rtol=0, atol=0)


def test_joint_cli_single_optimizer_checkpoint_rebuild_resume_and_evaluate(tmp_path):
    from curator.model.painn import Painn
    from curator.utils import load_model

    trained = _run(tmp_path, "joint", ["trainer.max_epochs=1"])
    assert trained["epoch"] == 0 and trained["global_step"] == 2
    assert len(trained["optimizer_states"]) == 1
    assert trained["optimizer_states"][0]["state"]
    assert isinstance(trained["model"].representation, Painn)
    assert all(parameter.requires_grad for parameter in trained["model"].parameters())
    metadata = trained["model"].output_modules[0].get_extra_state()["contract"]
    for key, value in {
        "training_mode": "joint", "energy_state": "predicted",
        "force_derivative": "full_proposal_path", "residual_gradient_policy": "proposal_only",
    }.items():
        assert metadata[key] == value

    state_only = dict(trained)
    state_only.pop("model")
    state_only["model_save_mode"] = "state_dict"
    state_path = tmp_path / "joint_state_dict.ckpt"
    torch.save(state_only, state_path)
    rebuilt = load_model(state_path, device="cpu")
    _assert_same_state(rebuilt.state_dict(), trained["model"].state_dict())
    checkpoint = tmp_path / "joint/model_path/last.ckpt"
    resumed = _run(tmp_path, "resumed", ["trainer.max_epochs=2",
                   f"model_path={{path: {checkpoint}, mode: resume}}"])
    continuous = _run(tmp_path, "continuous", ["trainer.max_epochs=2"])
    assert resumed["epoch"] == 1 and resumed["global_step"] == continuous["global_step"] == 4
    _assert_same_state(resumed["state_dict"], continuous["state_dict"])
    expected_optimizer = continuous["optimizer_states"][0]
    actual_optimizer = resumed["optimizer_states"][0]
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for index, state in expected_optimizer["state"].items():
        _assert_same_state(actual_optimizer["state"][index], state)
    _run(tmp_path, "evaluate", [f"model_path={checkpoint}"], evaluate=True)
    metrics = json.loads((tmp_path / "evaluate/metrics.json").read_text())
    assert metrics["metadata"]["energy_state"] == "predicted"
    assert metrics["comparison"]["metadata"]["energy_state"] == "converged"
    assert metrics["num_structures"] == 2 and metrics["num_conditions"] == 4


def test_joint_last_checkpoint_advances_when_best_does_not_improve(tmp_path):
    last = _run(tmp_path, "constant", ["trainer.max_epochs=2", "task.optimizer.lr=0.0"])
    assert last["epoch"] == 1 and last["global_step"] == 4
    best_paths = list((tmp_path / "constant/model_path").glob("best_model_*.ckpt"))
    assert len(best_paths) == 1
    best = torch.load(best_paths[0], map_location="cpu", weights_only=False)
    assert best["epoch"] == 0 and best["global_step"] == 2


def test_joint_resume_when_step_limit_lands_on_epoch_boundary(tmp_path):
    # Exercise both stopping limits. A finite max_steps changes Lightning's
    # restart path even when every batch of the checkpoint's epoch completed.
    dimensions = ["data.num_train=2", "data.batch_size=1"]
    first = _run(tmp_path, "step_limit_first", dimensions + ["trainer.max_epochs=1", "+trainer.max_steps=2"])
    assert first["epoch"] == 0 and first["global_step"] == 2
    checkpoint = tmp_path / "step_limit_first/model_path/last.ckpt"
    limits = ["trainer.max_epochs=11", "+trainer.max_steps=22"]
    resumed = _run(tmp_path, "step_limit_resumed", dimensions + limits + [
        f"model_path={{path: {checkpoint}, mode: resume}}"])
    continuous = _run(tmp_path, "step_limit_continuous", dimensions + limits)
    assert resumed["global_step"] == continuous["global_step"] == 22
    assert resumed["epoch"] == continuous["epoch"] == 10
    _assert_same_state(resumed["state_dict"], continuous["state_dict"])
    actual, expected = resumed["optimizer_states"][0], continuous["optimizer_states"][0]
    assert actual["param_groups"] == expected["param_groups"]
    for index, state in expected["state"].items():
        _assert_same_state(actual["state"][index], state)
    assert resumed["loops"]["fit_loop"] == continuous["loops"]["fit_loop"]
    _assert_same_state(resumed["VariationalOracleDataModule"], continuous["VariationalOracleDataModule"])
