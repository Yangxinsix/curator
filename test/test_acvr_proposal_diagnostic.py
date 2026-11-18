"""Bounded runner uses the same functional and never consumes test examples."""

import json
from types import SimpleNamespace

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest
import torch

from scripts import acvr_proposal_diagnostic as diagnostic


@pytest.fixture(autouse=True)
def restore_dtype():
    previous = torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    yield
    torch.set_default_dtype(previous)
    torch.use_deterministic_algorithms(deterministic)


def _arguments(output, communication="pointwise"):
    arguments = ["--communication", communication, "--output", str(output), "--steps", "2",
                 "--train-size", "1", "--val-size", "1", "--batch-size", "1",
                 "--hidden-features", "4", "--layers", "1", "--heads", "1", "--jacobian-geometries", "1"]
    for override in ("data.num_train=1", "data.num_val=1", "data.num_test=1", "data.plane_side=2",
                     "model.representation.num_features=4", "model.representation.num_interactions=1",
                     "model.output_modules.0.proposal.hidden_features=4", "model.output_modules.0.proposal.num_heads=1"):
        arguments.extend(("--override", override))
    return arguments


def test_fixed_teacher_smoke_matches_independent_oracle_and_skips_test(tmp_path, monkeypatch):
    original = diagnostic.PeriodicChargeOracleDataset
    splits = []

    def dataset(*args, **kwargs):
        splits.append(kwargs["split"])
        assert kwargs["split"] != "test"
        return original(*args, **kwargs)

    monkeypatch.setattr(diagnostic, "PeriodicChargeOracleDataset", dataset)
    output = tmp_path / "diagnostic"
    result = diagnostic.main(_arguments(output))
    assert result["steps_completed"] == 2 and result["physical_parameters_unchanged"]
    assert result["deterministic_algorithms"] is True
    assert result["final"]["reference_failures"] == 0
    assert result["final"]["jacobian_conditions"] == 6
    assert result["final"]["paired_left_response"]["pairs_successful"] == 2
    assert result["final"]["jacobian_by_location"]["right_zero_sum"]["reference_norm"] > 1e-12
    assert result["final"]["constraint_max"] < 1e-10
    assert splits == ["train", "val"]
    reference = original(split="val", num_train=1, num_val=1, num_test=1, plane_side=2)
    labels = {condition.condition_uid: condition.labels for condition in reference[0].variational.conditions}
    rows = torch.load(output / "final_states.pt", weights_only=False)
    for row in rows:
        torch.testing.assert_close(row["reference_state"], labels[row["condition_uid"]]["variable_value"].flatten(), atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(row["reference_energy"], labels[row["condition_uid"]]["condition_energy"].reshape(()), atol=1e-9, rtol=1e-8)
        centered = row["zero_sum_jacobians"]["reference"]
        torch.testing.assert_close(centered["right_zero_sum"].mean(1), torch.zeros(4, dtype=torch.float64), atol=1e-15, rtol=0)
        torch.testing.assert_close(centered["both_zero_sum"].mean(0), torch.zeros(4, dtype=torch.float64), atol=1e-15, rtol=0)
    assert json.loads((output / "data_manifest.json").read_text())["test_evaluated"] is False
    assert (output / "source_manifest.json").is_file()
    assert (output / "checkpoint.pt").is_file()
    migration = json.loads((output / "migration.json").read_text())
    assert migration["new_proposal_contract"]["state_scale"] == 0.05
    assert migration["new_proposal_contract"]["field_scale"] == 1.0


def test_loaded_checkpoint_replaces_only_proposal_and_records_reference_failures(tmp_path, monkeypatch):
    config = OmegaConf.load(diagnostic.ROOT / "curator/configs/acvr_periodic_charge.yaml")
    config.model.representation.num_features = 4
    config.model.representation.num_interactions = 1
    config.model.output_modules[0].proposal.hidden_features = 4
    config.model.output_modules[0].proposal.num_heads = 1
    model = instantiate(config.model, _convert_="all").double()
    checkpoint = tmp_path / "source.pt"
    torch.save({"model": model}, checkpoint)

    def failed_reference(*args, **kwargs):
        raise RuntimeError("deliberate reference failure")

    monkeypatch.setattr(diagnostic, "solve_functional_reference", failed_reference)
    output = tmp_path / "checkpoint_diagnostic"
    result = diagnostic.main(_arguments(output, "baseline") + ["--model-path", str(checkpoint)])
    assert result["steps_completed"] == 2 and result["physical_parameters_unchanged"]
    assert result["final"]["reference_failures"] == result["final"]["conditions"] == 6
    assert all("deliberate reference failure" in row["reference_status"]
               for row in result["final"]["reference_failures_detail"])
    migration = json.loads((output / "migration.json").read_text())
    assert migration["new_proposal"] == "VariationalStateHead"
    assert migration["functional"] == "checkpoint unchanged"
    assert len(migration["source_checkpoint_sha256"]) == 64
    excluded = {id(parameter) for parameter in model.output_modules[0].proposal.parameters()}
    assert migration["frozen_parameter_sha256"] == diagnostic.parameter_hash(model, excluded)


def test_budget_exhaustion_is_incomplete_and_keeps_every_reference_in_denominator(tmp_path):
    result = diagnostic.main(_arguments(tmp_path / "budget") + ["--time-limit", "1e-9"])
    assert result["status"] == "budget_exhausted"
    assert result["steps_completed"] == 0
    assert result["final"]["reference_failures"] == result["final"]["conditions"] == 6
    assert all(row["reference_status"] == "budget_exhausted" for row in result["final"]["reference_failures_detail"])


class _SubsetDataset:
    def __init__(self, indices):
        self.indices = indices
        self.metadata = {"source_hashes": {"data": "same-source"}}

    def __len__(self):
        return len(self.indices)


def test_replays_original_subset_instead_of_resampling_with_proposal_seed(tmp_path):
    path = tmp_path / "old_manifest.json"
    source = {"seed": 20260909, "dataset_subset_positions": {"train": [2, 0], "val": [1]},
              "source_indices": {"train": [30, 10], "val": [50]}, "source_hashes": {"data": "same-source"}}
    path.write_text(json.dumps(source))
    datasets = {"train": _SubsetDataset([10, 20, 30]), "val": _SubsetDataset([40, 50])}
    args = SimpleNamespace(subset_manifest=str(path), train_size=2, val_size=1, seed=17)
    generator = torch.Generator().manual_seed(17)
    before = generator.get_state().clone()
    indices, metadata = diagnostic.select_subsets(datasets, args, None, generator)
    assert indices == source["dataset_subset_positions"]
    assert metadata["selection_seed"] == 20260909
    torch.testing.assert_close(before, generator.get_state(), rtol=0, atol=0)
    datasets["train"].indices[2] = 999
    with pytest.raises(ValueError, match="source indices differ"):
        diagnostic.select_subsets(datasets, args, None, generator)
