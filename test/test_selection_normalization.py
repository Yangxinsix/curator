"""Selection must use the kernel represented by the cached coordinates."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from curator.commands.select import _legacy_variance_normalization
from curator.layer.feature.statistics import FeatureStatistics
from curator.select.active_learning import GeneralActiveLearning


RAW = torch.tensor([[[10.0, 0.0], [9.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)


def _stats(monkeypatch):
    stats = FeatureStatistics(models=[torch.nn.Linear(2, 1)], dataset=[], device="cpu")
    calculator = SimpleNamespace(kernels=[SimpleNamespace(spec=SimpleNamespace(mapping="identity"))])
    monkeypatch.setattr(stats, "_resolve_calculators", lambda: ([calculator], ["gnn-id"]))
    monkeypatch.setattr(stats, "_compute", lambda *args: None)
    monkeypatch.setattr(stats, "_load_features", lambda *args: {"gnn-id": RAW.clone()})
    return stats


def test_statistics_default_preserves_values_and_gram(monkeypatch):
    actual = _stats(monkeypatch).get_features()["gnn-id"]
    torch.testing.assert_close(actual, RAW, rtol=0, atol=0)
    torch.testing.assert_close(actual @ actual.transpose(-1, -2), RAW @ RAW.transpose(-1, -2))


def test_explicit_legacy_scaling_keeps_variance_formula(monkeypatch, caplog):
    actual = _stats(monkeypatch).get_features(normalize=True)["gnn-id"]
    expected = (RAW - RAW.mean(dim=1, keepdim=True)) / RAW.var(dim=1, keepdim=True)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, (RAW - RAW.mean(dim=1, keepdim=True)) / RAW.std(dim=1, keepdim=True))
    assert "legacy variance normalization" in caplog.text


def _selector(monkeypatch):
    monkeypatch.setattr(GeneralActiveLearning, "_build_calculators", lambda *args: [])
    al = GeneralActiveLearning(
        models=[torch.nn.Linear(2, 1)], selection="max_diag",
        feature_specs=[{"preset": "gnn-id", "pooling": "mean"}], device="cpu",
    )
    monkeypatch.setattr(al, "_read_trajectory", lambda source: list(range(3)))
    monkeypatch.setattr(al, "_make_dataset", lambda atoms: atoms)
    monkeypatch.setattr(al, "_filter_set", lambda dataset, label: (dataset, None))
    monkeypatch.setattr(al, "_stats", lambda *args, **kwargs: _stats(monkeypatch))
    return al


@pytest.mark.parametrize("feature_only", [False, True])
def test_selection_default_is_raw_and_records_policy(monkeypatch, tmp_path, feature_only):
    destination = tmp_path / "selected.json"
    selected = _selector(monkeypatch).select(
        "unused.traj", select_batch_size=1, save_json=destination,
        compute_features_only=feature_only,
    )
    assert selected == ([] if feature_only else [0])
    metadata = json.loads(destination.read_text())
    assert metadata["feature_normalization"] == "none"
    assert metadata["feature_store_normalization"] == "none"
    assert metadata["selection_protocol_version"] == 2


def test_selection_legacy_changes_geometry_only_when_requested(monkeypatch, tmp_path):
    destination = tmp_path / "selected.json"
    selected = _selector(monkeypatch).select(
        "unused.traj", select_batch_size=1, save_json=destination,
        normalize_features=True,
    )
    assert selected == [2]
    assert json.loads(destination.read_text())["feature_normalization"] == "legacy_variance"


def test_selection_records_effective_maxdet_ridge_and_equal_budget(monkeypatch, tmp_path):
    destination = tmp_path / "selected.json"
    al = _selector(monkeypatch)
    al.selection = "max_det_greedy"
    selected = al.select("unused.traj", select_batch_size=3, save_json=destination)
    assert len(set(selected)) == 3
    assert json.loads(destination.read_text())["selection_kwargs"]["regularization"] == 1e-6


def test_legacy_rejects_independent_pool_and_train_scaling(monkeypatch):
    with pytest.raises(ValueError, match="incompatible coordinates"):
        _selector(monkeypatch).select("pool.traj", train_set="initial.traj", normalize_features=True)


@pytest.mark.parametrize("previous", [{"selected": [1]}, {
    "selection_protocol_version": 2, "feature_normalization": "legacy_variance",
}])
def test_selection_does_not_overwrite_historical_or_different_geometry(monkeypatch, tmp_path, previous):
    destination = tmp_path / "selected.json"
    destination.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="new output path"):
        _selector(monkeypatch).select("unused.traj", select_batch_size=1, save_json=destination)
    assert json.loads(destination.read_text()) == previous


def test_cli_default_and_explicit_legacy_alias(caplog):
    config = OmegaConf.load(Path(__file__).parents[1] / "curator/configs/select.yaml")
    assert not _legacy_variance_normalization(config)
    assert not _legacy_variance_normalization(OmegaConf.create({}))
    assert _legacy_variance_normalization(OmegaConf.create({"legacy_variance_normalization": True}))
    assert _legacy_variance_normalization(OmegaConf.create({"export_normalized_features": True}))
    assert "affecting SELECTION" in caplog.text
