"""Coverage, physical labels, archive integrity, and unchanged legacy data."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import runpy

import pytest
import torch

from curator.data.periodic_charge_coverage import (
    PreparedChargeDataModule, expand_field_conditions, field_directions, teacher_matrices,
)
from curator.data.periodic_charge_oracle import PeriodicChargeOracleDataset, periodic_charge_reference


@pytest.fixture(scope="module")
def source():
    return PeriodicChargeOracleDataset(num_train=2, num_val=1, num_test=1, plane_side=2)


@pytest.fixture(scope="module")
def expanded(source):
    return expand_field_conditions(source[0], source.metadata)


def test_full_zero_sum_coverage_at_each_side_and_new_probe_directions():
    first = field_directions(16, 20260914, "geometry-a")
    same = field_directions(16, 20260914, "geometry-a")
    probe = field_directions(16, 20260915, "geometry-a")
    other_geometry = field_directions(16, 20260914, "geometry-b")
    for a, b in zip(first, same):
        torch.testing.assert_close(a[-1], b[-1], rtol=0, atol=0)
        assert float(a[-1].abs().max()) == 1
        torch.testing.assert_close(a[-1].reshape(2, 16).sum(1), torch.zeros(2, dtype=torch.float64),
                                   atol=1e-12, rtol=0)
    for group, start in (("left", 0), ("right", 16)):
        values = torch.stack([row[-1][start:start + 16] for row in first
                              if row[0] == group and row[1] == "dense_zero_sum"])
        assert torch.linalg.matrix_rank(values, atol=1e-10) == 15
    assert not torch.allclose(first[0][-1], probe[0][-1])
    assert not torch.allclose(first[0][-1], other_geometry[0][-1])


def test_legacy_geometry_inputs_and_labels_remain_exactly_preserved(source, expanded):
    assert expanded.atoms is source[0].atoms
    assert expanded.variational.geometry_hash == source[0].variational.geometry_hash
    for before, after in zip(source[0].variational.conditions, expanded.variational.conditions[:6]):
        assert before is after
    assert len(expanded.variational.conditions) == 150
    assert len({c.condition_uid for c in expanded.variational.conditions}) == 150
    with pytest.raises(ValueError, match="distinct finite positive"):
        expand_field_conditions(source[0], source.metadata, amplitudes=(0.1, float("nan")))


def test_independent_audit_checks_all_labels_and_rejects_corruption(source, expanded):
    audit = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                              "scripts/acvr_prepare_field_data.py"))["audit_sample"]
    row, response = audit(expanded, source.metadata)
    assert max(row["maximum_errors"].values()) < 1e-11
    assert len(row["direction_coverage"]) == 12  # two sides, two boundaries, three amplitudes
    assert all(value["rank"] == 3 for value in row["direction_coverage"].values())
    assert all(min(values) > 0 for values in response.values())
    first = expanded.variational.conditions[6]
    bad = replace(first, labels={**first.labels, "variable_value": first.labels["variable_value"] + 1e-3})
    item = replace(expanded, variational=replace(expanded.variational,
        conditions=expanded.variational.conditions[:6] + (bad,) + expanded.variational.conditions[7:]))
    with pytest.raises(ValueError, match="data audit failed"):
        audit(item, source.metadata)


def test_new_label_derivatives_match_independent_energy_differences(source, expanded):
    hessian, chi, constant = teacher_matrices(expanded, source.metadata)
    n, side = 8, 4
    groups = torch.zeros((2, n), dtype=torch.float64)
    groups[0, :side], groups[1, side:] = 1, 1
    for kind in ("conq", "conp"):
        condition = next(c for c in expanded.variational.conditions
                         if c.provenance.get("condition_design") and
                         c.provenance["boundary_kind"] == kind and c.provenance["field_peak_V"] == 1.0)
        constraint = groups if kind == "conq" else torch.ones((1, n), dtype=torch.float64)
        target = condition.inputs["constraint_target"]
        drive = condition.inputs["local_external_field"].flatten().clone()
        if kind == "conp":
            drive += groups.T @ condition.inputs["control_value"]
        for index in range(target.numel()):
            step = torch.zeros_like(target)
            step[index] = 1e-5
            energies = [periodic_charge_reference(hessian, chi, constant, constraint,
                                                  target + sign * step, drive)[2] for sign in (-1, 1)]
            torch.testing.assert_close((energies[1] - energies[0]) / 2e-5,
                condition.labels["constraint_derivative"][index], atol=1e-9, rtol=1e-6)
        if kind == "conp":
            for index in range(2):
                energies = [periodic_charge_reference(hessian, chi, constant, constraint,
                            target, drive + sign * 1e-5 * groups[index])[2] for sign in (-1, 1)]
                torch.testing.assert_close((energies[1] - energies[0]) / 2e-5,
                    condition.labels["control_derivative"][index], atol=1e-9, rtol=1e-6)


def test_prepared_loader_views_train_only_scales_hashes_and_resume(tmp_path):
    archives = {}
    for split in ("train", "val", "test"):
        original = PeriodicChargeOracleDataset(split=split, num_train=2, num_val=1, num_test=1,
                                                plane_side=2, seed=23)
        samples = [expand_field_conditions(sample, original.metadata) for sample in original]
        path = tmp_path / f"{split}.pt"
        torch.save({"split": split, "geometry_indices": original.indices, "samples": samples}, path)
        archives[split] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    metadata = {**original.metadata, "field_design": {"amplitudes_V": [0.1, 0.3, 1.0]}}
    (tmp_path / "manifest.json").write_text(json.dumps({"archives": archives, "dataset_metadata": metadata}))
    data = PreparedChargeDataModule(tmp_path, view="amplitude", amplitude=0.3, batch_size=1)
    data.setup()
    assert len(data.dataset[0].variational.conditions) == 52  # 48 perturbations, four original bases
    values = torch.cat([c.labels["condition_energy"] for s in data.dataset for c in s.variational.conditions])
    assert data.metadata["supervised_scales"]["condition_energy"] == float(values.square().mean().sqrt())
    state = data.state_dict()
    next(iter(data.val_dataloader()))
    assert torch.equal(data.generator.get_state(), state["generator_state"])
    restored = PreparedChargeDataModule(tmp_path, view="amplitude", amplitude=0.3, batch_size=1)
    restored.load_state_dict(state)
    assert next(iter(data.train_dataloader()))["meta"] == next(iter(restored.train_dataloader()))["meta"]
    old = PreparedChargeDataModule(tmp_path, view="legacy")
    old.setup()
    assert len(old.dataset[0].variational.conditions) == 6
    (tmp_path / "test.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="archive hash differs"):
        PreparedChargeDataModule(tmp_path).setup()
