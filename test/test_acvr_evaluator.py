import copy
import io
import json
from types import SimpleNamespace

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from curator.data import properties
from curator.data.atoms_data import AtomsData
from curator.data.collate_atoms_data import collate_atoms_data
from curator.data.variational_data import (
    VariationalCondition,
    VariationalData,
    canonical_boundary_hash,
)
from curator.evaluate.variational import VariationalEvaluator
from curator.train import VariationalJointLoss


class EvaluationDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        positions = torch.tensor([[float(index + 1), 0.0, 0.0]])
        conditions = []
        for condition in range((1, 3, 2)[index]):
            boundary = {"condition": condition}
            labels = {"condition_forces": torch.zeros(1, 3)}
            if (index, condition) != (1, 1):
                labels["condition_energy"] = torch.zeros(1)
            conditions.append(VariationalCondition(
                condition_uid=f"{index}:{condition}",
                electron_count=None,
                electron_count_convention=None,
                net_charge=torch.tensor(float(condition)),
                boundary_spec=boundary,
                boundary_hash=canonical_boundary_hash(boundary),
                gauge={},
                labels=labels,
                label_units={"condition_energy": "eV", "condition_forces": "eV/Ang"},
            ))
        return AtomsData(
            atoms={properties.positions: positions, properties.Z: torch.tensor([1]),
                   properties.n_atoms: torch.tensor([1])},
            targets={},
            task="variational",
            weight=(1.0, 0.5, 2.0)[index],
            variational=VariationalData(
                schema_version="1", geometry_uid=f"geometry:{index}",
                geometry_hash=f"geometry:{index}", variable_spec={},
                group_member_ptr=torch.tensor([0]), group_atom_index=torch.tensor([], dtype=torch.long),
                group_names=(), group_roles=(), conditions=tuple(conditions),
            ),
        )


class DerivativeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficient = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.unit = "eV"

    def forward(self, batch):
        assert torch.is_grad_enabled()
        positions = batch[properties.positions]
        target = batch["variational"]
        assert positions.dtype == target["condition_weight"].dtype == self.coefficient.dtype
        assert target["labels"]["condition_forces"]["values"].dtype == self.coefficient.dtype
        geometry = target["condition_geometry_index"]
        geometry_energy = self.coefficient * positions.square().sum(dim=1)
        forces = -torch.autograd.grad(geometry_energy.sum(), positions)[0]
        energy = geometry_energy[geometry] + target["net_charge"]
        count = len(geometry)
        ptr = torch.arange(count + 1, device=positions.device)
        return {"variational_prediction": {"value_space": "physical", "labels": {
            "condition_energy": {"values": energy, "ptr": ptr,
                                 "shape": ((1,),) * count, "unit": self.unit},
            "condition_forces": {"values": forces[geometry].reshape(-1), "ptr": 3 * ptr,
                                 "shape": ((1, 3),) * count, "unit": "eV/Ang"},
        }}}


def evaluation_loss():
    return VariationalJointLoss(
        supervised_weights={"condition_energy": 2.0, "condition_forces": 1.0, "dipole": 1.0},
        supervised_scales={"condition_energy": 2.0, "condition_forces": 1.0, "dipole": 1.0},
    )


def test_evaluator_uses_global_geometry_weights_masks_and_force_derivatives(tmp_path):
    model = DerivativeModel()
    results = []
    for batch_size in (1, 2, 3):
        evaluator = VariationalEvaluator(
            model, evaluation_loss(), output_dir=tmp_path / str(batch_size), batch_size=batch_size,
        )
        with torch.inference_mode():
            metrics = evaluator.evaluate(EvaluationDataset())
        assert model.training
        assert metrics["num_structures"] == 3
        assert metrics["num_conditions"] == 6
        energy = metrics["terms"]["condition_energy"]
        # Geometry 2 misses the middle condition: squared energies average 26.
        # Geometry 3 averages 9**2 and 10**2.  Geometry weights are 1, .5, 2.
        expected_energy_mse = (1 + 0.5 * 26 + 2 * 90.5) / 3.5 / 4
        assert energy["normalized_mse"] == pytest.approx(expected_energy_mse)
        assert energy["rmse"] == pytest.approx(2 * expected_energy_mse ** 0.5)
        assert energy["coverage"] == pytest.approx(5 / 6)
        assert energy["n_geometries"] == 3
        expected_force_mse = (4 / 3 + 0.5 * 16 / 3 + 2 * 36 / 3) / 3.5
        assert metrics["terms"]["condition_forces"]["normalized_mse"] == pytest.approx(expected_force_mse)
        assert metrics["loss"] == pytest.approx(2 * expected_energy_mse + expected_force_mse)
        assert metrics["terms"]["dipole"]["status"] == "inactive"
        assert metrics["terms"]["dipole"]["rmse"] is None
        assert json.loads((tmp_path / str(batch_size) / "metrics.json").read_text()) == metrics
        results.append(metrics["loss"])
    assert results == pytest.approx([results[0]] * 3)

    artifact = torch.load(tmp_path / "2" / "results.pt", weights_only=False)
    numerator = denominator = 0.0
    for record in artifact["batches"]:
        prediction = record["prediction"]["variational_prediction"]
        assert not prediction["labels"]["condition_energy"]["values"].requires_grad
        assert record["batch"]["variational"]["condition_geometry_index"].device.type == "cpu"
        term = evaluation_loss()(record["prediction"], record["batch"])["terms"]["condition_energy"]
        numerator += float(term["numerator"])
        denominator += float(term["denominator"])
    assert numerator / denominator == pytest.approx(expected_energy_mse)


def test_evaluator_accepts_hydra_configuration_and_dataloader(tmp_path):
    config = OmegaConf.create({
        "_target_": "curator.evaluate.variational.VariationalEvaluator",
        "output_dir": str(tmp_path), "save_data": False,
        "loss_fn": {
            "_target_": "curator.train.VariationalJointLoss",
            "supervised_weights": {"condition_forces": 1.0},
            "supervised_scales": {"condition_forces": 1.0},
        },
    })
    evaluator = instantiate(config, model=DerivativeModel())
    loader = DataLoader(EvaluationDataset(), batch_size=2, collate_fn=collate_atoms_data)
    metrics = evaluator.evaluate(loader)
    assert metrics["num_structures"] == 3
    assert (tmp_path / "metrics.json").is_file()
    assert not (tmp_path / "results.pt").exists()


def test_evaluator_rejects_unit_mismatch_and_restores_model_mode(tmp_path):
    model = DerivativeModel()
    model.unit = "hartree"
    evaluator = VariationalEvaluator(model, evaluation_loss(), output_dir=tmp_path)
    with pytest.raises(ValueError, match="units do not match"):
        evaluator.evaluate(EvaluationDataset())
    assert model.training
    assert not (tmp_path / "metrics.json").exists()


def oracle_evaluation_objects():
    from curator.data.variational_oracle import VariationalOracleDataset
    from curator.model import NeuralNetworkPotential
    from curator.model.variational_oracle import LearnedVariationalOracle

    dataset = VariationalOracleDataset(split="test", num_train=8, num_val=2, num_test=2, num_conditions=3)
    model = NeuralNetworkPotential(LearnedVariationalOracle())
    model.representation.setup_from_datamodule(SimpleNamespace(metadata=dataset.metadata))
    loss_fn = VariationalJointLoss(
        supervised_weights={"condition_energy": 1.0, "condition_forces": 1.0, "variable_value": 1.0},
        supervised_scales={"condition_energy": 1.0, "condition_forces": 1.0, "variable_value": 1.0},
        required_labels=("condition_energy", "condition_forces", "variable_value"),
        response_weight=1.0, response_scales={"1/eV": 1.0},
    )
    return model, dataset, loss_fn


def test_evaluator_handles_learned_oracle_force_and_response_contract(tmp_path):
    model, dataset, loss_fn = oracle_evaluation_objects()
    evaluator = VariationalEvaluator(model, loss_fn, output_dir=tmp_path, batch_size=1)
    metrics = evaluator.evaluate(dataset)
    assert metrics["num_structures"] == 2
    assert metrics["num_conditions"] == 6
    assert metrics["terms"]["response"]["n_conditions"] == 6
    assert metrics["terms"]["condition_forces"]["rmse"] > 0
    artifact = torch.load(tmp_path / "results.pt", weights_only=False)
    for record in artifact["batches"]:
        assert "responses" in record["prediction"]["variational_prediction"]
        assert all(item["split"] == "test" for item in record["batch"]["meta"])


@pytest.mark.parametrize("use_loader", [False, True])
def test_oracle_evaluation_rejects_changed_split_seed_that_leaks_training_geometries(tmp_path, use_loader):
    from curator.data.variational_oracle import VariationalOracleDataset

    model, original, loss_fn = oracle_evaluation_objects()
    changed = VariationalOracleDataset(split="test", num_train=8, num_val=2, num_test=2,
                                      num_conditions=3, seed=18)
    assert set(original.metadata["splits"]["train"]) & set(changed.indices)
    if use_loader:
        changed = DataLoader(changed, batch_size=2, collate_fn=collate_atoms_data)
    evaluator = VariationalEvaluator(model, loss_fn, output_dir=tmp_path)
    with pytest.raises(ValueError, match="dataset seed"):
        evaluator.evaluate(changed)
    assert not (tmp_path / "metrics.json").exists()


@pytest.mark.parametrize("mutation", ["splits", "missing", "unbound"])
def test_oracle_evaluation_requires_the_original_training_manifest(tmp_path, mutation):
    model, dataset, loss_fn = oracle_evaluation_objects()
    if mutation == "splits":
        dataset.metadata["splits"]["test"] = dataset.metadata["splits"]["train"]
    elif mutation == "missing":
        del dataset.metadata
    else:
        model.representation._dataset_metadata = None
    with pytest.raises(ValueError, match="metadata/splits"):
        VariationalEvaluator(model, loss_fn, output_dir=tmp_path).evaluate(dataset)


def test_oracle_state_dict_restores_and_checks_static_and_dataset_metadata():
    from curator.model import NeuralNetworkPotential
    from curator.model.variational_oracle import LearnedVariationalOracle

    model, dataset, _ = oracle_evaluation_objects()
    state = copy.deepcopy(model.state_dict())
    restored = NeuralNetworkPotential(LearnedVariationalOracle(stage="proposal"))
    restored.load_state_dict(state)
    restored.initialize_modules(SimpleNamespace(metadata=dataset.metadata))
    restored.representation.validate_dataset_metadata(dataset.metadata)
    assert restored.representation.get_extra_state() == model.representation.get_extra_state()

    invalid = copy.deepcopy(state)
    invalid["representation._extra_state"]["units"]["condition_energy"] = "hartree"
    with pytest.raises(ValueError, match="schema/units/gauge/layout"):
        restored.load_state_dict(invalid)
    invalid = copy.deepcopy(state)
    invalid["representation._extra_state"]["dataset_metadata"]["splits"]["test"] = []
    with pytest.raises(ValueError, match="metadata/splits"):
        restored.load_state_dict(invalid)
    invalid = copy.deepcopy(state)
    del invalid["representation._extra_state"]["dataset_metadata"]
    with pytest.raises(ValueError, match="missing dataset metadata"):
        restored.load_state_dict(invalid)


def test_full_object_checkpoint_preserves_snapshot_and_rejects_changed_code_semantics(tmp_path, monkeypatch):
    from curator.model import variational_oracle

    model, dataset, loss_fn = oracle_evaluation_objects()
    checkpoint = io.BytesIO()
    torch.save(model, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, weights_only=False)
    restored.representation.validate_dataset_metadata(dataset.metadata)
    monkeypatch.setitem(variational_oracle.ORACLE_UNITS, "condition_energy", "hartree")
    assert restored.representation.get_extra_state()["units"]["condition_energy"] == "eV"
    with pytest.raises(ValueError, match="schema/units/gauge/layout"):
        VariationalEvaluator(restored, loss_fn, output_dir=tmp_path).evaluate(dataset)


def test_oracle_predictions_do_not_read_supervised_or_response_labels():
    model, dataset, _ = oracle_evaluation_objects()
    model.eval()
    batch = collate_atoms_data([dataset[0], dataset[1]])
    poisoned = copy.deepcopy(batch)
    for label in poisoned["variational"]["labels"].values():
        label["values"].fill_(999999.0)
    poisoned["variational"]["response_values"].fill_(-999999.0)
    original = model(batch)
    changed = model(poisoned)
    for name, label in original["variational_prediction"]["labels"].items():
        torch.testing.assert_close(label["values"], changed["variational_prediction"]["labels"][name]["values"],
                                   rtol=0, atol=0)
    torch.testing.assert_close(original["variational_prediction"]["responses"]["values"],
                               changed["variational_prediction"]["responses"]["values"], rtol=0, atol=0)
    torch.testing.assert_close(original["amortized_training"].projected_state,
                               changed["amortized_training"].projected_state, rtol=0, atol=0)
