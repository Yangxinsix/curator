"""Real-label adapter and the runnable SPICE joint GNN training configuration."""

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from curator.data import properties
from curator.data.spice_acvr import SpiceACVRDataModule
from curator.variational import VariableLayout, compile_boundary
from curator.layer._variational_output import _boundary_from_dict
from test_spice_acvr_data import source


def test_adapter_preserves_labels_splits_and_uses_charge_as_boundary(source):
    ordinary = SpiceACVRDataModule(source, shuffle=False)
    joint = SpiceACVRDataModule(source, shuffle=False, variational=True)
    ordinary.setup()
    joint.setup()
    assert joint.metadata["split_indices"] == ordinary.metadata["split_indices"]
    assert joint.metadata["source_sha256"] == ordinary.metadata["source_sha256"]
    assert not joint.metadata["variational_adapter"]["atomic_charge_supervision"]
    for index in range(len(joint.train_dataset)):
        original, adapted = ordinary.train_dataset[index], joint.train_dataset[index]
        payload = adapted.variational
        payload.validate(2)
        condition = payload.conditions[0]
        torch.testing.assert_close(condition.labels["condition_energy"], original[properties.energy])
        torch.testing.assert_close(condition.labels["condition_forces"], original[properties.forces])
        assert "variable_value" not in condition.labels
        spec = _boundary_from_dict(condition.boundary_spec)
        compiled = compile_boundary(spec, VariableLayout(spec.variable, 2))
        torch.testing.assert_close(compiled.constraint_matrix, torch.ones(1, 2, dtype=torch.float64))
        torch.testing.assert_close(compiled.constraint_target, original[properties.total_charge])
    batch = next(iter(joint.train_dataloader()))
    assert batch["variational"]["condition_state_ptr"].tolist() == list(range(0, 17, 2))


def test_spice_config_trains_real_gnn_without_internal_teacher(source, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "curator/configs/acvr_spice_joint.yaml")
    config.data.datapath = str(source)
    config.data.batch_size = 3
    data = instantiate(config.data)
    data.setup()
    batch = next(iter(data.train_dataloader()))

    def forbidden(*args, **kwargs):
        raise AssertionError("joint training must not solve for an internal teacher or full Hessian")

    import curator.layer._variational_output as layer
    monkeypatch.setattr(layer, "solve_functional_reference", forbidden)
    monkeypatch.setattr(torch.autograd.functional, "hessian", forbidden)
    model = instantiate(config.model).double().train()
    task = instantiate(config.task, model=model)
    prediction = model(batch)
    losses, _ = task.loss_fn(prediction, batch, "train")
    loss = losses["train_total_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    groups = model.parameter_groups()
    assert {group.name for group in groups} == {"backbone", "baseline", "functional", "proposal"}
    assert all(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in group.params) for group in groups)
    values = prediction["variational_prediction"]
    charges = values["labels"]["variable_value"]["values"].reshape(-1, 2)
    torch.testing.assert_close(charges.sum(-1), batch[properties.total_charge])
    assert values["metadata"]["energy_state"] == "predicted"
    assert model.output_modules[0].get_extra_state()["contract"]["response_unit"] == "e^2/eV"
