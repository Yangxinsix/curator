"""Energy normalization uses each condition's geometry, including repeats."""
import pytest
import torch
from curator.data import properties
from curator.train import VariationalJointLoss


def example():
    values = torch.tensor([2., 4., 8.], requires_grad=True)
    entry = {"values": values, "ptr": torch.arange(4), "shape": ((1,),) * 3, "unit": "eV"}
    target = {**entry, "values": torch.zeros(3), "present": torch.ones(3, dtype=torch.bool), "metadata": ({},) * 3}
    batch = {properties.n_atoms: torch.tensor([2, 4]), "variational": {
        "condition_weight": torch.ones(3), "condition_geometry_index": torch.tensor([0, 0, 1]),
        "condition_ptr": torch.tensor([0, 2, 3]), "labels": {"condition_energy": target}}}
    prediction = {"variational_prediction": {"value_space": "physical", "labels": {"condition_energy": entry}}}
    return values, prediction, batch


def test_per_atom_energy_balances_repeated_conditions_and_gradients():
    values, prediction, batch = example()
    loss = VariationalJointLoss({"condition_energy": 1}, {"condition_energy": 1}, energy_per_atom=True)
    result = loss(prediction, batch)["loss"]
    assert result.item() == pytest.approx(3.25)
    result.backward()
    torch.testing.assert_close(values.grad, torch.tensor([.25, .5, .5]))
    torch.testing.assert_close(values.detach(), torch.tensor([2., 4., 8.]))


def test_default_and_legacy_loss_keep_total_energy_convention():
    _, prediction, batch = example()
    loss = VariationalJointLoss({"condition_energy": 1}, {"condition_energy": 1})
    assert loss(prediction, batch)["loss"].item() == pytest.approx(37.)
    del loss.energy_per_atom
    assert loss(prediction, batch)["loss"].item() == pytest.approx(37.)


def test_per_atom_energy_rejects_zero_atom_count():
    _, prediction, batch = example()
    batch[properties.n_atoms][0] = 0
    loss = VariationalJointLoss({"condition_energy": 1}, {"condition_energy": 1}, energy_per_atom=True)
    with pytest.raises(ValueError, match="positive atom counts"):
        loss(prediction, batch)
