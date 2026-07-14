import torch

from curator.train.model_output import ModelOutput


def test_per_atom_energy_loss_matches_mace_normalization():
    output = ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        per_atom_loss=True,
    )
    pred = {"energy": torch.tensor([6.0, 8.0])}
    target = {
        "energy": torch.tensor([2.0, 4.0]),
        "n_atoms": torch.tensor([2, 4]),
    }

    loss, num_obs = output.calculate_loss(pred, target)

    assert torch.isclose(loss, torch.tensor(2.5))
    assert num_obs == 2


def test_per_atom_energy_loss_can_be_disabled():
    output = ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        per_atom_loss=False,
    )
    pred = {"energy": torch.tensor([6.0, 8.0])}
    target = {
        "energy": torch.tensor([2.0, 4.0]),
        "n_atoms": torch.tensor([2, 4]),
    }

    loss, _ = output.calculate_loss(pred, target)

    assert torch.isclose(loss, torch.tensor(16.0))


def test_per_atom_energy_loss_broadcasts_over_trailing_dimensions():
    output = ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        per_atom_loss=True,
    )
    pred = {"energy": torch.tensor([[6.0], [8.0]])}
    target = {
        "energy": torch.tensor([[2.0], [4.0]]),
        "n_atoms": torch.tensor([2, 4]),
    }

    loss, _ = output.calculate_loss(pred, target)

    assert torch.isclose(loss, torch.tensor(2.5))
