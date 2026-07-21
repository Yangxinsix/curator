import torch

from curator.data import properties
from curator.train import (
    HuberLoss,
    SpeciesBalancedLoss,
    StructureBalancedLoss,
    VectorHuberLoss,
    VectorL2MAELoss,
)
from curator.train.model_output import ModelOutput


def test_vector_l2_mae_loss_reduces_over_cartesian_axis():
    prediction = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    target = torch.zeros_like(prediction)

    assert torch.isclose(VectorL2MAELoss()(prediction, target), torch.tensor(2.5))


def test_model_output_preserves_last_axis_when_flattening_hessian_rows():
    output = ModelOutput(name="rows")
    rows = [torch.zeros(4, 5, 3), torch.zeros(4, 2, 3)]

    flattened = output._flatten_value(rows)

    assert flattened.shape == (28, 3)


def test_huber_loss_is_available_from_curator_train():
    prediction = torch.tensor([0.5, 3.0])
    target = torch.zeros_like(prediction)

    assert torch.isclose(HuberLoss(delta=1.0)(prediction, target), torch.tensor(1.3125))


def test_vector_huber_loss_uses_vector_norm_before_robust_penalty():
    prediction = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    target = torch.zeros_like(prediction)

    assert torch.isclose(VectorHuberLoss(delta=2.0)(prediction, target), torch.tensor(4.0))


def test_structure_balanced_loss_weights_structures_equally():
    prediction = torch.tensor([[4.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = torch.zeros_like(prediction)
    batch = {properties.n_atoms: torch.tensor([1, 3])}

    assert torch.isclose(StructureBalancedLoss()(prediction, target, batch), torch.tensor(2.5))


def test_species_balanced_loss_weights_species_equally():
    prediction = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    target = torch.zeros_like(prediction)
    batch = {
        properties.n_atoms: torch.tensor([4]),
        properties.Z: torch.tensor([1, 1, 1, 8]),
    }

    assert torch.isclose(SpeciesBalancedLoss()(prediction, target, batch), torch.tensor(3.0))


def test_model_output_passes_batch_metadata_to_balanced_loss():
    prediction = torch.tensor([[4.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = {
        properties.forces: torch.zeros_like(prediction),
        properties.n_atoms: torch.tensor([1, 3]),
    }
    output = ModelOutput(name=properties.forces, loss_fn=StructureBalancedLoss())

    loss, _ = output.calculate_loss({properties.forces: prediction}, target)

    assert torch.isclose(loss, torch.tensor(2.5))
