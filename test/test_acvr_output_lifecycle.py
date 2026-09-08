"""Stage isolation and checkpoint contracts shared by both ACVR loss adapters."""

import io

import pytest
import torch

from curator.data import properties
from curator.train import (
    AmortizedTrainingModelOutput,
    AmortizedVariationalLoss,
    VariationalJointLoss,
    VariationalModelOutput,
)
from curator.variational import QuadraticProblem, prepare_amortized_training_batch


@pytest.fixture(params=["physical", "proposal"])
def adapter(request):
    if request.param == "physical":
        return VariationalModelOutput(
            VariationalJointLoss({"energy": 2.0}, {"energy": 1.0})
        ), "energy"
    return AmortizedTrainingModelOutput(
        AmortizedVariationalLoss(state_weight=2.0, kkt_weight=0.0)
    ), "proposal_state"


def _inputs(output, errors, weights=None):
    errors = torch.tensor(errors, dtype=torch.float64, requires_grad=True)
    count = errors.numel()
    batch = {
        "weight": torch.tensor(weights or [1.0] * count, dtype=torch.float64),
        "variational": {
            "condition_geometry_index": torch.arange(count),
            "condition_weight": torch.ones(count, dtype=torch.float64),
        },
    }
    if isinstance(output, AmortizedTrainingModelOutput):
        problem = QuadraticProblem.unconstrained(
            torch.eye(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
        )
        prediction = {
            "amortized_training": prepare_amortized_training_batch(
                (problem,) * count, errors
            )
        }
    else:
        metadata = {
            "ptr": torch.arange(count + 1),
            "shape": ((1,),) * count,
            "unit": "eV",
        }
        batch["variational"]["labels"] = {
            "energy": dict(
                metadata,
                values=torch.zeros_like(errors),
                present=torch.ones(count, dtype=torch.bool),
                metadata=({},) * count,
            )
        }
        prediction = {
            "variational_prediction": {
                "value_space": "physical",
                "labels": {"energy": dict(metadata, values=errors)},
            }
        }
    return prediction, batch


def test_interleaved_stages_keep_loss_and_pending_metrics_separate(adapter):
    output, term = adapter
    train = _inputs(output, [1.0, 3.0], [1.0, 3.0])
    val = _inputs(output, [10.0])
    test = _inputs(output, [20.0])
    loss, count = output.calculate_loss(*train, subset="train")
    assert loss.item() == pytest.approx(14.0)
    assert count == 2
    loss.backward()
    output.calculate_loss(*val, subset="val")
    output.calculate_loss(*test, subset="test")

    # Validation can run while the training epoch remains open.
    assert output.accumulate_loss("train") == pytest.approx(14.0)
    assert output.accumulate_loss("val") == pytest.approx(200.0)
    assert output.accumulate_loss("test") == pytest.approx(800.0)
    assert output.calculate_metrics(*val, "val")[f"val_{output.name}_{term}_loss"].item() == pytest.approx(200.0)
    assert output.calculate_metrics(*train, "train")[f"train_{output.name}_{term}_loss"].item() == pytest.approx(14.0)

    output.reset_metrics(subset="val")
    assert output.accumulate_metrics("val") == {}
    assert output.accumulate_metrics("train")[term + "_loss"] == pytest.approx(14.0)
    assert output.accumulate_metrics("test")[term + "_loss"] == pytest.approx(800.0)
    assert output.calculate_metrics(*test, "test")

    output.calculate_loss(*_inputs(output, [5.0], [2.0]), subset="train")
    # The aggregate is (2 * 28 + 2 * 50) / (4 + 2), not the mean of
    # the two batch losses (14 + 50) / 2.
    assert output.accumulate_loss("train") == pytest.approx(26.0)
    assert output.accumulate_metrics("train")[term + "_loss"] == pytest.approx(26.0)
    output.reset_metrics(subset="train")
    output.calculate_loss(*_inputs(output, [2.0]), subset="train")
    assert output.accumulate_loss("train") == pytest.approx(8.0)


def test_plain_trainer_protocol_still_assigns_metrics_to_the_requested_stage(adapter):
    output, term = adapter
    for subset, error in (("train", 1.0), ("val", 3.0), ("test", 5.0)):
        prediction, batch = _inputs(output, [error])
        loss = output.calculate_loss(prediction, batch, return_num_obs=False)
        metrics = output.calculate_metrics(prediction, batch, subset)
        assert loss.item() == pytest.approx(2.0 * error**2)
        assert metrics[f"{subset}_{output.name}_{term}_loss"].item() == pytest.approx(loss.item())
        assert output.calculate_metrics(prediction, batch, subset) == {}
    assert output.accumulate_loss() == pytest.approx(70.0 / 3.0)
    for subset, expected in (("train", 2.0), ("val", 18.0), ("test", 50.0)):
        assert output.accumulate_metrics(subset)[term + "_loss"] == pytest.approx(expected)
    output.reset_loss()
    assert output.accumulate_loss() == 0.0


def test_full_object_and_state_dict_checkpoints_do_not_restore_epoch_totals(adapter):
    output, term = adapter
    prediction, batch = _inputs(output, [3.0])
    output.calculate_loss(prediction, batch, subset="train")
    assert output.accumulate_metrics("train")[term + "_loss"] == pytest.approx(18.0)
    assert not any("_epoch_" in key for key in output.state_dict())

    checkpoint = io.BytesIO()
    torch.save({"outputs": [output]}, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, weights_only=False)["outputs"][0]
    assert restored.accumulate_metrics("train") == {}
    assert restored.calculate_metrics(prediction, batch, "train") == {}
    # Saving must leave the live epoch and pending batch untouched.
    assert output.accumulate_loss("train") == pytest.approx(18.0)
    assert output.calculate_metrics(prediction, batch, "train")

    output.load_state_dict(restored.state_dict())
    assert output.accumulate_loss("train") == 0.0
    assert output.accumulate_metrics("train") == {}
    restored.calculate_loss(prediction, batch, subset="train")
    assert restored.accumulate_loss("train") == pytest.approx(18.0)


def test_legacy_persistent_epoch_buffers_are_discarded_on_load(adapter):
    output, _ = adapter
    state = output.state_dict()
    state["_epoch_numerator"] = torch.ones(len(output.term_names)) * 100
    state["_epoch_denominator"] = torch.ones(len(output.term_names))
    output.load_state_dict(state, strict=True)
    assert output.accumulate_loss() == 0.0
    assert output.accumulate_loss("train") == 0.0


def test_output_weight_is_applied_once_and_frozen_in_each_batch_aggregate(adapter):
    output, term = adapter
    output.loss_weight = 0.5
    loss, _ = output.calculate_loss(*_inputs(output, [3.0]), subset="train")
    assert loss.item() == pytest.approx(9.0)
    output.loss_weight = 0.25
    loss, _ = output.calculate_loss(*_inputs(output, [3.0]), subset="train")
    assert loss.item() == pytest.approx(4.5)
    assert output.accumulate_metrics("train")[term + "_loss"] == pytest.approx(6.75)


def test_masked_terms_keep_their_own_epoch_denominators():
    output = VariationalModelOutput(
        VariationalJointLoss(
            {"energy": 2.0, "charge": 0.5}, {"energy": 1.0, "charge": 1.0}
        )
    )
    for errors, weights, present in (
        ([1.0, 3.0], [1.0, 3.0], [True, False]),
        ([5.0], [2.0], [True]),
    ):
        prediction, batch = _inputs(output, errors, weights)
        target = batch["variational"]["labels"]
        predicted = prediction["variational_prediction"]["labels"]
        target["charge"] = dict(target["energy"], present=torch.tensor(present))
        predicted["charge"] = dict(predicted["energy"])
        output.calculate_loss(prediction, batch, subset="train")
    metrics = output.accumulate_metrics("train")
    assert metrics["energy_loss"] == pytest.approx(26.0)
    assert metrics["charge_loss"] == pytest.approx(0.5 * (1 + 2 * 25) / 3)
    assert output.accumulate_loss("train") == pytest.approx(34.5)


def test_proposal_output_accepts_one_condition_per_geometry_in_plain_atom_batches():
    output = AmortizedTrainingModelOutput(
        AmortizedVariationalLoss(state_weight=2.0, kkt_weight=0.0)
    )
    prediction, explicit_batch = _inputs(output, [1.0, 3.0], [1.0, 3.0])
    plain_batch = {
        properties.n_atoms: torch.tensor([1, 1]),
        "weight": explicit_batch["weight"],
    }
    explicit_loss, explicit_count = output.calculate_loss(prediction, explicit_batch, subset="val")
    plain_loss, plain_count = output.calculate_loss(prediction, plain_batch, subset="train")
    assert plain_loss.item() == pytest.approx(14.0)
    assert plain_loss.item() == pytest.approx(explicit_loss.item())
    assert plain_count == explicit_count == 2
    assert output.accumulate_loss("train") == pytest.approx(output.accumulate_loss("val"))
    plain_loss.backward()
    assert prediction["amortized_training"].proposal.grad is not None


@pytest.mark.parametrize("n_atoms", [None, torch.tensor(2), torch.tensor([[1], [1]]), torch.tensor([2])])
def test_proposal_output_rejects_ambiguous_plain_atom_batches(n_atoms):
    output = AmortizedTrainingModelOutput(AmortizedVariationalLoss())
    prediction, batch = _inputs(output, [1.0, 3.0])
    del batch["variational"]
    if n_atoms is not None:
        batch[properties.n_atoms] = n_atoms
    with pytest.raises(ValueError, match="exactly one condition per geometry"):
        output.calculate_loss(prediction, batch)
