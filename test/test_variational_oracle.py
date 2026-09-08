"""Independent reference and phase-isolation tests for the learned oracle."""

import copy

import pytest
import torch

from curator.data.variational_oracle import VariationalOracleDataModule, oracle_coefficients
from curator.model import NeuralNetworkPotential
from curator.model.variational_oracle import LearnedVariationalOracle
from curator.train import AmortizedVariationalLoss, VariationalJointLoss


def _data():
    module = VariationalOracleDataModule(num_train=4, num_val=2, num_test=2,
                                        num_conditions=3, batch_size=2)
    module.setup()
    return module


def _physical_loss():
    labels = {"condition_energy": 1.0, "condition_forces": 1.0, "variable_value": 1.0}
    return VariationalJointLoss(supervised_weights=labels, supervised_scales=labels,
                                required_labels=tuple(labels), response_weight=1.0,
                                response_scales={"1/eV": 1.0})


def test_oracle_splits_are_reproducible_disjoint_and_checkpointed():
    first, rebuilt = _data(), _data()
    assert first.metadata == rebuilt.metadata
    splits = [set(first.datasets[name].indices) for name in ("train", "val", "test")]
    assert sum(len(values) for values in splits) == len(set.union(*splits)) == 8
    for name in first.datasets:
        for left, right in zip(first.datasets[name], rebuilt.datasets[name]):
            assert left.variational.geometry_hash == right.variational.geometry_hash
            for left_condition, right_condition in zip(left.variational.conditions, right.variational.conditions):
                for key, value in left_condition.labels.items():
                    assert value.dtype == torch.float64
                    torch.testing.assert_close(value, right_condition.labels[key], rtol=0, atol=0)
    next(iter(first.train_dataloader()))
    saved = copy.deepcopy(first.state_dict())
    # Sanity validation/test must not change the next training shuffle.
    list(first.val_dataloader())
    list(first.test_dataloader())
    assert torch.equal(saved["generator_state"], first.generator.get_state())
    rebuilt.load_state_dict(saved)
    assert next(iter(first.train_dataloader()))["meta"] == next(iter(rebuilt.train_dataloader()))["meta"]
    wrong_seed = VariationalOracleDataModule(num_train=4, num_val=2, num_test=2,
                                            num_conditions=3, seed=18)
    with pytest.raises(ValueError, match="metadata/splits"):
        wrong_seed.load_state_dict(saved)


def test_known_coefficients_reproduce_independent_energy_force_state_response_labels():
    batch = next(iter(_data().test_dataloader()))
    model = LearnedVariationalOracle()
    # Install the published analytic coefficients only in this reference test.
    # The production constructor initializes independently and never reads them.
    with torch.no_grad():
        model.physical.weight[:, 0] = torch.tensor([
            .10, .03, -.08, .02, -.02, .07, -.03, .02, .04, -.09,
            .12, .08, -.10, .06, .20,
        ], dtype=torch.float64)
        model.physical.weight[:, 1] = torch.tensor(
            [0.] * 10 + [.02, -.03, .01, .04, .18], dtype=torch.float64)
        model.physical.bias.copy_(torch.tensor([
            1.3, .12, 1.1, -.08, .07, 1.2, .05, -.09, .11, 1.4,
            .25, -.15, .20, -.30, .15,
        ], dtype=torch.float64))
    positions = batch["positions"].reshape(-1, 2, 3)
    for fitted, expected in zip(model.coefficients(positions), oracle_coefficients(positions)):
        torch.testing.assert_close(fitted, expected, rtol=1e-13, atol=1e-13)
    prediction = NeuralNetworkPotential(model)(batch)
    result = _physical_loss()(prediction, batch)
    assert float(result["loss"].detach()) < 1e-25


@pytest.mark.parametrize("stage", ["physical", "proposal"])
def test_stage_gradient_isolation_survives_generic_requires_grad_policy(stage):
    batch = next(iter(_data().train_dataloader()))
    model = NeuralNetworkPotential(LearnedVariationalOracle(stage=stage))
    # Generic training policies may mark every parameter as trainable.
    model.requires_grad_(True)
    before = {name: value.clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    prediction = model(batch)
    loss = _physical_loss()(prediction, batch)["loss"]
    loss = loss + AmortizedVariationalLoss()(prediction["amortized_training"])["loss"]
    loss.backward()
    active = model.representation.physical if stage == "physical" else model.representation.proposal
    frozen = model.representation.proposal if stage == "physical" else model.representation.physical
    assert all(parameter.grad is None for parameter in frozen.parameters())
    assert sum(float(parameter.grad.square().sum()) for parameter in active.parameters()) > 0
    optimizer.step()
    active_prefix = "representation." + stage + "."
    assert any(not torch.equal(value, before[name]) for name, value in model.named_parameters()
               if name.startswith(active_prefix))
    assert all(torch.equal(value, before[name]) for name, value in model.named_parameters()
               if not name.startswith(active_prefix))


def test_force_loss_gradient_matches_parameter_finite_difference_and_inputs_need_no_labels():
    batch = next(iter(_data().test_dataloader()))
    model = LearnedVariationalOracle()
    predictions = model(batch)["variational_prediction"]["labels"]
    target = batch["variational"]["labels"]["condition_forces"]["values"]
    force_loss = (predictions["condition_forces"]["values"] - target).square().mean()
    gradient = torch.autograd.grad(force_loss, model.physical.weight)[0][0, 0]
    epsilon = 1e-5
    values = []
    original = model.physical.weight[0, 0].detach().clone()
    for sign in (-1, 1):
        with torch.no_grad():
            model.physical.weight[0, 0] = original + sign * epsilon
        forces = model(batch)["variational_prediction"]["labels"]["condition_forces"]["values"]
        values.append((forces - target).square().mean().detach())
    torch.testing.assert_close(gradient, (values[1] - values[0]) / (2 * epsilon), rtol=1e-6, atol=1e-8)
    with torch.no_grad():
        model.physical.weight[0, 0] = original
    inputs_only = dict(batch)
    inputs_only["variational"] = {key: batch["variational"][key]
                                  for key in ("condition_geometry_index", "boundary_spec", "boundary_hash")}
    rebuilt = model(inputs_only)["variational_prediction"]["labels"]
    for key in predictions:
        torch.testing.assert_close(predictions[key]["values"], rebuilt[key]["values"])


def test_checkpoint_rebuild_preserves_semantics_and_allows_changing_training_stage():
    physical = NeuralNetworkPotential(LearnedVariationalOracle())
    proposal = NeuralNetworkPotential(LearnedVariationalOracle(stage="proposal"))
    proposal.load_state_dict(copy.deepcopy(physical.state_dict()), strict=True)
    assert proposal.representation.stage == "proposal"
    assert not any(parameter.requires_grad for parameter in proposal.representation.physical.parameters())
    batch = next(iter(_data().test_dataloader()))
    for name, expected in physical(batch)["variational_prediction"]["labels"].items():
        torch.testing.assert_close(proposal(batch)["variational_prediction"]["labels"][name]["values"],
                                   expected["values"])
    invalid = copy.deepcopy(physical.state_dict())
    invalid["representation._extra_state"]["data_seed"] = 999
    with pytest.raises(ValueError, match="units/gauge/layout/data seed"):
        proposal.load_state_dict(invalid)
