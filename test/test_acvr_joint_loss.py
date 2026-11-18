"""Teacher-free residual loss and optimizer isolation regressions."""

import pytest
import torch

from curator.train import VariationalJointLoss, VariationalModelOutput


def _batch():
    return {
        "weight": torch.tensor([1.0, 0.5]),
        "variational": {
            "condition_ptr": torch.tensor([0, 2, 3]),
            "condition_weight": torch.tensor([1.0, 2.0, 1.0]),
            "condition_geometry_index": torch.tensor([0, 0, 1]),
            "boundary_spec": ({}, {}, {}),
            "labels": {},
        },
    }


def _prediction(values):
    return {"variational_prediction": {
        "value_space": "physical",
        "residuals": {"stationarity": {
            "values": values,
            "ptr": torch.tensor([0, 2, 3, 6]),
            "source": "variational_energy_gradient",
            "gradient_policy": "proposal_only",
            "normalization": {"kind": "fixed_physical_scale", "scale": 1.0},
        }},
    }}


def test_residual_loss_is_geometry_balanced_and_requires_no_teacher():
    state = torch.arange(1.0, 7.0, requires_grad=True)
    # An actual energy-derived gradient, with an analytic expected loss.
    energy = 0.5 * state.square().sum()
    gradient, = torch.autograd.grad(energy, state, create_graph=True)
    loss_fn = VariationalJointLoss({}, {}, residual_weights={"stationarity": 0.2})
    output = VariationalModelOutput(loss_fn)
    result = loss_fn(_prediction(gradient), _batch())
    assert result["loss"].item() == pytest.approx(0.2 * 118 / 9)
    term = result["terms"]["residual:stationarity"]
    assert term["n_conditions"] == 3 and term["n_geometries"] == 2
    assert output.term_names == ("residual:stationarity",)
    result["loss"].backward()
    assert torch.all(state.grad > 0)


def test_residual_only_adamw_step_does_not_touch_physical_parameters():
    physical = torch.nn.Parameter(torch.tensor(0.7))
    proposal = torch.nn.Parameter(torch.tensor(0.4))
    state = proposal * torch.arange(1.0, 7.0)
    energy = 0.5 * physical.detach() * state.square().sum()
    gradient, = torch.autograd.grad(energy, state, create_graph=True)
    prediction = _prediction(gradient)
    # Last insertion deliberately makes the loss's reference tensor physical.
    prediction["variational_prediction"]["baseline_energy"] = physical.square()
    loss_fn = VariationalJointLoss(
        {"missing_optional": 1.0, "disabled": 0.0}, {"missing_optional": 1.0},
        residual_weights={"stationarity": 1.0},
    )
    optimizer = torch.optim.AdamW([physical, proposal], lr=0.1, weight_decay=0.1)
    before_physical, before_proposal = physical.detach().clone(), proposal.detach().clone()
    loss_fn(prediction, _batch())["loss"].backward()
    assert physical.grad is None
    assert proposal.grad is not None and proposal.grad.abs() > 0
    optimizer.step()
    assert torch.equal(physical, before_physical)
    assert not torch.equal(proposal, before_proposal)


@pytest.mark.parametrize("field,value,match", [
    ("source", "learned_residual_head", "energy gradient"),
    ("gradient_policy", "all_parameters", "proposal_only"),
    ("normalization", None, "normalization"),
])
def test_residual_contract_rejects_wrong_source_or_gradient_policy(field, value, match):
    prediction = _prediction(torch.ones(6))
    prediction["variational_prediction"]["residuals"]["stationarity"][field] = value
    with pytest.raises(ValueError, match=match):
        VariationalJointLoss({}, {}, residual_weights={"stationarity": 1})(prediction, _batch())


def test_disabled_residual_does_not_require_prediction_or_teacher():
    prediction = {"variational_prediction": {"value_space": "physical"}}
    result = VariationalJointLoss({}, {}, residual_weights={"stationarity": 0})(prediction, _batch())
    assert result["loss"] == 0
    assert result["terms"]["residual:stationarity"]["status"] == "disabled"
