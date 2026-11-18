"""Predicted-state and local-equilibrium evaluation keep distinct semantics."""

import json
import copy
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import pytest
import torch

from curator.evaluate.variational import VariationalEvaluator


def joint_objects():
    configs = Path(__file__).resolve().parents[1] / "curator/configs"
    with initialize_config_dir(version_base=None, config_dir=str(configs)):
        config = compose(config_name="acvr_joint", overrides=[
            "data.num_train=4", "data.num_val=2", "data.num_test=2",
            "data.num_conditions=2", "data.batch_size=2",
            "model.representation.num_features=4", "model.representation.num_interactions=1",
            "model.representation.num_basis=4", "model.output_modules.0.hidden_features=4",
        ])
    torch.manual_seed(config.seed)
    datamodule = instantiate(config.data)
    datamodule.setup()
    model = instantiate(config.model).double()
    model.initialize_modules(datamodule)
    model.double()
    loss = instantiate(config.task.outputs[0].loss_fn)
    return config, model, datamodule, loss


def test_joint_evaluator_compares_same_functional_and_restores_modes(tmp_path):
    _, model, data, loss = joint_objects()
    output = model.output_modules[0]
    assert model.training and output.state_mode == "predicted"
    evaluator = VariationalEvaluator(model, loss, output_dir=tmp_path, batch_size=2,
                                     state_mode="predicted", compare_converged=True)
    with torch.inference_mode():
        metrics = evaluator.evaluate(data.datasets["test"])
    assert model.training and output.state_mode == "predicted"
    expected_metadata = {
        "training_mode": "joint", "energy_state": "predicted",
        "force_derivative": "full_proposal_path", "residual_gradient_policy": "proposal_only",
    }
    assert expected_metadata.items() <= metrics["metadata"].items()
    assert metrics["comparison"]["metadata"]["energy_state"] == "converged"
    assert metrics["comparison"]["metadata"]["force_derivative"] == "stationary_envelope"
    assert metrics["reference_solver"]["scope"] == "local_stationary_point"
    assert metrics["reference_solver"]["n_conditions"] == 4
    assert metrics["reference_solver"]["n_converged"] == 4
    assert metrics["comparison"]["terms"]["condition_energy"]["rmse"] is not None
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    artifact = torch.load(tmp_path / "results.pt", weights_only=False)
    assert artifact["format_version"] == 2
    assert artifact["batches"]
    sums = {name: [0.0, 0] for name in metrics["comparison"]["difference_from_predicted"]}
    for record in artifact["batches"]:
        primary = record["prediction"]["variational_prediction"]
        comparison = record["comparison_prediction"]["variational_prediction"]
        assert primary["metadata"]["energy_state"] == "predicted"
        assert comparison["metadata"]["energy_state"] == "converged"
        for name, totals in sums.items():
            delta = primary["labels"][name]["values"] - comparison["labels"][name]["values"]
            assert not delta.requires_grad
            totals[0] += float(delta.square().sum())
            totals[1] += delta.numel()
    for name, (squared, count) in sums.items():
        expected = (squared / count) ** 0.5
        actual = metrics["comparison"]["difference_from_predicted"][name]
        assert actual["scalar_rmse"] == pytest.approx(expected)
        assert actual["n_scalars"] == count


def test_joint_converged_evaluation_is_explicit_and_restores_after_failure(tmp_path):
    _, model, data, loss = joint_objects()
    output = model.output_modules[0]
    evaluator = VariationalEvaluator(model, loss, output_dir=tmp_path, state_mode="converged")
    metrics = evaluator.evaluate(data.datasets["test"])
    assert metrics["metadata"]["energy_state"] == "converged"
    assert "comparison" not in metrics
    assert output.state_mode == "predicted" and model.training

    original = output.forward

    def failing_forward(batch):
        assert output.state_mode == "converged"
        raise RuntimeError("deliberate evaluation failure")

    output.forward = failing_forward
    try:
        with pytest.raises(RuntimeError, match="deliberate evaluation failure"):
            evaluator.evaluate(data.datasets["test"])
        assert output.state_mode == "predicted" and model.training
    finally:
        output.forward = original


def test_evaluator_rejects_unsupported_or_ambiguous_state_mode(tmp_path):
    _, model, _, loss = joint_objects()
    with pytest.raises(ValueError, match="state_mode must"):
        VariationalEvaluator(model, loss, state_mode="exact")
    with pytest.raises(ValueError, match="predicted primary"):
        VariationalEvaluator(model, loss, state_mode="converged", compare_converged=True)
    evaluator = VariationalEvaluator(torch.nn.Linear(1, 1), loss, output_dir=tmp_path,
                                     state_mode="predicted")
    with pytest.raises(ValueError, match="does not support"):
        evaluator._predict({}, "predicted")


def test_joint_response_comparison_keeps_proposal_and_implicit_derivatives_distinct(tmp_path):
    _, model, data, loss = joint_objects()
    loss.response_weight = 1.0
    loss.response_scales = {"1/eV": 1.0}
    output = model.output_modules[0]
    assert not output.compute_responses
    evaluator = VariationalEvaluator(model, loss, output_dir=tmp_path, state_mode="predicted",
                                     compare_converged=True, compute_responses=True)
    metrics = evaluator.evaluate(data.datasets["test"])
    assert not output.compute_responses
    assert metrics["metadata"]["response_derivative"] == "proposal_jacobian"
    assert metrics["comparison"]["metadata"]["response_derivative"] == "implicit_stationary_response"
    assert metrics["terms"]["response"]["n_conditions"] == 4
    assert metrics["comparison"]["terms"]["response"]["n_conditions"] == 4
    delta = metrics["comparison"]["response_differences"]
    assert len(delta) == 1
    assert delta[0]["input_ref"] == "field" and delta[0]["output_ref"] == "variable_value"
    assert delta[0]["unit"] == "1/eV" and delta[0]["n_scalars"] == 64
    assert delta[0]["scalar_rmse"] > 0


def test_response_difference_matches_identity_instead_of_block_position():
    primary = {"variational_prediction": {"responses": {
        "values": torch.tensor([1., 2.]), "ptr": torch.tensor([0, 1, 2]),
        "condition_index": torch.tensor([0, 1]), "input_ref": ("field", "field"),
        "output_ref": ("variable_value", "variable_value"), "shape": ((1, 1), (1, 1)),
        "unit": ("1/eV", "1/eV"), "gauge": ({}, {}), "boundary_hash": ("a", "b"),
    }}}
    reference = copy.deepcopy(primary)
    blocks = reference["variational_prediction"]["responses"]
    blocks["values"] = blocks["values"].flip(0)
    blocks["condition_index"] = blocks["condition_index"].flip(0)
    blocks["boundary_hash"] = ("b", "a")
    differences = {}
    VariationalEvaluator._accumulate_response_difference(differences, primary, reference)
    assert len(differences) == 1
    assert next(iter(differences.values()))["sum_squared"] == 0
    blocks["gauge"] = ({"changed": True}, {})
    with pytest.raises(ValueError, match="response identities differ"):
        VariationalEvaluator._accumulate_response_difference({}, primary, reference)
