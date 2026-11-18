"""Independent geometries share one coordinate derivative without losing training graphs."""

import copy
import io

import pytest
import torch

from curator.data import properties
from test_acvr_deployment import _four_geometries
from test_acvr_joint_model import _model, _prediction


def _assert_predictions_equal(first, second):
    for name in ("condition_energy", "condition_forces", "variable_value"):
        if name in first["labels"]:
            expected, actual = first["labels"][name], second["labels"][name]
            assert actual["shape"] == expected["shape"]
            torch.testing.assert_close(actual["ptr"], expected["ptr"])
            torch.testing.assert_close(actual["values"], expected["values"], rtol=2e-10, atol=2e-12)
    if first["residuals"]:
        torch.testing.assert_close(second["residuals"]["stationarity"]["values"],
                                   first["residuals"]["stationarity"]["values"],
                                   rtol=2e-10, atol=2e-12)


def _record_force_calls(monkeypatch):
    calls = []
    original_grad = torch.autograd.grad

    def record(*args, **kwargs):
        inputs = args[1] if len(args) > 1 else kwargs["inputs"]
        if isinstance(inputs, torch.Tensor) and inputs.shape == (10, 3):
            calls.append(kwargs.get("create_graph", False))
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", record)
    return calls


@pytest.mark.parametrize("backbone", ["painn", "mace", "qefa"])
@pytest.mark.parametrize("force_only", [False, True])
def test_training_batches_coordinate_derivatives_and_preserves_every_parameter_gradient(
        backbone, force_only, monkeypatch):
    if backbone == "qefa":
        from test_efa_fixed_charge_integration import model
        ordinary = model()
    else:
        ordinary = _model(backbone=backbone)
    ordinary.output_modules[0].batch_observables = False
    batched = copy.deepcopy(ordinary)
    batched.output_modules[0].batch_observables = True
    calls = _record_force_calls(monkeypatch)
    predictions = []
    for network, count in ((ordinary, 4), (batched, 1)):
        calls.clear()
        prediction = _prediction(network.train(), _four_geometries())
        predictions.append(prediction)
        assert calls == [True] * count
        assert prediction["metadata"]["batched_force_derivative"] is (count == 1)
        loss = (prediction["labels"]["condition_forces"]["values"] - 0.13).square().mean()
        if not force_only:
            loss = loss + prediction["labels"]["condition_energy"]["values"].square().mean()
            loss = loss + prediction["residuals"]["stationarity"]["values"].square().mean()
        loss.backward()
        for group in network.parameter_groups():
            gradients = [parameter.grad for parameter in group.params if parameter.grad is not None]
            assert gradients and all(torch.isfinite(value).all() for value in gradients), group.name
            assert sum(float(value.abs().sum()) for value in gradients) > 0, group.name
    _assert_predictions_equal(*predictions)
    for (name, expected), (other, actual) in zip(ordinary.named_parameters(), batched.named_parameters()):
        assert name == other
        assert (expected.grad is None) == (actual.grad is None), name
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-9, atol=2e-11, msg=name)


@pytest.mark.parametrize("order", [[3, 0, 2, 1], [2, 0], [0, 0, 2, 3], [1]])
def test_evaluation_preserves_condition_order_subset_and_repeated_geometry_fallback(order, monkeypatch):
    network = _model().eval()
    head = network.output_modules[0]
    _, boundaries = head._conditions(_four_geometries())
    # Repeated geometry 0 deliberately receives different boundary conditions.
    selected = tuple(boundaries[index] for index in range(len(order)))
    monkeypatch.setattr(head, "_conditions", lambda data: (torch.tensor(order), selected))
    calls = _record_force_calls(monkeypatch)
    expected = _prediction(network, _four_geometries())
    assert len(calls) == len(order)
    head.batch_observables = True
    calls.clear()
    actual = _prediction(network, _four_geometries())
    batched = len(order) > 1 and len(set(order)) == len(order)
    assert calls == [False] * (1 if batched else len(order))
    assert actual["metadata"]["batched_force_derivative"] is batched
    _assert_predictions_equal(expected, actual)


def test_evaluation_retains_coordinate_second_derivatives_when_requested():
    network = _model(retain_response_graph=True).eval()
    derivatives = []
    for enabled in (False, True):
        network.output_modules[0].batch_observables = enabled
        data = _four_geometries()
        prediction = _prediction(network, data)
        force = prediction["labels"]["condition_forces"]["values"]
        derivatives.append(torch.autograd.grad(force.square().sum(), data[properties.positions])[0])
    assert torch.isfinite(derivatives[1]).all() and derivatives[1].abs().sum() > 0
    torch.testing.assert_close(*derivatives, rtol=2e-9, atol=2e-11)


@pytest.mark.parametrize("enabled", [False, True])
def test_deployment_retains_explicit_batch_force_opt_in(enabled, monkeypatch):
    network = _model(batch_observables=True).eval()
    expected = _prediction(network, _four_geometries())
    network.output_modules[0].configure_deployment(compute_residuals=True, batch_forces=enabled)
    calls = _record_force_calls(monkeypatch)
    actual = _prediction(network, _four_geometries())
    assert calls == [False] * (1 if enabled else 4)
    assert actual["metadata"]["batched_force_derivative"] is enabled
    _assert_predictions_equal(expected, actual)


@pytest.mark.parametrize("mode", ["converged", "no_forces", "energy_only"])
def test_unsupported_or_disabled_force_paths_remain_unbatched(mode, monkeypatch):
    network = _model(batch_observables=True).eval()
    head = network.output_modules[0]
    if mode == "converged":
        head.state_mode = mode
    elif mode == "no_forces":
        head.compute_forces = False
    else:
        head.configure_deployment(energy_only=True, batch_forces=True)
    calls = _record_force_calls(monkeypatch)
    actual = _prediction(network, _four_geometries())
    assert actual["metadata"]["batched_force_derivative"] is False
    if mode == "converged":
        assert calls == [False] * 4
        assert actual["metadata"]["force_derivative"] == "stationary_envelope"
        assert all(entry["converged"] and entry["stable"] for entry in actual["diagnostics"])
    else:
        assert calls == []
        assert "condition_forces" not in actual["labels"]


def test_old_whole_model_checkpoint_defaults_to_explicit_unbatched_contract():
    network = _model().eval()
    expected = _prediction(network, _four_geometries())
    del network.output_modules[0].batch_observables
    serialized = io.BytesIO()
    torch.save(network, serialized)
    serialized.seek(0)
    restored = torch.load(serialized, weights_only=False)
    assert restored.output_modules[0].batch_observables is False
    actual = _prediction(restored, _four_geometries())
    assert actual["metadata"]["batched_force_derivative"] is False
    _assert_predictions_equal(expected, actual)
