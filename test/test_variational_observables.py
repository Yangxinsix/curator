"""Analytic tests for configured derivatives of one scalar energy."""

import pytest
import torch

from curator.variational.observables import EnergyObservables, ObservableSpec


DTYPE = torch.float64


def test_quartic_proposal_energy_preserves_full_path_and_parameter_gradgrad():
    positions = torch.tensor([[0.2, -0.1, 0.3], [0.5, 0.4, -0.2]], dtype=DTYPE, requires_grad=True)
    field = torch.tensor([0.1, -0.2, 0.3], dtype=DTYPE, requires_grad=True)
    parameters = tuple(torch.tensor(value, dtype=DTYPE, requires_grad=True) for value in (0.4, 1.7, 0.6, 0.8))
    proposal, quadratic, quartic, baseline = parameters
    state = proposal * field + positions.sum(0)
    energy = (0.5 * baseline * positions.square().sum() + 0.5 * quadratic * state.square().sum()
              + 0.25 * quartic * state.pow(4).sum() - field @ state)
    executor = EnergyObservables(energy, {"positions": positions, "external_field": field}, create_graph=True)
    result = executor.evaluate([
        ObservableSpec("energy"),
        ObservableSpec("forces", ("positions",), factor=-1),
        ObservableSpec("dipole", ("external_field",), factor=-1),
        ObservableSpec("alpha", ("external_field", "external_field"), factor=-1),
        ObservableSpec("born", ("external_field", "positions"), factor=-1, permutation=(1, 0, 2)),
    ])
    residual = quadratic * state + quartic * state.pow(3) - field
    curvature = quadratic + 3 * quartic * state.square()
    expected = {
        "energy": energy,
        "forces": -baseline * positions - residual.unsqueeze(0),
        "dipole": state - proposal * residual,
        "alpha": torch.diag(2 * proposal - proposal.square() * curvature),
        "born": torch.diag(1 - proposal * curvature).expand(2, 3, 3),
    }
    for name in expected:
        torch.testing.assert_close(result[name], expected[name])
    # The observable is the derivative of assembled predicted-state energy;
    # simply reporting the internal state would miss the proposal path.
    assert not torch.allclose(result["dipole"], state)
    actual_loss = sum(value.square().sum() for value in result.values())
    expected_loss = sum(value.square().sum() for value in expected.values())
    actual_grads = torch.autograd.grad(actual_loss, parameters, create_graph=True, retain_graph=True)
    expected_grads = torch.autograd.grad(expected_loss, parameters, create_graph=True, retain_graph=True)
    for actual, target in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, target)
        assert torch.isfinite(actual) and actual.abs() > 0
    actual_second = torch.autograd.grad(sum(actual_grads), parameters, retain_graph=True)
    expected_second = torch.autograd.grad(sum(expected_grads), parameters)
    for actual, target in zip(actual_second, expected_second):
        torch.testing.assert_close(actual, target)


def test_mixed_axis_order_normalization_and_configuration_dicts():
    positions = torch.arange(6, dtype=DTYPE).reshape(2, 3).requires_grad_()
    field = torch.tensor([0.1, -0.2, 0.3], dtype=DTYPE, requires_grad=True)
    strain = (0.01 * torch.arange(9, dtype=DTYPE).reshape(3, 3)).requires_grad_()
    coupling = torch.arange(18, dtype=DTYPE).reshape(3, 6) / 10
    stiffness = torch.tensor([[2.0, 0.3, 0.2], [0.3, 3.0, -0.1], [0.2, -0.1, 4.0]], dtype=DTYPE)
    parameter = torch.tensor(1.3, dtype=DTYPE, requires_grad=True)
    volume = 3 + strain.sum()
    energy = -field @ coupling @ positions.flatten() - 0.5 * field @ stiffness @ field + parameter * strain.square().sum()
    executor = EnergyObservables(
        energy, {"positions": positions, "external_field": field, "strain": strain},
        create_graph=True, normalizers={"volume": volume},
    )
    result = executor.evaluate([
        {"name": "mixed", "derivatives": ["external_field", "positions"], "factor": -1, "permutation": [1, 0, 2]},
        {"name": "flat", "derivatives": ["external_field", "positions"], "factor": -1, "reshape": [3, -1]},
        {"name": "stress", "derivatives": ["strain"], "normalization": "volume", "unit": "eV/Angstrom^3"},
    ])
    torch.testing.assert_close(result["mixed"], coupling.reshape(3, 2, 3).permute(1, 0, 2))
    torch.testing.assert_close(result["flat"], coupling)
    torch.testing.assert_close(result["stress"], 2 * parameter * strain / volume)
    actual = torch.autograd.grad(result["stress"].sum(), strain, retain_graph=True)[0]
    expected = torch.autograd.grad((2 * parameter * strain / volume).sum(), strain)[0]
    torch.testing.assert_close(actual, expected)


def test_local_field_plus_independent_uniform_probe_uses_only_three_perturbations():
    local = torch.arange(15, dtype=DTYPE).reshape(5, 3) / 10
    probe = torch.tensor([0.2, -0.1, 0.4], dtype=DTYPE, requires_grad=True)
    live_field = local + probe
    energy = -0.5 * live_field.square().sum()
    executor = EnergyObservables(energy, {"uniform_probe": probe})
    result = executor.evaluate([
        ObservableSpec("dipole", ("uniform_probe",), factor=-1),
        ObservableSpec("alpha", ("uniform_probe", "uniform_probe"), factor=-1),
    ])
    torch.testing.assert_close(result["dipole"], live_field.sum(0))
    torch.testing.assert_close(result["alpha"], 5 * torch.eye(3, dtype=DTYPE))


def test_nine_component_strain_second_derivative_and_exact_live_view():
    strain = torch.linspace(0.1, 0.9, 9, dtype=DTYPE).reshape(3, 3).requires_grad_()
    probes = torch.zeros((2, 3), dtype=DTYPE, requires_grad=True)
    probe = probes[1]
    parameter = torch.tensor(1.4, dtype=DTYPE, requires_grad=True)
    energy = parameter * strain.pow(4).sum() - probe @ strain.sum(0)
    executor = EnergyObservables(
        energy, {"strain": strain, "field": probe}, create_graph=True, normalizers={"volume": 2.0},
    )
    result = executor.evaluate([
        ObservableSpec("curvature", ("strain", "strain"), reshape=(9, 9), normalization="volume"),
        ObservableSpec("vector", ("field",), factor=-1),
    ])
    torch.testing.assert_close(result["curvature"], torch.diag(6 * parameter * strain.flatten().square()))
    torch.testing.assert_close(result["vector"], strain.sum(0))
    gradient = torch.autograd.grad(result["curvature"].sum(), parameter)[0]
    torch.testing.assert_close(gradient, 6 * strain.square().sum())


def test_first_derivative_is_cached_and_alpha_does_not_consume_born_or_energy_graph(monkeypatch):
    positions = torch.ones((4, 3), dtype=DTYPE, requires_grad=True)
    field = torch.ones(3, dtype=DTYPE, requires_grad=True)
    parameter = torch.tensor(2.0, dtype=DTYPE, requires_grad=True)
    energy = parameter * (field.square().sum() + field @ positions.sum(0))
    original_grad = torch.autograd.grad
    calls = []

    def counted(value, inputs, *args, **kwargs):
        assert value.ndim == 0
        assert kwargs["retain_graph"] is True
        calls.append(inputs.numel())
        return original_grad(value, inputs, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("full Hessian/Jacobian API called")

    monkeypatch.setattr(torch.autograd, "grad", counted)
    monkeypatch.setattr(torch.autograd.functional, "hessian", forbidden)
    monkeypatch.setattr(torch.autograd.functional, "jacobian", forbidden)
    executor = EnergyObservables(energy, {"positions": positions, "external_field": field})
    with torch.no_grad():
        result = executor.evaluate([
            ObservableSpec("dipole", ("external_field",)),
            ObservableSpec("alpha", ("external_field", "external_field")),
            ObservableSpec("born", ("external_field", "positions")),
            ObservableSpec("forces", ("positions",)),
        ])
    assert calls == [3, 3, 3, 3, 12, 12, 12, 12]
    assert all(not value.requires_grad for value in result.values())
    torch.testing.assert_close(result["born"], parameter.detach() * torch.eye(3, dtype=DTYPE).unsqueeze(1).expand(3, 4, 3))
    energy.backward()
    torch.testing.assert_close(parameter.grad, torch.tensor(15.0, dtype=DTYPE))


@pytest.mark.parametrize("kind", ["constant", "unrelated_parameter", "affine"])
def test_zero_unused_and_affine_derivatives_support_backward(kind):
    point = torch.tensor([0.2, -0.4, 0.7], dtype=DTYPE, requires_grad=True)
    parameter = torch.tensor(4.0, dtype=DTYPE, requires_grad=True)
    energy = point.new_tensor(2.0) if kind == "constant" else parameter
    if kind == "affine":
        energy = energy + 3 * point.sum()
    executor = EnergyObservables(energy, {"x": point}, create_graph=True)
    result = executor.evaluate([ObservableSpec("gradient", ("x",)), ObservableSpec("curvature", ("x", "x"))])
    torch.testing.assert_close(result["gradient"], torch.full_like(point, 3 if kind == "affine" else 0))
    torch.testing.assert_close(result["curvature"], torch.zeros((3, 3), dtype=DTYPE))
    (result["gradient"].square().sum() + result["curvature"].sum()).backward()
    torch.testing.assert_close(point.grad, torch.zeros_like(point))
    assert parameter.grad is None


def test_large_coordinate_hvp_never_builds_full_hessian_and_handles_live_direction(monkeypatch):
    point = torch.linspace(-0.5, 0.7, 300, dtype=DTYPE).reshape(100, 3).requires_grad_()
    parameter = torch.tensor(1.7, dtype=DTYPE, requires_grad=True)
    executor = EnergyObservables(parameter * point.pow(3).sum(), {"positions": point}, create_graph=True)
    with pytest.raises(ValueError, match="request hvp"):
        executor.derivative(("positions", "positions"))

    def forbidden(*args, **kwargs):
        raise AssertionError("dense Hessian called")

    monkeypatch.setattr(torch.autograd.functional, "hessian", forbidden)
    direction = point.square()
    product = executor.hvp("positions", direction)
    torch.testing.assert_close(product, 6 * parameter * point.pow(3))
    actual = torch.autograd.grad(product.sum(), (point, parameter))
    torch.testing.assert_close(actual[0], 18 * parameter * point.square())
    torch.testing.assert_close(actual[1], 6 * point.pow(3).sum())


def test_converged_detached_state_refuses_unrelaxed_second_derivatives():
    point = torch.ones(3, dtype=DTYPE, requires_grad=True)
    executor = EnergyObservables(point.square().sum(), {"x": point}, state_mode="converged")
    torch.testing.assert_close(executor.derivative(("x",)), 2 * point)
    with pytest.raises(ValueError, match="implicit linear response"):
        executor.derivative(("x", "x"))
    with pytest.raises(ValueError, match="implicit linear response"):
        executor.derivative(("x",), create_graph=True)
    with pytest.raises(ValueError, match="implicit linear response"):
        executor.hvp("x", point)


@pytest.mark.parametrize("arguments", [
    {"name": ""}, {"name": "x", "derivatives": "positions"},
    {"name": "x", "derivatives": ("x", "x", "x")},
    {"name": "x", "derivatives": (3,)}, {"name": "x", "factor": float("nan")},
    {"name": "x", "factor": True}, {"name": "x", "permutation": (0, 0)},
    {"name": "x", "reshape": (-1, -1)}, {"name": "x", "reshape": (0,)},
    {"name": "x", "normalization": ""}, {"name": "x", "unit": 3},
])
def test_invalid_specifications_are_rejected(arguments):
    with pytest.raises((ValueError, TypeError)):
        ObservableSpec(**arguments)


def test_invalid_live_inputs_output_axes_and_normalizers_are_rejected():
    point = torch.ones(3, dtype=DTYPE, requires_grad=True)
    executor = EnergyObservables(point.square().sum(), {"x": point, "frozen": point.detach()}, normalizers={"zero": 0, "many": point, "nan": float("nan")})
    for name in ("missing", "frozen"):
        with pytest.raises(ValueError):
            executor.derivative((name,))
    with pytest.raises(ValueError, match="unique"):
        executor.evaluate([ObservableSpec("x"), ObservableSpec("x")])
    with pytest.raises(ValueError, match="rank"):
        executor.evaluate([ObservableSpec("x", ("x",), permutation=(1, 0))])
    with pytest.raises(ValueError, match="size"):
        executor.evaluate([ObservableSpec("x", ("x",), reshape=(2, 2))])
    for normalization in ("zero", "many", "nan", "missing"):
        with pytest.raises(ValueError, match="normalization"):
            executor.evaluate([ObservableSpec("x", normalization=normalization)])
    with pytest.raises(ValueError, match="scalar"):
        EnergyObservables(point, {"x": point})
    with pytest.raises(ValueError, match="shape"):
        executor.hvp("x", point.reshape(1, 3))
    with pytest.raises(ValueError, match="dtype"):
        executor.hvp("x", point.float())
