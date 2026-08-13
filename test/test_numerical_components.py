import math

import pytest
import torch

from curator.layer import (
    ResidualAdd,
    ScaledSiLU,
    VarianceScale,
    reset_linear,
    safe_norm,
)


def test_scaled_silu_applies_fixed_scale():
    x = torch.linspace(-2.0, 2.0, 9)
    torch.testing.assert_close(ScaledSiLU()(x), torch.nn.functional.silu(x) / 0.6)


def test_safe_norm_has_finite_first_and_second_derivatives_at_zero():
    x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    value = safe_norm(x, dim=0, eps=1e-8)
    first = torch.autograd.grad(value, x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]

    assert torch.isfinite(value)
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    torch.testing.assert_close(first, torch.zeros_like(first))


def test_safe_norm_without_epsilon_has_finite_first_derivative_at_zero():
    x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    value = safe_norm(x, dim=0, eps=0.0)
    first = torch.autograd.grad(value, x)[0]

    torch.testing.assert_close(value, torch.zeros_like(value))
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, torch.zeros_like(first))


def test_residual_add_scales_sum():
    layer = ResidualAdd(1.0 / math.sqrt(2.0))
    actual = layer(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(actual, torch.tensor([4.0, 6.0]) / math.sqrt(2.0))


def test_reset_linear_applies_xavier_weights_and_zero_bias():
    torch.manual_seed(7)
    linear = torch.nn.Linear(5, 3)
    reset_linear(linear)

    torch.testing.assert_close(linear.bias, torch.zeros_like(linear.bias))
    assert linear.weight.abs().max() <= math.sqrt(6.0 / 8.0)


def test_variance_scale_observes_streamed_batches_and_fits_unit_variance():
    layer = VarianceScale()
    x = torch.tensor([[-1.0, -2.0], [1.0, 2.0]])

    layer(x)
    with pytest.raises(RuntimeError, match="did not observe"):
        layer.fit()

    layer.start_fitting()
    torch.testing.assert_close(layer(x[:1]), x[:1])
    torch.testing.assert_close(layer(x[1:]), x[1:])
    expected = math.sqrt(1.0 / 2.5)
    assert layer.fit() == pytest.approx(expected)
    layer.stop_fitting()

    assert bool(layer.fitted)
    torch.testing.assert_close(layer(x), x * expected)


def test_variance_scale_fits_against_reference_variance():
    layer = VarianceScale()
    ref = torch.tensor([[-1.0, -2.0], [1.0, 2.0]])
    x = 2.0 * ref

    layer.start_fitting()
    layer(x[:1], ref=ref[:1])
    layer(x[1:], ref=ref[1:])
    assert layer.fit() == pytest.approx(0.5)
    layer.stop_fitting()


def test_variance_scale_state_dict_preserves_scale_and_fitted_state():
    source = VarianceScale()
    source.set_scale(1.75)
    restored = VarianceScale()
    restored.load_state_dict(source.state_dict())

    assert bool(restored.fitted)
    torch.testing.assert_close(restored.scale, torch.tensor(1.75))
    torch.testing.assert_close(restored(torch.tensor([2.0])), torch.tensor([3.5]))

    restored.reset()
    assert not bool(restored.fitted)
    torch.testing.assert_close(restored.scale, torch.tensor(1.0))
    torch.testing.assert_close(restored(torch.tensor([2.0])), torch.tensor([2.0]))


def test_numerical_modules_are_scriptable():
    x = torch.tensor([1.0])
    torch.testing.assert_close(torch.jit.script(ScaledSiLU())(x), ScaledSiLU()(x))
    torch.testing.assert_close(
        torch.jit.script(ResidualAdd())(x, x),
        ResidualAdd()(x, x),
    )
    torch.testing.assert_close(
        torch.jit.script(VarianceScale(2.0))(x),
        VarianceScale(2.0)(x),
    )
    torch.testing.assert_close(
        torch.jit.script(safe_norm)(x, 0, False, 1e-8),
        safe_norm(x, 0, False, 1e-8),
    )
