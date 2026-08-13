from collections import OrderedDict

import pytest
import torch
from torch import nn

from curator.layer import VarianceScale
from curator.train import (
    ensure_variance_scales_fitted,
    fit_variance_scales,
)


class _ScaledChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = VarianceScale()
        self.second = VarianceScale()

    def forward(self, x):
        return self.second(3.0 * self.first(2.0 * x))


def test_variance_scales_fit_in_forward_order():
    model = _ScaledChain()
    batches = [
        torch.tensor([[-1.0], [1.0]]),
        torch.tensor([[-1.0], [1.0]]),
    ]

    fitted = fit_variance_scales(model, batches)

    assert isinstance(fitted, OrderedDict)
    assert list(fitted) == ["first", "second"]
    assert fitted["first"] == pytest.approx(0.5)
    assert fitted["second"] == pytest.approx(1.0 / 3.0)
    ensure_variance_scales_fitted(model)
    torch.testing.assert_close(model(batches[0]), batches[0])


def test_variance_scale_validation_and_fixed_scale_preservation():
    model = _ScaledChain()
    model.first.set_scale(0.25)

    with pytest.raises(RuntimeError, match="second"):
        ensure_variance_scales_fitted(model)

    fitted = fit_variance_scales(
        model,
        [torch.tensor([[-1.0], [1.0]])],
        reset=False,
    )

    assert fitted["first"] == pytest.approx(0.25)
    assert bool(model.second.fitted.item())


def test_variance_scale_fitter_rejects_unexecuted_modules():
    class Conditional(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = VarianceScale()
            self.unused = VarianceScale()

        def forward(self, x):
            return self.used(x)

    with pytest.raises(RuntimeError, match="unused"):
        fit_variance_scales(
            Conditional(),
            [torch.tensor([[-1.0], [1.0]])],
        )
