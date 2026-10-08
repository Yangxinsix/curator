"""Native readout Jacobians and forward/backward factor alignment."""
import pytest
import torch
from torch import nn

from curator.data import properties
from curator.layer.feature.extractor import FeatureExtractor
from curator.layer.feature.readout import parameter_gradient_factors, readout_parameter_layout


@pytest.mark.parametrize("bias", [False, True])
def test_torch_linear_factors_equal_native_parameter_derivatives(bias):
    layer = nn.Linear(4, 3, bias=bias).double()
    x = torch.tensor([[0.2, -0.4, 1.1, 0.7], [0.9, 0.3, -0.2, 0.5]], dtype=torch.float64)
    output = layer(x)
    atomic_energy = output.sin().sum(dim=1)
    adjoint = torch.autograd.grad(atomic_energy.sum(), output, retain_graph=True)[0]
    captured = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
    layout = readout_parameter_layout(layer)
    (a, b), = parameter_gradient_factors([captured], [adjoint], [layout])
    factors = torch.einsum("ni,nj->nij", a, b).flatten(1)
    for atom in range(len(x)):
        native = torch.autograd.grad(atomic_energy[atom], tuple(layer.parameters()), retain_graph=True)
        expected = native[0].T.flatten()
        if bias:
            expected = torch.cat((expected, native[1]))
        torch.testing.assert_close(factors[atom], expected, rtol=1e-14, atol=1e-14)
    assert factors.shape[1] == sum(p.numel() for p in layer.parameters())


def test_scalar_e3nn_factors_remove_virtual_bias_and_apply_path_coefficient():
    o3 = pytest.importorskip("e3nn.o3")
    layer = o3.Linear("4x0e", "3x0e", biases=False).double()
    x = torch.tensor([[0.2, -0.4, 1.1, 0.7], [0.9, 0.3, -0.2, 0.5]], dtype=torch.float64)
    output = layer(x)
    atomic_energy = output.square().sum(dim=1)
    adjoint = torch.autograd.grad(atomic_energy.sum(), output, retain_graph=True)[0]
    captured = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
    layout = readout_parameter_layout(layer)
    assert layout["has_bias"] is False
    assert layout["weight_scale"] == layer.instructions[0].path_weight
    (a, b), = parameter_gradient_factors([captured], [adjoint], [layout])
    factors = torch.einsum("ni,nj->nij", a, b).flatten(1)
    for atom in range(len(x)):
        native, = torch.autograd.grad(atomic_energy[atom], layer.weight, retain_graph=True)
        torch.testing.assert_close(factors[atom], native, rtol=1e-14, atol=1e-14)
    assert factors.shape[1] == 12


def test_extractor_aligns_independent_backwards_with_actual_forward_execution():
    class Branches(nn.Module):
        def __init__(self):
            super().__init__()
            self.readout = nn.ModuleDict({"first": nn.Linear(2, 2), "second": nn.Linear(2, 2)})

    model = Branches().double()
    extractor = FeatureExtractor(model)
    x = torch.tensor([[0.2, 0.7], [-0.5, 1.2]], dtype=torch.float64)
    second = model.readout["second"](2 * x)
    first = model.readout["first"](x)
    # Backward visits second then first, deliberately not reverse-forward order.
    second.square().sum().backward()
    first.sin().sum().backward()
    data = extractor({})
    assert [layout["module_name"] for layout in data["readout_layouts"]] == ["second", "first"]
    torch.testing.assert_close(data[properties.feature][0][:, :-1], 2 * x)
    torch.testing.assert_close(data[properties.gradient][0], 2 * second)
    torch.testing.assert_close(data[properties.gradient][1], first.cos())
    assert extractor._features == []
    assert extractor._readout_layouts == []
    extractor.unhook()


def test_unsupported_equivariant_layout_does_not_break_feature_extraction():
    o3 = pytest.importorskip("e3nn.o3")
    model = nn.Module()
    model.readout = o3.Linear("1x1o", "1x1o")
    extractor = FeatureExtractor(model)
    output = model.readout(torch.ones(2, 3))
    output.sum().backward()
    data = extractor({})
    assert data[properties.feature][0].shape == (2, 4)
    assert data[properties.gradient][0].shape == (2, 3)
    assert not data["readout_layouts"][0]["supported"]
    with pytest.raises(ValueError, match="Unsupported readout parameter layout"):
        parameter_gradient_factors(data[properties.feature], data[properties.gradient], data["readout_layouts"])
    extractor.unhook()


def test_reused_readout_parameters_are_rejected_only_by_joint_conversion():
    model = nn.Module()
    model.readout = nn.Linear(2, 1)
    extractor = FeatureExtractor(model)
    output = model.readout(torch.ones(2, 2)) + model.readout(torch.zeros(2, 2))
    output.sum().backward()
    data = extractor({})
    assert len(data[properties.feature]) == len(data[properties.gradient]) == 2
    with pytest.raises(ValueError, match="Repeated/shared readout parameters"):
        parameter_gradient_factors(data[properties.feature], data[properties.gradient], data["readout_layouts"])
    extractor.unhook()


def test_joint_conversion_rejects_missing_or_misaligned_factors():
    layout = readout_parameter_layout(nn.Linear(2, 1))
    a, b = torch.ones(3, 3), torch.ones(3, 1)
    with pytest.raises(ValueError, match="requires readout_layouts"):
        parameter_gradient_factors([a], [b], None)
    with pytest.raises(ValueError, match="same nonzero length"):
        parameter_gradient_factors([a], [], [layout])
    with pytest.raises(ValueError, match="widths"):
        parameter_gradient_factors([a[:, :2]], [b], [layout])
    with pytest.raises(ValueError, match="atom population"):
        parameter_gradient_factors([a], [b[:2]], [layout])
    with pytest.raises(ValueError, match="constant bias"):
        parameter_gradient_factors([torch.zeros_like(a)], [b], [layout])
