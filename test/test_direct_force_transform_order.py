import torch

from curator.data import properties
from curator.data.properties import HeadConfig
from curator.layer import (
    DirectForceOutput,
    EnergyHessianOutput,
    GlobalRescaleShift,
    GradientOutput,
)
from curator.model import NeuralNetworkPotential, Painn
from curator.model.conversion import transform_model_to_direct_force


class _Marker(torch.nn.Module):
    def forward(self, data):
        return data


def _energy_rescale():
    return GlobalRescaleShift(
        heads=[
            HeadConfig(
                key=properties.energy,
                scale_by=3.0,
                shift_by=7.0,
            )
        ]
    )


def test_nnp_preserves_configured_output_module_order():
    modules = [
        _energy_rescale(),
        GradientOutput(grad_on_edge_diff=False, grad_on_positions=True),
        EnergyHessianOutput(),
        _Marker(),
    ]

    model = NeuralNetworkPotential(
        representation=Painn(1, 8, 5.0),
        output_modules=modules,
    )

    assert list(model.output_modules) == modules


def test_direct_transform_replaces_gradient_without_changing_rescale_or_order():
    source_rescale = _energy_rescale()
    source = NeuralNetworkPotential(
        representation=Painn(1, 8, 5.0),
        output_modules=[
            _Marker(),
            source_rescale,
            GradientOutput(grad_on_edge_diff=False, grad_on_positions=True),
            EnergyHessianOutput(),
            _Marker(),
        ],
    )
    source._initialized = True
    expected_rescale_state = {
        key: value.detach().clone() for key, value in source_rescale.state_dict().items()
    }

    transformed = transform_model_to_direct_force(source)

    assert [type(module) for module in transformed.output_modules] == [
        _Marker,
        GlobalRescaleShift,
        DirectForceOutput,
        EnergyHessianOutput,
        _Marker,
    ]
    transformed_rescale = transformed.output_modules[1]
    assert transformed_rescale is not source_rescale
    assert [head.key for head in transformed_rescale.heads] == [properties.energy]
    assert transformed_rescale.state_dict().keys() == expected_rescale_state.keys()
    for key, value in expected_rescale_state.items():
        torch.testing.assert_close(transformed_rescale.state_dict()[key], value)
    assert transformed._initialized is True


def test_clone_rebinds_direct_head_and_preserves_compatible_weights():
    source = NeuralNetworkPotential(
        representation=Painn(1, 8, 5.0),
        output_modules=[DirectForceOutput()],
    )
    source_head = source.output_modules[0].head
    with torch.no_grad():
        for parameter in source_head.parameters():
            parameter.fill_(0.125)

    cloned = source.clone_with_representation(Painn(1, 8, 5.0))
    cloned_output = cloned.output_modules[0]

    assert isinstance(cloned_output, DirectForceOutput)
    assert cloned_output.is_bound
    assert cloned_output.head is not source_head
    for key, value in source_head.state_dict().items():
        torch.testing.assert_close(cloned_output.head.state_dict()[key], value)
