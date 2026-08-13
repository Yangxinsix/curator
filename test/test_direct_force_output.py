import copy

import pytest
import torch

from curator.data import properties
from curator.layer import (
    DirectForceOutput,
    EnergyHessianOutput,
    GatedEquivariantBlock,
    GradientOutput,
    PairwiseDistance,
    ScaledSiLU,
)
from curator.model import MACE, NeuralNetworkPotential, Painn
from curator.model.conversion import transform_model_to_direct_force
from curator.model.checkpoint_upgrade import _upgrade_legacy_checkpoint_model


def _batch(positions):
    return {
        properties.Z: torch.tensor([1, 6, 8]),
        properties.positions: positions,
        properties.edge_idx: torch.tensor(
            [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]]
        ),
        properties.n_atoms: torch.tensor([3]),
        properties.image_idx: torch.tensor([0, 0, 0]),
    }


def _painn_direct(*extra_outputs):
    return NeuralNetworkPotential(
        representation=Painn(
            num_interactions=2,
            num_features=16,
            cutoff=5.0,
        ),
        input_modules=[
            PairwiseDistance(
                compute_distance_from_R=True,
                compute_forces=True,
            )
        ],
        output_modules=[DirectForceOutput(), *extra_outputs],
        model_outputs=[properties.energy, properties.forces],
    )


def test_gated_equivariant_block_uses_separate_equivariant_projections():
    torch.manual_seed(3)
    block = GatedEquivariantBlock(
        scalar_in=4,
        vector_in=5,
        scalar_out=3,
        vector_out=2,
        invariant_channels=6,
        hidden_channels=7,
        activation=torch.nn.Tanh(),
        scalar_activation=torch.nn.Softplus(),
    )
    scalars = torch.randn(8, 4)
    vectors = torch.randn(8, 3, 5)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    scalar_out, vector_out = block(scalars, vectors)
    rotated_scalar_out, rotated_vector_out = block(
        scalars,
        torch.einsum("ij,njc->nic", rotation, vectors),
    )

    assert block.invariant_vector_projection.weight.shape == (6, 5)
    assert block.output_vector_projection.weight.shape == (2, 5)
    assert scalar_out.shape == (8, 3)
    assert vector_out.shape == (8, 3, 2)
    assert torch.all(scalar_out > 0)
    torch.testing.assert_close(rotated_scalar_out, scalar_out)
    torch.testing.assert_close(
        rotated_vector_out,
        torch.einsum("ij,njc->nic", rotation, vector_out),
    )


def test_gated_equivariant_block_defaults_preserve_force_head_widths():
    block = GatedEquivariantBlock(8, 8, 1, 1)

    assert block.invariant_vector_projection.out_features == 1
    assert block.output_vector_projection.out_features == 1
    assert block.scalar_net[0].in_features == 9
    assert block.scalar_net[0].out_features == 2


def test_direct_force_head_accepts_per_block_components():
    model = NeuralNetworkPotential(
        representation=Painn(1, 8, 5.0),
        output_modules=[
            DirectForceOutput(
                hidden_channels=4,
                block_kwargs=[
                    {
                        "invariant_channels": 8,
                        "hidden_channels": 8,
                        "activation": ScaledSiLU(),
                        "scalar_activation": ScaledSiLU(),
                        "norm_eps": 0.0,
                    },
                    {
                        "invariant_channels": 4,
                        "hidden_channels": 4,
                        "activation": ScaledSiLU(),
                        "scalar_activation": ScaledSiLU(),
                        "norm_eps": 0.0,
                    },
                ],
            )
        ],
    )
    blocks = model.output_modules[0].head.blocks

    assert blocks[0].invariant_vector_projection.out_features == 8
    assert blocks[1].scalar_net[0].out_features == 4
    assert isinstance(blocks[0].scalar_net[1], ScaledSiLU)
    assert isinstance(blocks[1].scalar_activation, ScaledSiLU)


def test_painn_direct_force_is_rotation_equivariant_and_trainable():
    torch.manual_seed(7)
    model = _painn_direct()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.1, 1.0, 0.3]]
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    prediction = model(_batch(positions))
    rotated = model(_batch(positions @ rotation.T))

    torch.testing.assert_close(rotated[properties.energy], prediction[properties.energy])
    torch.testing.assert_close(
        rotated[properties.forces],
        prediction[properties.forces] @ rotation.T,
        atol=2e-6,
        rtol=2e-6,
    )

    prediction[properties.forces].square().mean().backward()
    force_head = model.output_modules[0].head
    assert any(parameter.grad is not None for parameter in force_head.parameters())


def test_direct_force_hessian_uses_force_jacobian_and_reaches_head():
    model = _painn_direct(
        EnergyHessianOutput(
            vectorize=False,
            num_probes=2,
            model_outputs=[properties.energy_hessian_projected],
        )
    )
    model.model_outputs.append(properties.energy_hessian_projected)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.1, 1.0, 0.3]]
    )

    prediction = model(_batch(positions))
    projected = prediction[properties.energy_hessian_projected]
    assert projected.shape == (2, 3, 3)
    projected.square().mean().backward()
    force_head = model.output_modules[0].head
    assert any(parameter.grad is not None for parameter in force_head.parameters())


def test_model_rejects_multiple_force_producers():
    with pytest.raises(ValueError, match="only one force-producing output"):
        NeuralNetworkPotential(
            representation=Painn(2, 16, 5.0),
            output_modules=[GradientOutput(), DirectForceOutput()],
        )


def test_gradient_output_updates_force_producer_marker():
    output = GradientOutput(model_outputs=[properties.virial])
    assert output.produces_forces is False

    output.update_model_outputs(properties.forces)

    assert output.produces_forces is True


def test_checkpoint_transform_preserves_representation_and_adds_output_head():
    source = NeuralNetworkPotential(
        representation=Painn(2, 16, 5.0),
        input_modules=[PairwiseDistance()],
        output_modules=[GradientOutput()],
        model_outputs=[properties.energy, properties.forces],
    )
    source._initialized = True
    source_state = copy.deepcopy(source.representation.state_dict())

    transformed = transform_model_to_direct_force(source)

    assert isinstance(transformed.output_modules[0], DirectForceOutput)
    for key, value in source_state.items():
        torch.testing.assert_close(transformed.representation.state_dict()[key], value)
    assert transformed._initialized is True


def test_direct_force_model_is_scriptable_after_binding():
    model = _painn_direct()
    scripted = torch.jit.script(model)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.1, 1.0, 0.3]]
    )
    assert scripted(_batch(positions))[properties.forces].shape == (3, 3)


def test_legacy_painn_vector_biases_are_removed_before_direct_transform():
    source = NeuralNetworkPotential(
        representation=Painn(1, 8, 5.0),
        input_modules=[PairwiseDistance()],
        output_modules=[GradientOutput()],
    )
    update = source.representation.update_layers[0]
    update.update_U.bias = torch.nn.Parameter(torch.ones(8))
    update.update_V.bias = torch.nn.Parameter(torch.ones(8))

    _upgrade_legacy_checkpoint_model(source)

    assert update.update_U.bias is None
    assert update.update_V.bias is None


def test_mace_direct_force_uses_irreps_feature_adapter():
    torch.manual_seed(11)
    model = NeuralNetworkPotential(
        representation=MACE(
            cutoff=5.0,
            num_interactions=2,
            correlation=2,
            species=["H", "C", "O"],
            lmax=1,
            parity=True,
            num_features=4,
            num_basis=4,
            avg_num_neighbors=2.0,
        ),
        input_modules=[PairwiseDistance(compute_distance_from_R=True)],
        output_modules=[DirectForceOutput()],
        model_outputs=[properties.energy, properties.forces],
    )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.1, 1.0, 0.3]]
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    prediction = model(_batch(positions))
    rotated = model(_batch(positions @ rotation.T))

    assert prediction[properties.energy].shape == (1,)
    assert prediction[properties.forces].shape == (3, 3)
    torch.testing.assert_close(rotated[properties.energy], prediction[properties.energy])
    torch.testing.assert_close(
        rotated[properties.forces],
        prediction[properties.forces] @ rotation.T,
        atol=2e-5,
        rtol=2e-5,
    )
