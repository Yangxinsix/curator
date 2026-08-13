import math

import torch
from torch import nn

from curator.data import properties
from curator.layer import (
    PairwiseDistance,
    ResidualAdd,
    ScaledSiLU,
    VarianceScale,
)
from curator.model import NeuralNetworkPotential, Painn
from e3nn import o3


class _CountingBasis(nn.Module):
    def __init__(self, num_basis):
        super().__init__()
        self.num_basis = num_basis
        self.irreps_out = o3.Irreps(f"{num_basis}x0e")
        self.calls = 0

    def forward(self, distances):
        self.calls += 1
        return distances.unsqueeze(-1).expand(-1, self.num_basis)


def _batch():
    return {
        properties.positions: torch.tensor(
            [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.1, 1.0, 0.3]]
        ),
        properties.Z: torch.tensor([1, 6, 8]),
        properties.n_atoms: torch.tensor([3]),
        properties.cell: torch.eye(3).unsqueeze(0),
        properties.edge_idx: torch.tensor(
            [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]]
        ),
    }


def test_painn_computes_shared_radial_encoding_once():
    basis = _CountingBasis(5)
    model = NeuralNetworkPotential(
        representation=Painn(
            num_interactions=3,
            num_features=8,
            cutoff=5.0,
            num_basis=5,
            radial_basis=basis,
        ),
        input_modules=[PairwiseDistance(compute_distance_from_R=True)],
        model_outputs=[properties.energy],
    )

    prediction = model(_batch())

    assert prediction[properties.energy].shape == (1,)
    assert basis.calls == 1


def test_painn_composes_optional_engineering_components():
    representation = Painn(
        num_interactions=2,
        num_features=8,
        cutoff=5.0,
        activation=ScaledSiLU,
        scalar_norm=nn.LayerNorm,
        message_residual_scale=1 / math.sqrt(2),
        state_vector_scale=1 / math.sqrt(3),
        message_vector_scale=1 / math.sqrt(8),
        inner_product_scale=1 / math.sqrt(8),
        scalar_update_scale=1 / math.sqrt(2),
        norm_eps=1e-8,
        vector_bias=False,
        layer_scales=[None, 1.25],
    )

    assert isinstance(representation.message_layers[0].scalar_norm, nn.LayerNorm)
    assert isinstance(representation.message_layers[0].scalar_message_mlp[1], ScaledSiLU)
    assert isinstance(representation.message_residuals[0], ResidualAdd)
    assert representation.message_residuals[0].scale.item() == torch.tensor(
        1 / math.sqrt(2)
    ).item()
    assert isinstance(representation.layer_scales[0], VarianceScale)
    assert not bool(representation.layer_scales[0].fitted.item())
    assert representation.layer_scales[1].scale.item() == 1.25
    assert representation.update_layers[0].update_U.bias is None


def test_painn_defaults_use_stable_equivariant_updates_without_residual_state():
    representation = Painn(
        num_interactions=2,
        num_features=8,
        cutoff=5.0,
    )

    assert representation.norm_eps == 1e-8
    assert representation.update_layers[0].update_U.bias is None
    assert not any(
        key.startswith("message_residuals.")
        for key in representation.state_dict()
    )


def test_painn_export_preserves_engineering_components():
    source = Painn(
        num_interactions=1,
        num_features=8,
        cutoff=5.0,
        activation=ScaledSiLU,
        scalar_norm=nn.LayerNorm,
        layer_scales=[1.5],
    )

    rebuilt = Painn(**source.export_init_kwargs())

    assert isinstance(rebuilt.message_layers[0].scalar_norm, nn.LayerNorm)
    assert isinstance(rebuilt.message_layers[0].scalar_message_mlp[1], ScaledSiLU)
    assert rebuilt.layer_scales[0].scale.item() == 1.5


def test_painn_supports_explicit_atomic_number_indexing_and_update_order():
    representation = Painn(
        num_interactions=1,
        num_features=8,
        cutoff=5.0,
        num_elements=83,
        atomic_number_offset=1,
        update_scalar_first=True,
    )

    assert representation.atom_embedding.num_embeddings == 83
    assert representation.atomic_number_offset == 1
    assert representation.update_layers[0].scalar_first
