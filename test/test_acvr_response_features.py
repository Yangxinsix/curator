"""Backbone-independent response features preserve types, values and derivatives."""

import pytest
import torch

from curator.data import properties
from curator.layer import PairwiseDistance, VariationalEnergyOutput
from curator.model import MACE, Nequip, NeuralNetworkPotential, Painn
from curator.model.base import Representation
from curator.model.features import (
    IrrepsFeatureSpec,
    ScalarVectorFeatureSpec,
    scalar_vector_features,
)
from e3nn import o3


def _batch(positions):
    return PairwiseDistance(compute_distance_from_R=True)({
        properties.Z: torch.tensor([1, 6, 8]),
        properties.positions: positions,
        properties.edge_idx: torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]]),
        properties.n_atoms: torch.tensor([3]),
        properties.image_idx: torch.zeros(3, dtype=torch.long),
    })


def _representation(name):
    if name == "painn":
        return Painn(2, 4, 5.0, num_basis=4).double()
    common = dict(cutoff=5.0, num_interactions=2, species=["H", "C", "O"],
                  lmax=2, parity=True, num_features=3, num_basis=4)
    if name == "mace":
        return MACE(**common, correlation=2, avg_num_neighbors=2.0).double()
    return Nequip(**common, radial_mlp_width=8).double()


def test_scalar_vector_flatten_roundtrip_and_gradients():
    spec = ScalarVectorFeatureSpec("scalar", "vector", 2, 3)
    scalar = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    vector = torch.randn(4, 3, 3, dtype=torch.float64, requires_grad=True)
    flat = spec.flatten({"scalar": scalar, "vector": vector})
    assert spec.irreps == o3.Irreps("2x0e + 3x1o")
    unpacked_scalar, unpacked_vector = scalar_vector_features(flat, spec.irreps)
    torch.testing.assert_close(unpacked_scalar, scalar, rtol=0, atol=0)
    torch.testing.assert_close(unpacked_vector, vector, rtol=0, atol=0)
    flat.square().sum().backward()
    torch.testing.assert_close(scalar.grad, 2 * scalar)
    torch.testing.assert_close(vector.grad, 2 * vector)


@pytest.mark.parametrize("n_atoms", [0, 4])
def test_scalar_only_features_do_not_require_a_vector_tensor(n_atoms):
    features = torch.randn(n_atoms, 5)
    for spec in (ScalarVectorFeatureSpec("scalar", "", 5, 0), IrrepsFeatureSpec("scalar", "5x0e")):
        assert spec.flatten({"scalar": features}) is features
        scalar, vector = scalar_vector_features(features, spec.irreps)
        torch.testing.assert_close(scalar, features)
        assert vector.shape == (n_atoms, 3, 0)


def test_irreps_features_keep_high_order_and_nonpolar_channels():
    spec = IrrepsFeatureSpec("features", "2x0e + 1x2e + 1x0o + 2x1o + 1x1e + 1x0e")
    features = torch.randn(3, spec.irreps.dim, requires_grad=True)
    assert spec.flatten({"features": features}) is features
    scalar, vector = scalar_vector_features(features, spec.irreps)
    torch.testing.assert_close(scalar, torch.cat((features[:, :2], features[:, -1:]), dim=-1))
    torch.testing.assert_close(vector, features[:, 8:14].reshape(3, 2, 3).transpose(1, 2))
    features.square().sum().backward()
    assert bool(features.grad[:, 2:7].abs().sum() > 0)


@pytest.mark.parametrize("name", ["painn", "mace", "nequip"])
def test_real_backbones_export_all_declared_features_equivariantly(name):
    torch.manual_seed(23)
    representation = _representation(name)
    spec = representation.response_feature_spec()
    positions = torch.tensor([[0., 0., 0.], [1.1, .2, 0.], [.1, 1., .3]],
                             dtype=torch.float64, requires_grad=True)
    # Include a reflection to check parity as well as rotations.
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., -1.]], dtype=torch.float64)
    data = representation(_batch(positions))
    features = spec.flatten(data)
    transformed = spec.flatten(representation(_batch(positions @ rotation.T + 2.5)))
    assert features.shape == (3, spec.irreps.dim)
    torch.testing.assert_close(transformed, features @ spec.irreps.D_from_matrix(rotation).T,
                               rtol=2e-6, atol=2e-7)
    if name != "painn":
        assert any(irrep.l == 2 for _, irrep in spec.irreps)
    position_gradient = torch.autograd.grad(features.square().sum(), positions, create_graph=True)[0]
    assert torch.isfinite(position_gradient).all()
    assert bool(position_gradient.abs().sum() > 0)
    curvature = torch.autograd.grad(position_gradient.square().sum(), positions)[0]
    assert torch.isfinite(curvature).all()
    if name in ("painn", "mace"):
        assert spec == representation.direct_force_feature_spec()
    else:
        # The ordinary energy readout still receives its final scalar features.
        assert data[properties.node_feat].shape[-1] == representation.final_layer_irreps.dim


def test_nequip_response_tensor_is_every_interaction_output():
    representation = _representation("nequip")
    outputs = []
    handles = [layer.register_forward_hook(lambda _module, _inputs, output: outputs.append(output))
               for layer in representation.interactions]
    positions = torch.tensor([[0., 0., 0.], [1.1, .2, 0.], [.1, 1., .3]], dtype=torch.float64)
    try:
        data = representation(_batch(positions))
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(representation.response_feature_spec().flatten(data), torch.cat(outputs, dim=-1))
    torch.testing.assert_close(data[properties.node_feat], outputs[-1])


def test_variational_output_binds_response_interface_independently_of_force_interface():
    class ScalarRepresentation(Representation):
        cutoff = 5.0

        def response_feature_spec(self):
            return IrrepsFeatureSpec("scalar", "4x0e")

    representation = ScalarRepresentation()
    with pytest.raises(NotImplementedError, match="direct-force"):
        representation.direct_force_feature_spec()
    output = VariationalEnergyOutput(state_kind="scalar", components=1, hidden_features=4)
    NeuralNetworkPotential(representation, output_modules=[output])
    assert output.feature_spec == representation.response_feature_spec()


def test_existing_direct_force_representation_has_response_fallback():
    class ExistingRepresentation(Representation):
        def direct_force_feature_spec(self):
            return ScalarVectorFeatureSpec("scalar", "vector", 2, 2)

    representation = ExistingRepresentation()
    assert representation.response_feature_spec() == representation.direct_force_feature_spec()


@pytest.mark.parametrize("spec,data", [
    (IrrepsFeatureSpec("features", "2x0e + 1x2e"), {"features": torch.zeros(3, 2)}),
    (ScalarVectorFeatureSpec("s", "v", 2, 3), {"s": torch.zeros(3, 2), "v": torch.zeros(3, 3, 2)}),
])
def test_feature_shape_must_match_declared_types(spec, data):
    with pytest.raises(ValueError, match="shape"):
        spec.flatten(data)
