"""Nyström approximates the same atomic Gaussian kernel and its structure mean.

The reference calculations deliberately materialize small readout gradients and
pairwise differences; production code must work from factors and kernel blocks.
"""

import pytest
import torch

from curator.layer.feature.calculator import FeatureCalculator
from curator.layer.feature.common import ExtractedFeatures
from curator.layer.feature.kernel import FeatureKernel
from curator.layer.feature.nystrom import NystromMap, gaussian_cross_kernel
from curator.layer.feature.readout import readout_parameter_layout
from curator.layer.feature.store import H5Feature
from curator.layer.feature.statistics import FeatureStatistics
from curator.select.kernel import FeatureKernelMatrix, KernelMatrix
from curator.select.select import lcmd_greedy


def _pairs(n=14, seed=107):
    rng = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(n, p, generator=rng, dtype=torch.float64),
            torch.randn(n, q, generator=rng, dtype=torch.float64),
        )
        for p, q in [(3, 2), (2, 1)]
    ]


def _dense(pairs):
    return torch.cat([torch.einsum("ni,nj->nij", a, b).flatten(1) for a, b in pairs], dim=1)


def _take(pairs, indices):
    return [(a[indices], b[indices]) for a, b in pairs]


def _rbf(x, y, sigma):
    distances = (x[:, None, :] - y[None, :, :]).square().sum(-1)
    return torch.exp(-distances / (2 * sigma**2))


def _membership():
    # Different atom counts catch normalizing the whole batch instead of each
    # structure. The last two structures can also act as existing LCMD centers.
    counts = torch.tensor([2, 1, 3, 2, 4, 2])
    image_idx = torch.arange(len(counts)).repeat_interleave(counts)
    membership = torch.nn.functional.one_hot(image_idx).T.to(torch.float64)
    return image_idx, membership / counts[:, None]


@pytest.mark.parametrize("block_size", [1, 4, 64])
def test_factor_rectangular_kernel_equals_explicit_full_gradient_rbf(block_size):
    atoms, anchors = _pairs(9), _pairs(5, seed=233)
    expected = _rbf(_dense(atoms), _dense(anchors), sigma=2.7)

    actual = gaussian_cross_kernel(atoms, anchors, sigma=2.7, block_size=block_size)
    dense_actual = gaussian_cross_kernel(_dense(atoms), _dense(anchors), sigma=2.7, block_size=block_size)

    assert actual.shape == (9, 5)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-13)
    torch.testing.assert_close(dense_actual, expected, atol=2e-14, rtol=2e-13)


def test_factor_kernel_depends_on_gradient_not_choice_of_outer_product_factors():
    atoms = _pairs(6)
    same_gradients = [(4.0 * a, b / 4.0) for a, b in atoms]
    expected = _rbf(_dense(atoms), _dense(atoms), sigma=1.8)

    actual = gaussian_cross_kernel(atoms, same_gradients, sigma=1.8, block_size=2)

    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-13)
    torch.testing.assert_close(actual.diagonal(), torch.ones(6, dtype=torch.float64), atol=2e-14, rtol=0)


@pytest.mark.parametrize("use_factors", [False, True])
def test_all_atom_landmarks_recover_atomic_kernel_and_structure_kme(use_factors):
    pairs = _pairs()
    dense = _dense(pairs)
    raw = pairs if use_factors else dense
    image_idx, membership = _membership()
    expected_atomic = _rbf(dense, dense, sigma=2.2)
    mapper = NystromMap.fit(raw, sigma=2.2, eigenvalue_rtol=1e-13, block_size=3)

    atomic = mapper.transform(raw)
    structures = mapper.structure_features(raw, image_idx, pooling="mean")

    assert mapper.num_landmarks == len(dense)
    assert mapper.effective_rank == mapper.num_features == len(dense)
    torch.testing.assert_close(atomic @ atomic.T, expected_atomic, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        structures @ structures.T, membership @ expected_atomic @ membership.T, atol=2e-12, rtol=2e-12
    )
    # Self-similarity of a mean embedding is not identically one.
    assert (structures.square().sum(1)[[0, 2, 3, 4, 5]] < 0.99).all()


def test_finite_landmarks_use_nystrom_correction_not_raw_kernel_columns():
    dense = _dense(_pairs())
    anchors = dense[[0, 2, 5, 8, 11]]
    sigma = 3.2
    mapper = NystromMap.fit(anchors, sigma=sigma, eigenvalue_rtol=1e-13)
    columns = _rbf(dense, anchors, sigma)
    anchor_kernel = _rbf(anchors, anchors, sigma)
    # Solve is an independent expression for C W^{-1} C^T.
    expected = columns @ torch.linalg.solve(anchor_kernel, columns.T)
    features = mapper.transform(dense)

    torch.testing.assert_close(features @ features.T, expected, atol=2e-12, rtol=2e-12)
    assert not torch.allclose(columns @ columns.T, expected, atol=1e-3, rtol=1e-3)


def test_dense_and_factor_landmarks_produce_same_approximate_kernel():
    pairs = _pairs()
    indices = [0, 2, 4, 7, 10, 13]
    factor_map = NystromMap.fit(_take(pairs, indices), sigma=2.4, num_features=4)
    dense_map = NystromMap.fit(_dense(pairs)[indices], sigma=2.4, num_features=4)

    factor_features = factor_map.transform(pairs)
    dense_features = dense_map.transform(_dense(pairs))

    assert factor_features.shape == dense_features.shape == (14, 4)
    torch.testing.assert_close(factor_features @ factor_features.T, dense_features @ dense_features.T, atol=2e-12, rtol=2e-12)


def test_blocking_atom_batches_and_permutations_preserve_one_frozen_map():
    pairs = _pairs()
    anchors = _take(pairs, [0, 1, 4, 6, 9, 12])
    mapper = NystromMap.fit(anchors, sigma=2.1, num_features=4, block_size=2)
    alternate = NystromMap.fit(anchors, sigma=2.1, num_features=4, block_size=64)
    expected = mapper.transform(pairs)
    batched = torch.cat([mapper.transform(_take(pairs, part)) for part in [slice(0, 3), slice(3, 8), slice(8, None)]])
    permutation = torch.tensor([7, 2, 0, 13, 4, 6, 10, 9, 11, 5, 3, 12, 8, 1])

    torch.testing.assert_close(batched, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(mapper.transform(_take(pairs, permutation)), expected[permutation], atol=2e-12, rtol=2e-12)
    actual = alternate.transform(pairs)
    torch.testing.assert_close(actual @ actual.T, expected @ expected.T, atol=2e-12, rtol=2e-12)


def test_landmark_order_changes_coordinates_but_not_the_kernel():
    dense = _dense(_pairs())
    indices = torch.tensor([0, 1, 4, 6, 9, 12])
    first = NystromMap.fit(dense[indices], sigma=2.1, num_features=4).transform(dense)
    second = NystromMap.fit(dense[indices.flip(0)], sigma=2.1, num_features=4).transform(dense)

    torch.testing.assert_close(first @ first.T, second @ second.T, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("pooling", ["mean", "sum"])
def test_pooling_kernel_columns_before_correction_equals_pooling_atomic_vectors(pooling):
    pairs = _pairs()
    image_idx, membership = _membership()
    if pooling == "sum":
        membership = torch.nn.functional.one_hot(image_idx).T.to(torch.float64)
    mapper = NystromMap.fit(_take(pairs, [0, 3, 6, 10]), sigma=2.5, block_size=2)
    expected = membership @ mapper.transform(pairs)

    actual = mapper.structure_features(pairs, image_idx, pooling=pooling)

    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_duplicate_landmarks_have_finite_rank_and_stable_requested_output_width():
    anchors = torch.tensor([[0.2, 0.4], [0.2, 0.4], [1.1, -0.3], [1.1, -0.3]], dtype=torch.float64)
    mapper = NystromMap.fit(anchors, sigma=1.0, num_features=4)
    features = mapper.transform(anchors)

    assert mapper.num_landmarks == mapper.num_features == 4
    assert mapper.effective_rank == 2
    assert features.shape == (4, 4)
    assert torch.isfinite(features).all()
    torch.testing.assert_close(features @ features.T, _rbf(anchors, anchors, 1.0), atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("use_factors", [False, True])
def test_saved_map_preserves_features_metadata_and_fingerprint(tmp_path, use_factors):
    pairs = _pairs()
    raw = pairs if use_factors else _dense(pairs)
    anchors = _take(pairs, [0, 3, 7, 11]) if use_factors else raw[[0, 3, 7, 11]]
    metadata = {"raw_feature": "full-gradient" if use_factors else "gnn", "anchor_ids": [0, 3, 7, 11], "checkpoint": "test-checkpoint"}
    mapper = NystromMap.fit(anchors, sigma=2.0, num_features=3, metadata=metadata)
    before = mapper.transform(raw)
    state_path = tmp_path / "mapping.pt"
    mapper.save(state_path)

    restored = NystromMap.load(state_path).to("cpu")

    assert restored.metadata == metadata
    assert isinstance(restored.fingerprint, str) and restored.fingerprint
    assert restored.fingerprint == mapper.fingerprint
    assert restored.num_landmarks == 4
    assert restored.num_features == restored.effective_rank == 3
    assert restored.sigma == 2.0
    torch.testing.assert_close(restored.transform(raw), before, atol=0, rtol=0)
    changed = NystromMap.fit(anchors, sigma=2.3, num_features=3, metadata=metadata)
    assert changed.fingerprint != mapper.fingerprint


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_bandwidth_is_rejected(sigma):
    anchors = torch.eye(3, dtype=torch.float64)
    with pytest.raises((ValueError, TypeError)):
        NystromMap.fit(anchors, sigma=sigma)
    with pytest.raises((ValueError, TypeError)):
        gaussian_cross_kernel(anchors, anchors, sigma=sigma)


@pytest.mark.parametrize("invalid", [torch.tensor([1.0, 2.0]), torch.empty(0, 2), torch.tensor([[float("nan"), 0.0]]), torch.tensor([[float("inf"), 0.0]])])
def test_invalid_landmarks_are_rejected(invalid):
    with pytest.raises((ValueError, TypeError)):
        NystromMap.fit(invalid, sigma=1.0)


@pytest.mark.parametrize("width", [0, -1, 4, 1.5, True])
def test_invalid_requested_dimensions_are_rejected(width):
    with pytest.raises(ValueError):
        NystromMap.fit(torch.eye(3, dtype=torch.float64), sigma=1.0, num_features=width)


def test_query_mapping_is_frozen_and_block_size_does_not_change_state_identity():
    raw = _dense(_pairs(6)).requires_grad_(True)
    mapper = NystromMap.fit(raw[:4], sigma=2.2, block_size=2)
    before = mapper.transform(raw)
    fingerprint = mapper.fingerprint

    mapper.block_size = 16
    after = mapper.transform(raw)
    structures = mapper.structure_features(raw, torch.tensor([0, 0, 1, 1, 1, 1]))

    assert mapper.fingerprint == fingerprint
    assert not before.requires_grad and not after.requires_grad and not structures.requires_grad
    torch.testing.assert_close(after, before, atol=2e-12, rtol=2e-12)


def test_incompatible_query_width_or_factor_layout_is_rejected():
    mapper = NystromMap.fit(torch.eye(3, dtype=torch.float64), sigma=1.0)
    with pytest.raises((ValueError, TypeError)):
        mapper.transform(torch.zeros(2, 4, dtype=torch.float64))
    pairs = _pairs(4)
    factor_map = NystromMap.fit(pairs, sigma=1.0)
    with pytest.raises((ValueError, TypeError)):
        factor_map.transform(pairs[:1])
    with pytest.raises((ValueError, TypeError)):
        factor_map.transform([(pairs[0][0][:, :2], pairs[0][1]), pairs[1]])


def test_feature_kernel_gnn_produces_structure_and_local_nystrom_features(tmp_path):
    dense = _dense(_pairs())
    image_idx, membership = _membership()
    mapper = NystromMap.fit(dense[[0, 3, 6, 10]], sigma=2.3, num_features=3, metadata={"raw_feature": "gnn"})
    state_path = tmp_path / "gnn.pt"
    mapper.save(state_path)
    extracted = ExtractedFeatures(image_idx=image_idx, feats=[torch.cat([dense, torch.ones(len(dense), 1, dtype=dense.dtype)], dim=1)], grads=[])
    configuration = {"nystrom_state": str(state_path), "sigma": 2.3, "num_features": 3}

    structures = FeatureKernel({"preset": "gnn-nystrom", **configuration}).compute(extracted)
    atomic = FeatureKernel({"preset": "local_gnn-nystrom", **configuration}).compute(extracted)

    torch.testing.assert_close(atomic, mapper.transform(dense), atol=0, rtol=0)
    torch.testing.assert_close(structures, membership @ atomic, atol=2e-12, rtol=2e-12)


def test_feature_kernel_fg_uses_actual_parameter_layout_without_virtual_bias(tmp_path):
    rng = torch.Generator().manual_seed(32)
    layer = torch.nn.Linear(3, 2, bias=False).double()
    with torch.no_grad():
        layer.weight.copy_(torch.randn(2, 3, generator=rng, dtype=torch.float64) * 0.3)
    inputs = torch.randn(5, 3, generator=rng, dtype=torch.float64)
    outputs = layer(inputs)
    energies = outputs.square().sum(1)
    adjoint = torch.autograd.grad(energies.sum(), outputs, retain_graph=True)[0]
    native = torch.stack([
        torch.autograd.grad(energies[i], layer.weight, retain_graph=True)[0].T.flatten()
        for i in range(len(inputs))
    ])
    sigma = 1.7
    mapper = NystromMap.fit([(inputs, adjoint)], sigma=sigma, metadata={"raw_feature": "full-gradient"})
    state_path = tmp_path / "readout.pt"
    mapper.save(state_path)
    image_idx = torch.tensor([0, 0, 1, 1, 1])
    extracted = ExtractedFeatures(
        image_idx=image_idx,
        feats=[torch.cat([inputs, torch.ones(5, 1, dtype=torch.float64)], dim=1)],
        grads=[adjoint],
        readout_layouts=[readout_parameter_layout(layer)],
    )
    configuration = {"nystrom_state": str(state_path), "sigma": sigma, "num_features": 5}

    atomic = FeatureKernel({"preset": "local_fg-nystrom", **configuration}).compute(extracted)
    structures = FeatureKernel({"preset": "fg-nystrom", **configuration}).compute(extracted)

    expected_kernel = _rbf(native, native, sigma)
    membership = torch.tensor([[0.5, 0.5, 0, 0, 0], [0, 0, 1 / 3, 1 / 3, 1 / 3]], dtype=torch.float64)
    torch.testing.assert_close(atomic @ atomic.T, expected_kernel, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(structures @ structures.T, membership @ expected_kernel @ membership.T, atol=2e-12, rtol=2e-12)


def test_feature_kernel_rejects_different_readout_layout_metadata(tmp_path):
    layer = torch.nn.Linear(3, 2, bias=False).double()
    pairs = _pairs(4)[:1]
    layout = readout_parameter_layout(layer)
    mapper = NystromMap.fit(pairs, sigma=1.0, metadata={"raw_feature": "full-gradient", "readout_layouts": [layout]})
    state_path = tmp_path / "fg.pt"
    mapper.save(state_path)
    changed = {**layout, "has_bias": True}
    extracted = ExtractedFeatures(
        image_idx=torch.zeros(4, dtype=torch.long),
        feats=[torch.cat([pairs[0][0], torch.ones(4, 1, dtype=torch.float64)], dim=1)],
        grads=[pairs[0][1]],
        readout_layouts=[changed],
    )
    kernel = FeatureKernel({"preset": "fg-nystrom", "nystrom_state": str(state_path), "sigma": 1.0, "num_features": 4})

    with pytest.raises(ValueError, match="readout_layouts"):
        kernel.compute(extracted)


@pytest.mark.parametrize("override", [{"sigma": 0.5}, {"num_features": 2}, {"preset": "llg-nystrom"}])
def test_feature_kernel_rejects_state_with_different_declared_kernel(tmp_path, override):
    mapper = NystromMap.fit(torch.eye(3, dtype=torch.float64), sigma=1.0, metadata={"raw_feature": "gnn"})
    state_path = tmp_path / "gnn.pt"
    mapper.save(state_path)
    configuration = {"preset": "gnn-nystrom", "nystrom_state": str(state_path), "sigma": 1.0, "num_features": 3, **override}

    with pytest.raises(ValueError):
        FeatureKernel(configuration)


def test_feature_store_preserves_double_precision_nystrom_coordinates(tmp_path):
    dense = _dense(_pairs(5))
    features = NystromMap.fit(dense, sigma=2.1).transform(dense)
    store = H5Feature(tmp_path / "features.h5", num_models=1, kernels=["gnn-nystrom"], dataset_size=5)

    store.append("gnn-nystrom", 0, features)
    restored = store.load("gnn-nystrom")

    assert restored.dtype == torch.float64
    torch.testing.assert_close(restored[0], features, atol=0, rtol=0)


def test_feature_store_rejects_other_anchor_mapping_and_unversioned_coordinates(tmp_path):
    dense = _dense(_pairs(5))
    mapper = NystromMap.fit(dense, sigma=2.1)
    other = NystromMap.fit(dense[[0, 1, 2, 3]], sigma=2.1)
    identity = [[{"kernel": "gnn-nystrom", "fingerprint": mapper.fingerprint}]]
    store = H5Feature(tmp_path / "features.h5", num_models=1, kernels=["gnn-nystrom"], dataset_size=5)
    store.ensure(feature_identities=identity)
    store.append("gnn-nystrom", 0, mapper.transform(dense))
    reopened = H5Feature(store.path, num_models=1, kernels=["gnn-nystrom"], dataset_size=5)
    reopened.ensure(feature_identities=identity)

    with pytest.raises(ValueError, match="identit"):
        reopened.ensure(feature_identities=[[{"kernel": "gnn-nystrom", "fingerprint": other.fingerprint}]])

    old = H5Feature(tmp_path / "old.h5", num_models=1, kernels=["gnn-nystrom"], dataset_size=5)
    old.append("gnn-nystrom", 0, mapper.transform(dense))
    with pytest.raises(ValueError, match="identity"):
        old.ensure(feature_identities=identity)


def test_statistics_rejects_coordinate_standardization_that_changes_nystrom_kernel(tmp_path):
    state_path = tmp_path / "gnn.pt"
    NystromMap.fit(torch.eye(3, dtype=torch.float64), sigma=1.0, metadata={"raw_feature": "gnn"}).save(state_path)
    calculator = FeatureCalculator(kernels=[{
        "preset": "gnn-nystrom", "nystrom_state": str(state_path), "sigma": 1.0, "num_features": 3,
    }])
    stats = FeatureStatistics(models=[torch.nn.Linear(3, 1)], dataset=[], calculators=[calculator], device="cpu")

    with pytest.raises(ValueError, match="normalize=False"):
        stats.get_features(normalize=True)


class _DenseKernel(KernelMatrix):
    """Independent exact-kernel adapter used only to check LCMD decisions."""

    def __init__(self, matrix):
        super().__init__(len(matrix))
        self.matrix = matrix

    def get_column(self, i):
        return self.matrix[:, i]

    def get_diag(self):
        return self.matrix.diagonal()


@pytest.mark.parametrize("n_train", [0, 2])
def test_existing_lcmd_accepts_nystrom_vectors_and_matches_exact_kme(n_train):
    dense = _dense(_pairs())
    image_idx, membership = _membership()
    sigma = 2.2
    exact_gram = membership @ _rbf(dense, dense, sigma) @ membership.T
    exact = _DenseKernel(exact_gram)
    mapper = NystromMap.fit(dense, sigma=sigma, eigenvalue_rtol=1e-13)
    vectors = mapper.structure_features(dense, image_idx)
    mapped = FeatureKernelMatrix(vectors.unsqueeze(0))

    for i in range(len(exact_gram)):
        torch.testing.assert_close(mapped.get_sq_dists(i), exact.get_sq_dists(i), atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(lcmd_greedy(mapped, batch_size=3, n_train=n_train), lcmd_greedy(exact, batch_size=3, n_train=n_train))
