"""Verify the joint readout map against explicit full-gradient references."""

import math

import pytest
import torch

from curator.layer.feature.common import ExtractedFeatures, feature_spec_from_object
from curator.layer.feature.gaussian import BlockGaussianProjection
from curator.layer.feature.kernel import FeatureKernel
from curator.layer.feature.kme import RandomFourierKMEAggregator, SketchingKMEAggregator


def _pairs():
    generator = torch.Generator().manual_seed(915)
    return [
        (
            torch.randn(5, n_in, generator=generator, dtype=torch.float64),
            torch.randn(5, n_out, generator=generator, dtype=torch.float64),
        )
        for n_in, n_out in [(3, 2), (2, 3)]
    ]


def _dense(pairs):
    return torch.cat([(a[:, :, None] * b[:, None, :]).flatten(1) for a, b in pairs], dim=1)


def _raw(pairs):
    return [a for a, _ in pairs], [b for _, b in pairs]


def test_block_projection_matches_explicit_gradient_and_random_matrix():
    pairs = _pairs()
    dense = _dense(pairs)
    projector = BlockGaussianProjection(11, seed=71, row_tile=5, column_tile=4)
    rows = []
    for row_index in range(math.ceil(dense.shape[1] / 5)):
        row = torch.cat(
            [projector.tile(row_index, j, dense.device, dense.dtype) for j in range(3)],
            dim=1,
        )
        rows.append(row)
    weight = torch.cat(rows, dim=0)[: dense.shape[1], :11]
    expected = dense @ weight

    torch.testing.assert_close(projector.project_pairs(pairs), expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(projector.project_dense(dense), expected, rtol=1e-13, atol=1e-13)


def test_projection_does_not_depend_on_layer_partition():
    dense = _dense(_pairs())
    ones = torch.ones((dense.shape[0], 1), dtype=dense.dtype)
    projector = BlockGaussianProjection(13, seed=71, row_tile=5, column_tile=4)
    single = projector.project_pairs([(dense, ones)])
    partitioned = projector.project_pairs([(dense[:, :3], ones), (dense[:, 3:8], ones), (dense[:, 8:], ones)])

    torch.testing.assert_close(partitioned, single, rtol=1e-13, atol=1e-13)


def test_atom_batches_and_projection_cache_preserve_the_map():
    pairs = _pairs()
    uncached = BlockGaussianProjection(11, seed=71, row_tile=5, column_tile=4, cache_bytes=0)
    cached = BlockGaussianProjection(11, seed=71, row_tile=5, column_tile=4, cache_bytes=512)
    expected = uncached.project_pairs(pairs)
    batched = torch.cat(
        [cached.project_pairs([(a[part], b[part]) for a, b in pairs]) for part in [slice(0, 2), slice(2, 5)]],
        dim=0,
    )

    torch.testing.assert_close(cached.project_pairs(pairs), expected, rtol=0, atol=0)
    torch.testing.assert_close(batched, expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(cached.project_pairs(pairs), expected, rtol=0, atol=0)


def test_larger_output_dimension_preserves_projection_and_phase_prefix():
    dense = _dense(_pairs())
    small = BlockGaussianProjection(11, seed=71, row_tile=5, column_tile=4)
    large = BlockGaussianProjection(19, seed=71, row_tile=5, column_tile=4)

    torch.testing.assert_close(small.project_dense(dense), large.project_dense(dense)[:, :11], rtol=0, atol=0)
    torch.testing.assert_close(
        small.phase(dense.device, dense.dtype), large.phase(dense.device, dense.dtype)[:11], rtol=0, atol=0
    )


def test_joint_rff_depends_on_gradient_not_its_factorization():
    pairs = _pairs()
    equivalent_pairs = [(2.0 * a, 0.5 * b) for a, b in pairs]
    mapper = RandomFourierKMEAggregator(97, layer_combine="joint", sigma=1.7, seed=9)

    torch.testing.assert_close(mapper.transform(_raw(pairs)), mapper.transform(_raw(equivalent_pairs)), rtol=0, atol=0)


def test_sketch_and_rff_share_one_global_gaussian_projection():
    pairs = _pairs()
    dimension, seed, sigma = 513, 41, 1.7
    projector = BlockGaussianProjection(dimension, seed=seed)
    projected = projector.project_dense(_dense(pairs))
    sketch = SketchingKMEAggregator(dimension, layer_combine="joint", seed=seed)
    rff = RandomFourierKMEAggregator(dimension, layer_combine="joint", sigma=sigma, seed=seed)
    expected_rff = math.sqrt(2.0 / dimension) * torch.cos(
        projected / sigma + projector.phase(projected.device, projected.dtype)
    )

    assert sketch.transform(_raw(pairs)).shape == (5, dimension)
    torch.testing.assert_close(sketch.transform(_raw(pairs)), projected / math.sqrt(dimension), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(rff.transform(_raw(pairs)), expected_rff, rtol=1e-12, atol=1e-12)


def test_joint_rff_approximates_full_gradient_rbf_and_its_set_mean():
    # The first two atoms have identical outer products but different a/b factors.
    # A Gaussian kernel on a and b separately would assign them similarity < 1.
    a = torch.tensor([[1.0], [2.0], [-0.5], [0.2]], dtype=torch.float64)
    b = torch.tensor([[1.0, 0.4], [0.5, 0.2], [0.6, 1.0], [-0.5, 0.9]], dtype=torch.float64)
    c = torch.tensor([[0.1], [0.1], [0.7], [-0.3]], dtype=torch.float64)
    pairs = [(a, b), (c, torch.ones_like(c))]
    dense = _dense(pairs)
    sigma = 1.1
    mapper = RandomFourierKMEAggregator(32768, pooling="mean", layer_combine="joint", sigma=sigma, seed=37)
    features = mapper.transform(_raw(pairs))
    expected = torch.exp(-torch.cdist(dense, dense).square() / (2 * sigma**2))

    torch.testing.assert_close(features @ features.T, expected, atol=0.025, rtol=0)
    torch.testing.assert_close(features[0], features[1], rtol=0, atol=0)
    membership = torch.tensor([[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]], dtype=torch.float64)
    mean_features = mapper.reduce(features, torch.tensor([0, 0, 1, 1]))
    torch.testing.assert_close(mean_features @ mean_features.T, membership @ expected @ membership.T, atol=0.025, rtol=0)


def test_joint_kme_applies_cosine_before_atom_mean():
    a = torch.tensor([[0.0], [2.0], [-1.0], [1.0]], dtype=torch.float64)
    b = torch.ones_like(a)
    mapper = RandomFourierKMEAggregator(67, pooling="mean", layer_combine="joint", sigma=0.8, seed=8)
    atomic = mapper.transform(([a], [b]))
    actual = mapper.reduce(atomic, torch.tensor([0, 0, 1, 1]))
    expected = torch.stack([atomic[:2].mean(0), atomic[2:].mean(0)])
    mapped_mean = mapper.transform(([torch.tensor([[1.0], [0.0]], dtype=a.dtype)], [torch.ones(2, 1, dtype=a.dtype)]))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.linalg.vector_norm(actual - mapped_mean).item() > 0.2


@pytest.mark.parametrize("preset", ["fg-rff", "fg-sketch", "local_fg-rff", "local_fg-sketch"])
def test_fg_presets_use_one_joint_output_and_mean_pooling(preset):
    spec = feature_spec_from_object({"preset": preset, "num_features": 31})
    kernel = FeatureKernel(spec)

    assert spec.layer_combine == "joint"
    assert spec.layer_norm == "none"
    assert spec.pooling == "mean"
    assert kernel.kme.transform(_raw(_pairs())).shape == (5, 31)


@pytest.mark.parametrize("mapping", ["rff", "gaussian_sketch"])
def test_joint_layer_normalization_is_rejected(mapping):
    with pytest.raises(ValueError):
        feature_spec_from_object({"preset": "fg-rff", "mapping": mapping, "layer_norm": "rms"})
    mapper = RandomFourierKMEAggregator if mapping == "rff" else SketchingKMEAggregator
    with pytest.raises(ValueError):
        mapper(16, layer_combine="joint", layer_norm="rms")


@pytest.mark.parametrize("rff_kernel", ["matern32", "matern52", "laplacian_l1"])
def test_joint_nongaussian_kernel_is_rejected(rff_kernel):
    with pytest.raises(ValueError):
        feature_spec_from_object({"preset": "fg-rff", "rff_kernel": rff_kernel})
    with pytest.raises(ValueError):
        RandomFourierKMEAggregator(16, layer_combine="joint", rff_kernel=rff_kernel)


def test_joint_feature_kernel_requires_actual_readout_parameter_layout():
    pairs = _pairs()
    extracted = ExtractedFeatures(image_idx=torch.zeros(5, dtype=torch.long), feats=_raw(pairs)[0], grads=_raw(pairs)[1])
    kernel = FeatureKernel({"preset": "fg-rff", "num_features": 16})

    with pytest.raises(ValueError, match="[Rr]eadout|layout|metadata"):
        kernel.compute(extracted)
