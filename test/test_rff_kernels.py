import math

import pytest
import torch

from curator.layer.feature.common import feature_spec_from_object
from curator.layer.feature.kernel import FeatureKernel
from curator.layer.feature.kme import RandomFourierKMEAggregator


@pytest.mark.parametrize("rff_kernel", ["rbf", "matern32", "matern52", "laplacian_l1"])
def test_rff_approximates_stationary_kernel(rff_kernel):
    x = torch.tensor(
        [[0.0, 0.0], [0.3, -0.4], [1.0, 0.2], [-0.7, 0.8]],
        dtype=torch.float64,
    )
    sigma = 1.3
    distance = torch.cdist(x, x)

    if rff_kernel == "rbf":
        expected = torch.exp(-distance.square() / (2 * sigma**2))
    elif rff_kernel == "matern32":
        scaled = math.sqrt(3) * distance / sigma
        expected = (1 + scaled) * torch.exp(-scaled)
    elif rff_kernel == "matern52":
        scaled = math.sqrt(5) * distance / sigma
        expected = (1 + scaled + scaled.square() / 3) * torch.exp(-scaled)
    else:
        expected = torch.exp(-torch.cdist(x, x, p=1) / sigma)

    transformed = RandomFourierKMEAggregator(
        num_features=32768,
        sigma=sigma,
        rff_kernel=rff_kernel,
        seed=7,
    ).transform_simple(x, 0)

    torch.testing.assert_close(transformed @ transformed.T, expected, atol=0.025, rtol=0)


def test_rbf_remains_the_default_and_feature_spec_accepts_rff_kernel():
    x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    default = RandomFourierKMEAggregator(64, "sum", "concat", "none", 2.0, 5).transform_simple(x, 0)
    explicit = RandomFourierKMEAggregator(64, sigma=2.0, rff_kernel="rbf", seed=5).transform_simple(x, 0)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(8)
    weight = torch.randn(2, 64, generator=generator) / 2.0
    bias = 2.0 * math.pi * torch.rand(64, generator=generator)
    expected = math.sqrt(2.0 / 64) * torch.cos(x @ weight + bias)

    torch.testing.assert_close(default, explicit, rtol=0, atol=0)
    torch.testing.assert_close(default, expected, rtol=0, atol=0)
    spec = feature_spec_from_object(
        {
            "name": "gnn-matern32",
            "raw_feature": "gnn",
            "mapping": "rff",
            "rff_kernel": "matern32",
        }
    )
    assert spec.rff_kernel == "matern32"
    assert FeatureKernel(spec).kme.rff_kernel == "matern32"


def test_invalid_rff_kernel_is_rejected():
    with pytest.raises(ValueError, match="Unsupported RFF kernel"):
        feature_spec_from_object(
            {
                "name": "gnn-invalid-rff",
                "raw_feature": "gnn",
                "mapping": "rff",
                "rff_kernel": "invalid",
            }
        )


def test_matern_group_mean_matches_exact_set_kernel():
    x = torch.tensor(
        [[0.0, 0.0], [0.3, -0.4], [1.0, 0.2], [-0.7, 0.8]],
        dtype=torch.float64,
    )
    image_idx = torch.tensor([0, 0, 1, 1])
    sigma = 1.3
    aggregator = RandomFourierKMEAggregator(
        num_features=32768,
        pooling="mean",
        sigma=sigma,
        seed=7,
        rff_kernel="matern32",
    )
    groups = aggregator.reduce(aggregator.transform_simple(x, 0), image_idx)

    scaled = math.sqrt(3) * torch.cdist(x, x) / sigma
    point_kernel = (1 + scaled) * torch.exp(-scaled)
    membership = torch.tensor([[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]], dtype=torch.float64)
    expected = membership @ point_kernel @ membership.T

    torch.testing.assert_close(groups @ groups.T, expected, atol=0.025, rtol=0)
