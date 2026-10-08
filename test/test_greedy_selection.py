"""Selection correctness against direct Euclidean and determinant objectives."""

import math

import pytest
import torch

from curator.select.kernel import FeatureKernelMatrix, KernelMatrix
from curator.select.select import _call_selection, max_det_greedy, max_dist_greedy


class DenseKernel(KernelMatrix):
    def __init__(self, kernel):
        super().__init__(kernel.shape[0])
        self.kernel = kernel

    def get_diag(self):
        return self.kernel.diagonal()

    def get_column(self, index):
        return self.kernel[:, index]


def determinant_reference(kernel, budget, n_train=0, regularization=1e-6):
    """Enumerate candidate determinants without the production Cholesky update."""
    kernel = kernel.double()
    n_pool = len(kernel) - n_train
    scale = kernel.diagonal()[:n_pool].mean().item() or 1.0
    regularized = kernel + regularization * scale * torch.eye(len(kernel), dtype=torch.float64)
    selected = []
    conditioning = list(range(n_pool, len(kernel)))
    for _ in range(min(budget, n_pool)):
        candidates = [index for index in range(n_pool) if index not in selected]
        scores = []
        for index in candidates:
            subset = conditioning + selected + [index]
            sign, score = torch.linalg.slogdet(regularized[subset][:, subset])
            assert sign > 0
            scores.append(score.item())
        selected.append(candidates[max(range(len(scores)), key=scores.__getitem__)])
    return selected


def farthest_point_reference(features, budget, n_train=0):
    n_pool = len(features) - n_train
    selected = []
    distances = torch.cdist(features.double(), features.double()).square()
    for _ in range(min(budget, n_pool)):
        centres = list(range(n_pool, len(features))) + selected
        candidates = [index for index in range(n_pool) if index not in selected]
        if centres:
            scores = distances[candidates][:, centres].min(dim=1).values
        else:
            scores = features[candidates].square().sum(dim=1)
        selected.append(candidates[int(scores.argmax())])
    return selected


def test_maxdist_does_not_keep_origin_as_unselected_centre():
    features = torch.tensor([[[10.0], [5.0], [-1.0]]])
    assert max_dist_greedy(FeatureKernelMatrix(features), 2).tolist() == [0, 2]


@pytest.mark.parametrize("n_train", [0, 1, 3])
def test_maxdist_matches_direct_farthest_point_objective(n_train):
    generator = torch.Generator().manual_seed(18)
    features = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    actual = max_dist_greedy(FeatureKernelMatrix(features[None]), 6, n_train=n_train)
    assert actual.tolist() == farthest_point_reference(features, 6, n_train)


@pytest.mark.parametrize("selector", [max_dist_greedy, max_det_greedy])
@pytest.mark.parametrize("n_total,n_train,budget", [(0, 0, 3), (4, 0, 0), (4, 4, 3), (4, 0, 8), (4, 2, 8)])
def test_budget_boundaries_zero_features_and_unique_pool_indices(selector, n_total, n_train, budget):
    kernel = DenseKernel(torch.zeros(n_total, n_total, dtype=torch.float64))
    actual = selector(kernel, budget, n_train=n_train)
    expected_count = min(budget, n_total - n_train)
    assert actual.dtype == torch.long
    assert actual.device == kernel.kernel.device
    assert actual.tolist() == list(range(expected_count))


@pytest.mark.parametrize("selector", [max_dist_greedy, max_det_greedy])
@pytest.mark.parametrize("kwargs", [{"batch_size": -1}, {"batch_size": 1.5}, {"batch_size": 1, "n_train": -1}, {"batch_size": 1, "n_train": 4}, {"batch_size": 1, "n_train": 0.5}])
def test_invalid_pool_budget_or_train_layout(selector, kwargs):
    with pytest.raises(ValueError):
        selector(DenseKernel(torch.eye(3)), **kwargs)


@pytest.mark.parametrize("n_train", [0, 2])
@pytest.mark.parametrize("regularization", [1e-6, 0.03, 1.0])
def test_regularized_maxdet_matches_independent_logdet_greedy(n_train, regularization):
    generator = torch.Generator().manual_seed(51)
    features = torch.randn(9, 3, generator=generator, dtype=torch.float64)
    kernel = features @ features.T
    expected = determinant_reference(kernel, 5, n_train, regularization)
    actual = max_det_greedy(DenseKernel(kernel), 5, n_train, regularization)
    assert actual.tolist() == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kernel_scale", [1e-20, 1.0, 1e20])
def test_relative_regularization_is_scale_invariant_for_low_rank_kernel(dtype, kernel_scale):
    features = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=dtype)
    kernel = (features @ features.T) * kernel_scale
    actual = max_det_greedy(DenseKernel(kernel), 4)
    assert actual.tolist() == [3, 2, 1, 0]
    assert actual.tolist() == determinant_reference(kernel, 4)


def test_regularized_maxdet_handles_duplicates_without_random_completion():
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    kernel = features @ features.T
    actual = max_det_greedy(DenseKernel(kernel), 4)
    assert actual.tolist() == [0, 2, 1, 3]


def test_pool_diagonal_mean_does_not_overflow_for_large_finite_kernel():
    features = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
    kernel = (features @ features.T) * 1e307
    assert torch.isfinite(kernel).all()
    assert max_det_greedy(DenseKernel(kernel), 4).tolist() == [3, 2, 1, 0]


def test_maxdet_default_selects_beyond_feature_rank():
    features = torch.tensor([[[1.0], [2.0], [3.0]]])
    assert max_det_greedy(FeatureKernelMatrix(features), 3).tolist() == [2, 1, 0]


def test_maxdet_conditions_on_train_and_never_selects_train_tail():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [10.0, 0.0]], dtype=torch.float64)
    kernel = features @ features.T
    actual = _call_selection(
        max_det_greedy,
        matrix=DenseKernel(kernel),
        batch_size=2,
        n_train=1,
        selection_kwargs={"regularization": 1e-6},
    )
    assert actual.tolist() == [1, 0]
    assert actual.tolist() == determinant_reference(kernel, 2, n_train=1)


def test_maxdet_unregularized_full_rank_remains_available():
    generator = torch.Generator().manual_seed(83)
    features = torch.randn(6, 6, generator=generator, dtype=torch.float64)
    kernel = features @ features.T + torch.eye(6, dtype=torch.float64)
    actual = max_det_greedy(DenseKernel(kernel), 5, n_train=1, regularization=0)
    assert actual.tolist() == determinant_reference(kernel, 5, n_train=1, regularization=0)


@pytest.mark.parametrize("n_train", [0, 2])
def test_unregularized_rank_exhaustion_raises_instead_of_returning_short_batch(n_train):
    kernel = torch.ones(5, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="exhausted.*rank.*positive regularization"):
        max_det_greedy(DenseKernel(kernel), 3, n_train=n_train, regularization=0)


@pytest.mark.parametrize("regularization", [-1, math.inf, -math.inf, math.nan, None, "bad"])
def test_invalid_regularization(regularization):
    with pytest.raises(ValueError, match="regularization must"):
        max_det_greedy(DenseKernel(torch.eye(3)), 2, regularization=regularization)


@pytest.mark.parametrize("bad_value", [-1.0, math.nan, math.inf])
def test_invalid_diagonal(bad_value):
    kernel = torch.eye(3, dtype=torch.float64)
    kernel[0, 0] = bad_value
    with pytest.raises(ValueError, match="kernel diagonal"):
        max_det_greedy(DenseKernel(kernel), 2)


def test_nonfinite_column_rejected():
    kernel = torch.eye(2, dtype=torch.float64)
    kernel[1, 0] = math.nan
    with pytest.raises(ValueError, match="finite kernel columns"):
        max_det_greedy(DenseKernel(kernel), 2)


def test_indefinite_kernel_is_not_silently_clipped_or_completed():
    kernel = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="not PSD"):
        max_det_greedy(DenseKernel(kernel), 2)


def test_maxdet_works_with_exact_gaussian_kernel_columns():
    features = torch.tensor([[0.0], [0.1], [0.7], [1.3], [3.0]], dtype=torch.float64)
    kernel = torch.exp(-torch.cdist(features, features).square() / 2)
    actual = max_det_greedy(DenseKernel(kernel), 4)
    assert actual.tolist() == determinant_reference(kernel, 4)
