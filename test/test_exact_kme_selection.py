"""Exact atom-pair kernel oracles and production-selector integration."""
import pytest
import torch

from curator.select.exact_kme import ExactGaussianKMEKernelMatrix
from curator.select.kernel import KernelMatrix
from curator.select.select import lcmd_greedy, max_det_greedy, max_dist_greedy


def _data(dtype=torch.float64):
    generator = torch.Generator().manual_seed(5173)
    counts = [2, 3, 2, 5, 4, 2, 3, 4, 3, 6]
    pairs = [(torch.randn(sum(counts), p, generator=generator, dtype=dtype),
              torch.randn(sum(counts), q, generator=generator, dtype=dtype))
             for p, q in [(3, 4), (2, 3)]]
    return counts, pairs


def _dense(pairs):
    return torch.cat([(a.double()[:, :, None] * b.double()[:, None, :]).flatten(1)
                      for a, b in pairs], dim=1)


def _oracle(raw, counts, sigma):
    raw = raw.double()
    # Explicit differences are an independent reference for norm/inner contraction.
    atoms = torch.exp(-((raw[:, None, :] - raw[None, :, :]) ** 2).sum(-1) / (2 * sigma ** 2))
    membership = torch.zeros((len(counts), sum(counts)), dtype=torch.float64)
    start = 0
    for i, count in enumerate(counts):
        membership[i, start:start + count] = 1 / count
        start += count
    return membership @ atoms @ membership.T


class StoredKernel(KernelMatrix):
    def __init__(self, kernel):
        super().__init__(len(kernel))
        self.kernel = kernel

    def get_diag(self):
        return self.kernel.diag()

    def get_column(self, i):
        return self.kernel[:, i]


@pytest.mark.parametrize('factors', [False, True])
@pytest.mark.parametrize('block_size', [1, 4, 128])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_matches_explicit_gradient_atomic_kernel_and_membership(factors, block_size, dtype):
    counts, pairs = _data(dtype)
    dense = _dense(pairs)
    raw = pairs if factors else dense.to(dtype)
    expected = _oracle(dense if factors else raw, counts, 3.7)
    matrix = ExactGaussianKMEKernelMatrix(raw, counts, sigma=3.7, block_size=block_size)
    actual = torch.stack([matrix.get_column(i) for i in range(len(counts))], dim=1)
    assert matrix.get_number_of_columns() == len(counts)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=3e-13, atol=3e-13)
    torch.testing.assert_close(matrix.get_diag(), expected.diag(), rtol=3e-13, atol=3e-13)
    for i in range(len(counts)):
        distances = expected[i, i] + expected.diag() - 2 * expected[:, i]
        torch.testing.assert_close(matrix.get_sq_dists(i), distances, rtol=3e-13, atol=3e-13)
        assert matrix.get_sq_dists(i)[i] == 0


@pytest.mark.parametrize('selector', [lcmd_greedy, max_dist_greedy, max_det_greedy])
@pytest.mark.parametrize('n_train', [0, 2])
def test_selectors_match_explicit_kernel(selector, n_train):
    counts, pairs = _data()
    expected = StoredKernel(_oracle(_dense(pairs), counts, 3.7))
    actual = ExactGaussianKMEKernelMatrix(pairs, counts, 3.7, block_size=4)
    oracle_ids = selector(expected, batch_size=6, n_train=n_train)
    actual_ids = selector(actual, batch_size=6, n_train=n_train)
    assert len(actual_ids) == len(set(actual_ids.tolist())) == 6
    assert actual_ids.max() < len(counts) - n_train
    assert torch.equal(actual_ids, oracle_ids)


def test_identical_structures_single_atoms_and_no_diagonal_normalisation():
    raw = torch.tensor([[0., 2.], [0., 2.], [1., 2.], [3., 5.], [1., 2.], [3., 5.]])
    matrix = ExactGaussianKMEKernelMatrix(raw, [1, 1, 2, 2], 0.8, block_size=1)
    torch.testing.assert_close(matrix.get_diag()[:2], torch.ones(2, dtype=torch.float64))
    assert (matrix.get_diag()[2:] < 1).all()
    assert matrix.get_sq_dists(0)[1] == 0
    assert matrix.get_sq_dists(2)[3] == 0
    for select in [lcmd_greedy, max_dist_greedy, max_det_greedy]:
        ids = select(matrix, batch_size=4, n_train=0)
        assert len(set(ids.tolist())) == 4


def test_atom_permutations_within_structures_preserve_kernel():
    counts, pairs = _data()
    original = _dense(pairs)
    permutation = []
    start = 0
    for count in counts:
        permutation.extend(reversed(range(start, start + count)))
        start += count
    a = ExactGaussianKMEKernelMatrix(original, counts, 2.4, block_size=3)
    b = ExactGaussianKMEKernelMatrix(original[permutation], counts, 2.4, block_size=7)
    for i in range(len(counts)):
        torch.testing.assert_close(a.get_column(i), b.get_column(i), atol=1e-13, rtol=1e-13)


def test_atomic_temporary_axes_remain_bounded(monkeypatch):
    counts, pairs = _data()
    matrix = ExactGaussianKMEKernelMatrix(pairs, counts, 2.1, block_size=2)
    inner = matrix._inner
    shapes = []
    def observed(x, y):
        result = inner(x, y)
        shapes.append(result.shape)
        assert result.shape[-2] <= 2
        assert result.shape[-1] <= 2
        return result
    monkeypatch.setattr(matrix, '_inner', observed)
    matrix.get_diag()
    matrix.get_column(4)
    assert shapes


def test_finite_lru_column_cache_and_diagonal_cache(monkeypatch):
    counts, pairs = _data()
    matrix = ExactGaussianKMEKernelMatrix(pairs, counts, 2.1, block_size=4, cache_columns=2)
    diagonal = matrix.get_diag()
    first = matrix.get_column(0)
    matrix.get_column(1)
    assert matrix.get_column(torch.tensor(0)) is first
    matrix.get_column(2)
    assert list(matrix._columns) == [0, 2]
    matrix.clear_column_cache()
    assert not matrix._columns
    assert matrix.get_diag() is diagonal
    matrix.cache_columns = 0
    matrix.get_column(0)
    assert not matrix._columns


def test_only_roundoff_negative_structure_distances_are_clamped(monkeypatch):
    matrix = ExactGaussianKMEKernelMatrix(torch.tensor([[0.], [1.]]), [1, 1], 1.)
    diag = matrix.get_diag()
    monkeypatch.setattr(matrix, 'get_column', lambda i: diag + 1e-15)
    assert torch.equal(matrix.get_sq_dists(0), torch.zeros(2, dtype=torch.float64))
    monkeypatch.setattr(matrix, 'get_column', lambda i: diag + 1e-4)
    with pytest.raises(ValueError, match='materially negative'):
        matrix.get_sq_dists(0)


@pytest.mark.parametrize('kwargs', [
    {'sigma': 0}, {'sigma': -1}, {'sigma': float('nan')}, {'sigma': float('inf')},
    {'block_size': 0}, {'block_size': 1.5}, {'block_size': True}, {'cache_columns': -1},
    {'counts': []}, {'counts': [0, 2]}, {'counts': [1.0, 1.0]},
    {'counts': [True, 1]}, {'counts': [1]}, {'counts': torch.tensor([[1, 1]])},
    {'raw': torch.ones(2, 1, dtype=torch.long)}, {'raw': torch.ones(2, 0)},
    {'raw': torch.tensor([[1.], [float('nan')]])},
    {'raw': torch.tensor([[1e200], [1e200]], dtype=torch.float64)},
    {'raw': []}, {'raw': [(torch.ones(2, 1),)]},
    {'raw': [(torch.ones(2, 1), torch.ones(3, 1))]},
    {'raw': [(torch.ones(2, 1), torch.ones(2, 1, dtype=torch.float64))]},
])
def test_invalid_inputs_fail_explicitly(kwargs):
    arguments = dict(raw=torch.ones(2, 1), counts=[1, 1], sigma=1.)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        ExactGaussianKMEKernelMatrix(**arguments)


@pytest.mark.parametrize('index', [-1, 2, 1.2, True])
def test_invalid_column_indices(index):
    matrix = ExactGaussianKMEKernelMatrix(torch.ones(2, 1), [1, 1], 1.)
    with pytest.raises((ValueError, IndexError)):
        matrix.get_column(index)
    with pytest.raises((ValueError, IndexError)):
        matrix.get_sq_dists(index)


def test_kernel_is_not_rbf_of_mean():
    raw = torch.tensor([[-1.], [1.], [0.], [0.]])
    matrix = ExactGaussianKMEKernelMatrix(raw, [2, 2], 1.)
    assert matrix.get_sq_dists(0)[1] > 0.1  # Both structure means are zero.
