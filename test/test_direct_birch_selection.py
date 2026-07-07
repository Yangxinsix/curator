import pytest
import torch

from curator.select.kernel import FeatureKernelMatrix
from curator.select.select import _call_selection, direct_birch, max_diag


def test_direct_birch_selects_unique_pool_indices():
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(1, 24, 6, generator=generator)
    matrix = FeatureKernelMatrix(features)

    selected = direct_birch(
        matrix,
        batch_size=8,
        n_clusters=8,
        k=1,
        random_state=11,
    )

    assert selected.shape == (8,)
    assert selected.unique().numel() == 8
    assert int(selected.min()) >= 0
    assert int(selected.max()) < 24


def test_direct_birch_excludes_train_tail():
    generator = torch.Generator().manual_seed(13)
    features = torch.randn(1, 12, 5, generator=generator)
    matrix = FeatureKernelMatrix(features)

    selected = direct_birch(
        matrix,
        batch_size=5,
        n_train=3,
        n_clusters=5,
        k=1,
    )

    assert selected.numel() == 5
    assert int(selected.max()) < 9


def test_call_selection_rejects_unknown_user_kwargs():
    matrix = FeatureKernelMatrix(torch.eye(4).reshape(1, 4, 4))

    with pytest.raises(TypeError, match="unsupported selection_kwargs"):
        _call_selection(
            max_diag,
            selection_kwargs={"not_a_parameter": 1},
            matrix=matrix,
            batch_size=2,
        )
