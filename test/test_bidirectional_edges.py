from __future__ import annotations

import torch

from curator.data import EnsureBidirectionalEdges, properties


def _edge_key(pair, displacement):
    return (int(pair[0]), int(pair[1]), *map(float, displacement))


def test_adds_missing_reverse_and_preserves_edge_fields():
    data = {
        properties.edge_idx: torch.tensor([[0, 1]], dtype=torch.long),
        properties.edge_diff: torch.tensor([[0.4, -0.2, 0.1]]),
        properties.edge_dist: torch.tensor([0.4583]),
        properties.cell_displacements: torch.tensor([[1.0, 0.0, -2.0]]),
        properties.n_pairs: torch.tensor([1], dtype=torch.long),
    }

    output = EnsureBidirectionalEdges()(data)

    assert torch.equal(
        output[properties.edge_idx],
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    assert torch.allclose(
        output[properties.edge_diff],
        torch.tensor([[0.4, -0.2, 0.1], [-0.4, 0.2, -0.1]]),
    )
    assert torch.allclose(output[properties.edge_dist], torch.tensor([0.4583, 0.4583]))
    assert torch.allclose(
        output[properties.cell_displacements],
        torch.tensor([[1.0, 0.0, -2.0], [-1.0, 0.0, 2.0]]),
    )
    assert torch.equal(output[properties.n_pairs], torch.tensor([2]))


def test_deduplicates_edges_without_overwriting_existing_reverse_values():
    data = {
        properties.edge_idx: torch.tensor(
            [[0, 1], [0, 1], [1, 0]], dtype=torch.long
        ),
        properties.edge_diff: torch.tensor(
            [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]
        ),
        properties.edge_dist: torch.tensor([0.5, 9.0, 0.6]),
        properties.n_pairs: torch.tensor([3]),
    }

    output = EnsureBidirectionalEdges()(data)

    assert torch.equal(
        output[properties.edge_idx],
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    assert torch.allclose(output[properties.edge_dist], torch.tensor([0.5, 0.6]))
    assert torch.equal(output[properties.n_pairs], torch.tensor([2]))


def test_periodic_self_edge_gets_distinct_reverse_image():
    data = {
        properties.edge_idx: torch.tensor([[0, 0]], dtype=torch.long),
        properties.edge_diff: torch.tensor([[2.0, 0.0, 0.0]]),
        properties.cell_displacements: torch.tensor([[2.0, 0.0, 0.0]]),
    }

    output = EnsureBidirectionalEdges()(data)

    keys = {
        _edge_key(pair, displacement)
        for pair, displacement in zip(
            output[properties.edge_idx].tolist(),
            output[properties.edge_diff].tolist(),
        )
    }
    assert keys == {
        (0, 0, 2.0, 0.0, 0.0),
        (0, 0, -2.0, 0.0, 0.0),
    }
    assert torch.allclose(
        output[properties.cell_displacements],
        torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]),
    )


def test_copies_registered_edge_fields_to_generated_reverse(monkeypatch):
    custom_edge_field = "_edge_payload"
    monkeypatch.setattr(
        properties,
        "_EDGE_FIELDS",
        properties._EDGE_FIELDS | {custom_edge_field},
    )
    data = {
        properties.edge_idx: torch.tensor([[0, 1]], dtype=torch.long),
        properties.edge_diff: torch.tensor([[0.2, 0.0, 0.0]]),
        custom_edge_field: torch.tensor([[2.0, 3.0]]),
    }

    output = EnsureBidirectionalEdges()(data)

    assert torch.equal(
        output[custom_edge_field],
        torch.tensor([[2.0, 3.0], [2.0, 3.0]]),
    )


def test_preserves_batch_edge_grouping_and_updates_each_pair_count():
    data = {
        properties.edge_idx: torch.tensor(
            [[0, 1], [1, 0], [2, 3]], dtype=torch.long
        ),
        properties.edge_diff: torch.tensor(
            [[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0], [0.8, 0.0, 0.0]]
        ),
        properties.edge_dist: torch.tensor([0.3, 0.3, 0.8]),
        properties.n_pairs: torch.tensor([2, 1]),
    }

    output = EnsureBidirectionalEdges()(data)

    assert torch.equal(output[properties.n_pairs], torch.tensor([2, 2]))
    assert torch.equal(
        output[properties.edge_idx],
        torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]], dtype=torch.long),
    )
    assert torch.allclose(
        output[properties.edge_dist],
        torch.tensor([0.3, 0.3, 0.8, 0.8]),
    )


def test_accepts_an_empty_batch():
    data = {
        properties.edge_idx: torch.empty((0, 2), dtype=torch.long),
        properties.edge_diff: torch.empty((0, 3)),
        properties.n_pairs: torch.empty((0,), dtype=torch.long),
    }

    output = EnsureBidirectionalEdges()(data)

    assert output[properties.edge_idx].shape == (0, 2)
    assert output[properties.edge_diff].shape == (0, 3)
    assert output[properties.n_pairs].shape == (0,)
