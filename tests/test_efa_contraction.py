"""Equivalent EFA contractions, including derivatives used in E/F/q training."""
import copy

import pytest
import torch

from curator.layer.efa.core import EuclideanFastAttention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires allocated GPU")


@pytest.fixture(autouse=True)
def _gpu_tensors():
    with torch.device("cuda"):
        yield


def _inputs(atoms=17, width=112):
    generator = torch.Generator(device="cuda").manual_seed(20261022)
    shapes = [(atoms, 4, 32), (atoms, 4, 32), (atoms, width),
              (atoms, 3), (atoms,)]
    return [torch.randn(shape, generator=generator, dtype=torch.float64)
            for shape in shapes]


def _energy_and_derivatives(core, inputs, cell):
    query, key, base_value, positions, charge = [
        x.clone().requires_grad_() for x in inputs]
    variables = (query, key, base_value, positions, charge)
    value = base_value * charge[:, None]
    update = core(query, key, value, positions, cell)
    energy = torch.nn.functional.softplus(update).mean() + 0.05 * charge.square().sum()
    first = torch.autograd.grad(energy, variables, create_graph=True)
    force = -first[3]
    residual = first[4] - first[4].mean()
    # Parameter/input gradients of a joint energy, force and residual objective
    # exercise the second derivatives needed by the production training step.
    loss = energy.square() + force.square().mean() + residual.square().mean()
    second = torch.autograd.grad(loss, variables)
    return (update, energy, force, residual, *first, *second)


@pytest.mark.parametrize("width", [32, 64, 112])
@pytest.mark.parametrize("periodic", [False, True])
def test_pairwise_matches_energy_force_stationarity_and_second_derivatives(width, periodic):
    linear = EuclideanFastAttention(contraction="linear").double()
    pairwise = copy.deepcopy(linear)
    pairwise.contraction = "pairwise"
    cell = torch.tensor([[10., 0.2, 0.1], [0., 11., 0.3], [0., 0., 12.]],
                        dtype=torch.float64) if periodic else None
    inputs = _inputs(width=width)
    expected = _energy_and_derivatives(linear, inputs, cell)
    actual = _energy_and_derivatives(pairwise, inputs, cell)
    for lhs, rhs in zip(actual, expected):
        torch.testing.assert_close(lhs, rhs, rtol=2e-10, atol=2e-11)


def test_auto_uses_cost_and_memory_cap():
    core = EuclideanFastAttention(contraction="auto", pairwise_max_atoms=256).double()
    # Test policy using cheap dummy prepared shapes; no large contraction needed.
    assert core._contraction_order(torch.empty(17, 110, 128), torch.empty(17, 112)) == "pairwise"
    assert core._contraction_order(torch.empty(120, 110, 128), torch.empty(120, 32)) == "linear"
    assert core._contraction_order(torch.empty(257, 110, 128), torch.empty(257, 512)) == "linear"
    query, key, value, positions, _ = _inputs(width=64)
    expected = copy.deepcopy(core)
    expected.contraction = "linear"
    torch.testing.assert_close(core(query, key, value, positions),
                               expected(query, key, value, positions),
                               rtol=2e-10, atol=2e-11)


def test_old_serialized_core_keeps_linear_order_and_state_dict():
    core = EuclideanFastAttention().double()
    original_keys = set(core.state_dict())
    del core.contraction
    del core.pairwise_max_atoms
    query, key, value, positions, _ = _inputs(width=64)
    prepared = core.prepare(query, key, positions)
    assert core._contraction_order(prepared.query, value) == "linear"
    expected = torch.einsum("nmd,mdv,m->nv", prepared.query,
                            torch.einsum("nmd,nv->mdv", prepared.key, value),
                            prepared.weights)
    torch.testing.assert_close(core.aggregate(prepared, value), expected, rtol=0, atol=0)
    assert set(core.state_dict()) == original_keys
    # Preserve nn.Module.apply, which traverses submodules during initialization.
    assert core.apply(lambda module: None) is core


def test_prepare_reuse_different_value_widths_and_structure_isolation():
    core = EuclideanFastAttention(contraction="pairwise").double()
    query, key, value, positions, _ = [x.requires_grad_() for x in _inputs(width=112)]
    prepared = core.prepare(query, key, positions)
    first = core.aggregate(prepared, value)
    # Preparing another structure must not overwrite or connect the first graph.
    other = [x.requires_grad_() for x in _inputs(atoms=19, width=64)]
    other_prepared = core.prepare(other[0], other[1], other[3])
    core.aggregate(other_prepared, other[2])
    second = core.aggregate(prepared, value[:, :64])
    torch.testing.assert_close(second, first[:, :64], rtol=2e-10, atol=2e-11)
    assert torch.autograd.grad(second.sum(), other[3], allow_unused=True,
                               retain_graph=True)[0] is None
    reused_grad = torch.autograd.grad(first.square().sum() + second.square().sum(),
                                     positions)[0]
    standalone = (core(query, key, value, positions).square().sum()
                  + core(query, key, value[:, :64], positions).square().sum())
    torch.testing.assert_close(reused_grad, torch.autograd.grad(standalone, positions)[0],
                               rtol=2e-10, atol=2e-11)


def test_forced_pairwise_rejects_oversized_matrix():
    core = EuclideanFastAttention(contraction="pairwise", pairwise_max_atoms=16)
    with pytest.raises(ValueError, match="pairwise_max_atoms"):
        core._contraction_order(torch.empty(17, 110, 128), torch.empty(17, 64))


def test_large_structure_auto_reproduces_linear_forward():
    core = EuclideanFastAttention(hidden_features=4, num_directions=None,
                                  contraction="auto").double()
    query = torch.linspace(-1., 1., 257 * 4, dtype=torch.float64).reshape(257, 1, 4)
    positions = torch.linspace(-2., 2., 257 * 3, dtype=torch.float64).reshape(257, 3)
    value = query[:, 0]
    cell = 10 * torch.eye(3, dtype=torch.float64)
    prepared = core.prepare(query, query, positions, cell)
    assert core._contraction_order(prepared.query, value) == "linear"
    automatic = core.aggregate(prepared, value)
    core.contraction = "linear"
    torch.testing.assert_close(automatic, core(query, query, value, positions, cell),
                               rtol=0, atol=0)
