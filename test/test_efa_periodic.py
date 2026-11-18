"""Author-style lattice-direction EFA, including its image-invariance limitation."""

import io
import math

import pytest
import torch
from e3nn import o3

from curator.data import properties as p
from curator.layer._efa_adapter import EFAAdapter, EuclideanFastAttention
from curator.model.features import IrrepsFeatureSpec


def periodic_sample(n=4):
    generator = torch.Generator().manual_seed(31 + n)
    return {
        p.positions: torch.randn(n, 3, generator=generator, dtype=torch.float64).requires_grad_(),
        "features": torch.randn(n, 5, generator=generator, dtype=torch.float64),
        p.cell: torch.tensor([[3., .2, .1], [.4, 4., .2], [.1, .3, 5.]], dtype=torch.float64),
        p.pbc: torch.ones(1, 3, dtype=torch.bool),
        p.n_atoms: torch.tensor([n]),
        p.image_idx: torch.zeros(n, dtype=torch.long),
        p.energy: torch.zeros(1, dtype=torch.float64),
    }


def periodic_adapter(**kwargs):
    torch.manual_seed(43)
    return EFAAdapter(IrrepsFeatureSpec("features", "2x0e+1x1o"),
                      hidden_features=4, **kwargs).double()


def dense_lattice_attention(query, key, value, positions, cell, frequencies):
    """Independent pairwise trigonometric identity, without rotary()/centering."""
    displacement = positions[:, None, :] - positions[None, :, :]
    phase = torch.einsum("ijd,md,c->ijmc", displacement, cell, frequencies)
    qe, qo = query[..., 0::2], query[..., 1::2]
    ke, ko = key[..., 0::2], key[..., 1::2]
    dot = qe[:, None] * ke[None, :] + qo[:, None] * ko[None, :]
    cross = qe[:, None] * ko[None, :] - qo[:, None] * ke[None, :]
    matrix = (torch.einsum("ijac,ijmc->ij", dot, phase.cos())
              + torch.einsum("ijac,ijmc->ij", cross, phase.sin()))
    return matrix @ value / (3 * math.sqrt(query.shape[1] * query.shape[2]))


def test_periodic_linear_attention_matches_independent_dense_output_and_gradients():
    torch.manual_seed(47)
    core = EuclideanFastAttention(hidden_features=4).double()
    query, key = [torch.randn(4, 3, 4, dtype=torch.float64, requires_grad=True)
                  for _ in range(2)]
    value = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    positions = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    cell = periodic_sample()[p.cell].requires_grad_()
    actual = core(query, key, value, positions, cell=cell)
    expected = dense_lattice_attention(query, key, value, positions, cell, core.frequencies)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-13)
    probe = torch.randn_like(actual)
    inputs = (query, key, value, positions, cell)
    grads_actual = torch.autograd.grad((actual * probe).sum(), inputs)
    grads_expected = torch.autograd.grad((expected * probe).sum(), inputs)
    for actual_grad, expected_grad in zip(grads_actual, grads_expected):
        torch.testing.assert_close(actual_grad, expected_grad, atol=5e-14, rtol=5e-12)


@pytest.mark.parametrize("stacked_cells", [False, True])
@pytest.mark.parametrize("mixed_boundaries", [False, True])
def test_periodic_batch_uses_each_cell_and_preserves_permutation_translation(stacked_cells, mixed_boundaries):
    model = periodic_adapter()
    a, b = periodic_sample(3), periodic_sample(4)
    b[p.cell] = b[p.cell] * 1.7
    if mixed_boundaries:
        a[p.pbc].zero_()
        a[p.cell].zero_()  # Nonperiodic members may have ASE's zero dummy cell.
    batch = {
        p.positions: torch.cat([a[p.positions], b[p.positions]]),
        "features": torch.cat([a["features"], b["features"]]),
        p.cell: torch.stack([a[p.cell], b[p.cell]]) if stacked_cells
                else torch.cat([a[p.cell], b[p.cell]]),
        p.pbc: torch.cat([a[p.pbc], b[p.pbc]]),
        p.n_atoms: torch.tensor([3, 4]),
        p.image_idx: torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        p.energy: torch.zeros(2, dtype=torch.float64),
    }
    expected = torch.cat([model(dict(a))[p.energy], model(dict(b))[p.energy]])
    torch.testing.assert_close(model(dict(batch))[p.energy], expected)
    permutation = torch.tensor([4, 0, 6, 2, 5, 1, 3])
    permuted = dict(batch)
    for name in [p.positions, "features", p.image_idx]:
        permuted[name] = batch[name][permutation]
    shifts = torch.tensor([[70., -31., 25.], [-29., 12., 99.]], dtype=torch.float64)
    permuted[p.positions] = permuted[p.positions] + shifts[permuted[p.image_idx]]
    torch.testing.assert_close(model(permuted)[p.energy], expected, atol=2e-13, rtol=2e-11)


def test_periodic_joint_rotation_of_positions_features_cell_and_forces():
    model = periodic_adapter()
    data = periodic_sample()
    rotation = o3.rand_matrix(dtype=torch.float64)
    rotated = dict(data)
    rotated[p.positions] = (data[p.positions].detach() @ rotation.T).requires_grad_()
    rotated[p.cell] = data[p.cell] @ rotation.T
    rotated["features"] = data["features"] @ model.feature_spec.irreps.D_from_matrix(rotation).T
    original_energy = model(dict(data))[p.energy]
    rotated_energy = model(rotated)[p.energy]
    original_force = -torch.autograd.grad(original_energy.sum(), data[p.positions])[0]
    rotated_force = -torch.autograd.grad(rotated_energy.sum(), rotated[p.positions])[0]
    torch.testing.assert_close(original_energy, rotated_energy, atol=2e-12, rtol=2e-10)
    torch.testing.assert_close(original_force @ rotation.T, rotated_force, atol=2e-12, rtol=2e-9)


def test_periodic_total_force_finite_difference_and_energy_force_training_backward():
    """Forces include both explicit positions and position-dependent backbone features."""
    torch.manual_seed(53)
    model = EFAAdapter(IrrepsFeatureSpec("features", "5x0e"), hidden_features=4).double()
    backbone = torch.nn.Linear(4, 5).double()
    data = periodic_sample()

    def energy_at(positions):
        squared_radius = (positions - positions.mean(0)).square().sum(-1)
        features = torch.stack([squared_radius, squared_radius.square(),
                                squared_radius.sin(), squared_radius.cos()], dim=-1)
        example = dict(data)
        example[p.positions] = positions
        example["features"] = backbone(features)
        example[p.energy] = (.03 * squared_radius.square().sum()).reshape(1)
        return model(example)[p.energy]

    energy = energy_at(data[p.positions])
    force = -torch.autograd.grad(energy.sum(), data[p.positions], create_graph=True)[0]
    step = 1e-5
    offset = torch.zeros_like(data[p.positions])
    offset[1, 0] = step
    finite_difference = (energy_at(data[p.positions].detach() + offset)
                         - energy_at(data[p.positions].detach() - offset)) / (2 * step)
    torch.testing.assert_close(-force[1, 0], finite_difference.squeeze(), atol=2e-8, rtol=3e-6)
    loss = (energy - .2).square().mean() + (force + .1).square().mean()
    loss.backward()
    for module in [backbone, model.query, model.key, model.value, model.readout]:
        gradients = [parameter.grad for parameter in module.parameters()]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0


def test_author_lattice_directions_do_not_enforce_individual_image_invariance():
    """Document the official branch's limitation; do not imply Ewald/image sums."""
    core = EuclideanFastAttention(hidden_features=4).double()
    query = torch.zeros(2, 1, 4, dtype=torch.float64)
    query[..., 2] = 1
    value = torch.tensor([[1.], [2.]], dtype=torch.float64)
    positions = torch.tensor([[0., 0., 0.], [.5, .2, .1]], dtype=torch.float64)
    cell = torch.diag(torch.tensor([2., 3., 4.], dtype=torch.float64))
    shifted = positions.clone()
    shifted[1] += cell[0]
    original = core(query, query, value, positions, cell=cell)
    other_image = core(query, query, value, shifted, cell=cell)
    assert (original - other_image).abs().max() > 1e-3


def test_periodic_only_configuration_and_serialization():
    model = periodic_adapter(num_directions=None, max_frequency=math.pi / 4, max_length=1.)
    data = periodic_sample()
    data[p.pbc] = data[p.pbc].reshape(3)
    expected = model(dict(data))[p.energy]
    stream = io.BytesIO()
    torch.save(model, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    torch.testing.assert_close(restored(dict(data))[p.energy], expected)
    restored_state = periodic_adapter(num_directions=None, max_frequency=math.pi / 4, max_length=1.)
    restored_state.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored_state(dict(data))[p.energy], expected)
    nonperiodic = dict(data)
    nonperiodic[p.pbc] = torch.zeros(3, dtype=torch.bool)
    with pytest.raises(ValueError):
        model(nonperiodic)


@pytest.mark.parametrize("bad_cell", [None, torch.ones(2, 3), torch.zeros(3, 3),
                                       torch.full((3, 3), float("nan"))])
def test_periodic_requires_a_finite_full_rank_cell(bad_cell):
    data = periodic_sample()
    if bad_cell is None:
        del data[p.cell]
    else:
        data[p.cell] = bad_cell.double()
    with pytest.raises(ValueError, match="cell"):
        periodic_adapter()(data)


def test_periodic_rejects_nonboolean_boundary_flags():
    data = periodic_sample()
    data[p.pbc] = data[p.pbc].float()
    with pytest.raises((TypeError, ValueError), match="bool"):
        periodic_adapter()(data)


def test_single_frequency_matches_author_initialization():
    # One frequency means the upper endpoint, not torch.linspace(..., steps=1)'s zero.
    core = EuclideanFastAttention(hidden_features=2, num_directions=None,
                                  max_frequency=math.pi / 4, max_length=2.).double()
    torch.testing.assert_close(core.frequencies, torch.tensor([math.pi / 8], dtype=torch.float64))
