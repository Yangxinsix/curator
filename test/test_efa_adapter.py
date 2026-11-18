import copy
import io

import pytest
import torch
from e3nn import o3

from curator.data import properties as p
from curator.model.features import IrrepsFeatureSpec, ScalarVectorFeatureSpec
from curator.layer._efa_adapter import EFAAdapter, EuclideanFastAttention, _lebedev_grid


def sample(n=4):
    torch.manual_seed(11)
    return {
        p.positions: (torch.randn(n, 3, dtype=torch.float64) * 2).requires_grad_(),
        "features": torch.randn(n, 5, dtype=torch.float64),
        p.n_atoms: torch.tensor([n]),
        p.image_idx: torch.zeros(n, dtype=torch.long),
        p.pbc: torch.zeros(1, 3, dtype=torch.bool),
        p.energy: torch.zeros(1, dtype=torch.float64),
    }


def adapter(directions=110):
    return EFAAdapter(IrrepsFeatureSpec("features", "2x0e+1x1o"),
                      hidden_features=4, num_directions=directions).double()


@pytest.mark.parametrize("num", [50, 110, 194])
def test_quadrature(num):
    r, w = _lebedev_grid(num)
    torch.testing.assert_close(r.norm(dim=-1), torch.ones(num, dtype=r.dtype), atol=3e-14, rtol=0)
    torch.testing.assert_close(w.sum(), torch.ones((), dtype=w.dtype), atol=2e-14, rtol=0)
    torch.testing.assert_close(torch.einsum("mi,mj,m->ij", r, r, w), torch.eye(3).double()/3,
                               atol=2e-14, rtol=0)


def test_linear_attention_matches_explicit_dense_erope():
    torch.manual_seed(7)
    core = EuclideanFastAttention(hidden_features=4).double()
    q, k = [torch.randn(5, 4, 4, dtype=torch.float64) for _ in range(2)]
    v = torch.randn(5, 3, dtype=torch.float64)
    r = torch.randn(5, 3, dtype=torch.float64)
    # The quadratic oracle is intentionally confined to this tiny test.
    qr, kr = core.rotary(q, r), core.rotary(k, r)
    a = torch.einsum("imd,jmd,m->ij", qr, kr, core.weights) / 4.
    torch.testing.assert_close(core(q, k, v, r), a @ v, atol=2e-14, rtol=2e-13)


def test_batch_permutation_and_translation():
    m = adapter()
    a, b = sample(3), sample(4)
    a[p.positions] = a[p.positions] + 5
    batch = {
        p.positions: torch.cat([a[p.positions], b[p.positions]]),
        "features": torch.cat([a["features"], b["features"]]),
        p.n_atoms: torch.tensor([3, 4]),
        p.image_idx: torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        p.pbc: torch.zeros(2, 3, dtype=torch.bool),
        p.energy: torch.zeros(2, dtype=torch.float64),
    }
    expected = torch.cat([m(dict(a))[p.energy], m(dict(b))[p.energy]])
    actual = m(dict(batch))[p.energy]
    torch.testing.assert_close(actual, expected)
    perm = torch.tensor([4, 0, 6, 2, 5, 1, 3])
    permuted = dict(batch)
    for key in [p.positions, "features", p.image_idx]:
        permuted[key] = batch[key][perm]
    permuted[p.positions] = permuted[p.positions] + torch.tensor([170., -92., 30.])
    torch.testing.assert_close(m(permuted)[p.energy], expected, atol=2e-14, rtol=2e-12)


def test_energy_force_rotation_covariance():
    m = adapter()
    data = sample()
    r = o3.rand_matrix(dtype=torch.float64)
    irreps = m.feature_spec.irreps
    rotated = dict(data)
    rotated[p.positions] = (data[p.positions].detach() @ r.T).requires_grad_()
    rotated["features"] = data["features"] @ irreps.D_from_matrix(r).T
    e1, e2 = m(dict(data))[p.energy], m(rotated)[p.energy]
    f1 = -torch.autograd.grad(e1.sum(), data[p.positions])[0]
    f2 = -torch.autograd.grad(e2.sum(), rotated[p.positions])[0]
    torch.testing.assert_close(e1, e2, atol=2e-10, rtol=2e-7)
    torch.testing.assert_close(f1 @ r.T, f2, atol=2e-10, rtol=2e-6)


def test_energy_force_backward_and_finite_difference():
    m = adapter()
    data = sample()
    data["features"].requires_grad_()
    energy = m(dict(data))[p.energy]
    forces = -torch.autograd.grad(energy.sum(), data[p.positions], create_graph=True)[0]
    assert forces.abs().max() > 1e-7
    step = 1e-5
    displaced = []
    for sign in [1, -1]:
        d = dict(data)
        d[p.positions] = data[p.positions].detach().clone()
        d[p.positions][1, 0] += sign * step
        displaced.append(m(d)[p.energy].sum())
    derivative = (displaced[0] - displaced[1]) / (2 * step)
    torch.testing.assert_close(-forces[1, 0], derivative, atol=1e-9, rtol=2e-6)
    loss = (energy - .2).square().mean() + (forces + .1).square().mean()
    loss.backward()
    for module in [m.query, m.key, m.value, m.readout]:
        gradients = [x.grad for x in module.parameters()]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    assert data["features"].grad.abs().sum() > 0


def test_binding_serialization_and_scalar_vector_spec():
    m = adapter()
    state = copy.deepcopy(m.state_dict())
    m.bind(m.feature_spec)
    for name, tensor in m.state_dict().items():
        torch.testing.assert_close(tensor, state[name])
    with pytest.raises(ValueError, match="different"):
        m.bind(IrrepsFeatureSpec("features", "5x0e"))
    buffer = io.BytesIO()
    torch.save(m, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    data = sample()
    torch.testing.assert_close(m(dict(data))[p.energy], restored(dict(data))[p.energy])
    n = EFAAdapter(ScalarVectorFeatureSpec("s", "v", 2, 1), hidden_features=4).double()
    other = dict(data)
    other["s"] = other["features"][:, :2]
    other["v"] = other["features"][:, 2:, None]
    assert torch.isfinite(n(other)[p.energy]).all()


def test_partial_periodic_or_missing_boundary_rejected():
    m = adapter()
    data = sample()
    data[p.pbc][0, 0] = True
    with pytest.raises(NotImplementedError, match="partial"):
        m(dict(data))
    del data[p.pbc]
    with pytest.raises(ValueError, match="explicit pbc"):
        m(data)


def test_late_bind_preserves_requested_precision():
    m = EFAAdapter(hidden_features=4).double()
    m.bind(IrrepsFeatureSpec("features", "2x0e+1x1o"))
    assert m(sample())[p.energy].dtype == torch.float64


def test_spherical_integral_matches_analytic_sinc_and_force():
    """Independent exact spherical integral: mean cos(w*u.dr) = sinc(w*r)."""
    torch.manual_seed(21)
    core = EuclideanFastAttention(hidden_features=4, num_directions=110).double()
    q, k = [torch.randn(2, 3, 4, dtype=torch.float64) for _ in range(2)]
    v = torch.randn(2, 3, dtype=torch.float64)
    r = torch.tensor([[0., 0., 0.], [19., 17., 18.]], dtype=torch.float64,
                     requires_grad=True)
    out = core(q, k, v, r)
    distance = torch.linalg.vector_norm(r[0] - r[1])
    weights = torch.sinc(core.frequencies * distance / torch.pi)
    pair_dot = torch.einsum("ad,ad->d", q[0], k[1]).reshape(2, 2).sum(-1)
    self_dot = (q[0] * k[0]).sum()
    exact = (self_dot * v[0] + (pair_dot * weights).sum() * v[1]) / (12 ** .5)
    # Separation near max_length is the challenging edge of the configured grid.
    torch.testing.assert_close(out[0], exact, atol=2e-7, rtol=2e-6)
    approximate_force = torch.autograd.grad(out[0].sum(), r, retain_graph=True)[0]
    exact_force = torch.autograd.grad(exact.sum(), r)[0]
    torch.testing.assert_close(approximate_force, exact_force, atol=2e-7, rtol=2e-5)

