"""Physical isolated Coulomb boundary, derivative training, and Ewald regression."""
import pytest
import torch
from torch import nn

from curator.data import properties
from curator.layer._ewald import EwaldSummation
from curator.layer._quadratic_taylor_energy import QuadraticTaylorEnergy
from curator.variational.backends.ewald import EwaldChargeOperator
from test_ewald_theory import FRACTIONAL_POSITIONS, TRICLINIC_CELL, _periodic_data

DTYPE = torch.float64


def _isolated(position):
    return {
        properties.positions: position,
        properties.n_atoms: torch.tensor([len(position)]),
        properties.cell: position.new_zeros((1, 3, 3)),
        properties.pbc: torch.zeros((1, 3), dtype=torch.bool),
    }


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('charges', [[0.7, -0.4], [0.7, 0.4]])
def test_isolated_two_charge_energy_force_and_hvp(charges, device):
    pos = torch.tensor([[0., 0., 0.], [8., 0., 0.]], dtype=DTYPE, device=device, requires_grad=True)
    q = torch.tensor(charges, dtype=DTYPE, device=device, requires_grad=True)
    data = _isolated(pos)
    # Separation exceeds both cutoffs: isolated interactions must remain intact.
    op = EwaldChargeOperator(EwaldSummation(cutoff=2., k_cutoff=1., alpha=0.8))
    kernel = op.geometry_kernels(data)[0]
    c = EwaldSummation.CONV_FACT
    torch.testing.assert_close(kernel, kernel.T, rtol=0., atol=0.)
    torch.testing.assert_close(kernel.diag(), torch.zeros(2, dtype=DTYPE, device=device))
    energy = op.energy(data, q)[0]
    torch.testing.assert_close(energy, c * q[0] * q[1] / 8.)
    force = -torch.autograd.grad(energy, pos, create_graph=True)[0]
    expected_force = torch.zeros_like(pos)
    expected_force[0, 0] = -c * q[0] * q[1] / 64.
    expected_force[1, 0] = c * q[0] * q[1] / 64.
    torch.testing.assert_close(force, expected_force)
    gradient = torch.autograd.grad(energy, q, create_graph=True)[0]
    torch.testing.assert_close(gradient, op.matvec(data, q))
    v = q.new_tensor([0.2, -0.3])
    hvp = torch.autograd.grad(gradient, q, grad_outputs=v, create_graph=True)[0]
    torch.testing.assert_close(hvp, kernel @ v)
    torch.testing.assert_close(op.bind(data).kernel(), kernel)
    mixed = torch.autograd.grad(force.square().sum(), q)[0]
    assert torch.isfinite(mixed).all() and mixed.abs().max() > 0


def test_isolated_position_charge_gradgrad_and_symmetries():
    op = EwaldChargeOperator(EwaldSummation())
    pos = torch.tensor([[0.1, 0.2, 0.3], [3.1, -0.4, 0.2], [0.3, 2.8, -0.2]], dtype=DTYPE, requires_grad=True)
    q = torch.tensor([0.3, -0.2, 0.7], dtype=DTYPE, requires_grad=True)
    energy = lambda r, charge: op.energy(_isolated(r), charge).sum()
    assert torch.autograd.gradcheck(energy, (pos, q))
    assert torch.autograd.gradgradcheck(energy, (pos, q))
    reference = op.geometry_kernels(_isolated(pos))[0]
    rotation = pos.new_tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    torch.testing.assert_close(op.geometry_kernels(_isolated(pos @ rotation + 31.))[0], reference)
    different_split = EwaldChargeOperator(EwaldSummation(alpha=0.2, cutoff=0.1, k_cutoff=0.1))
    torch.testing.assert_close(different_split.geometry_kernels(_isolated(pos))[0], reference, rtol=0., atol=0.)
    # An isolated molecule's empty or arbitrary cell is not a hidden periodic box.
    changed = _isolated(pos)
    changed[properties.cell] = torch.eye(3, dtype=DTYPE)[None] * 100.
    torch.testing.assert_close(op.geometry_kernels(changed)[0], reference, rtol=0., atol=0.)
    del changed[properties.cell]
    torch.testing.assert_close(op.geometry_kernels(changed)[0], reference, rtol=0., atol=0.)


def test_isolated_single_charge_zero_self_and_invalid_geometries():
    op = EwaldChargeOperator(EwaldSummation())
    one = torch.zeros((1, 3), dtype=DTYPE, requires_grad=True)
    energy = op.energy(_isolated(one), one.new_tensor([2.]))
    torch.testing.assert_close(energy, one.new_zeros(1), rtol=0., atol=0.)
    torch.testing.assert_close(torch.autograd.grad(energy.sum(), one)[0], torch.zeros_like(one))
    with pytest.raises(ValueError, match='same position'):
        op.geometry_kernels(_isolated(torch.zeros((2, 3), dtype=DTYPE)))
    data = _isolated(one)
    data[properties.pbc] = torch.tensor([[True, True, False]])
    with pytest.raises(ValueError, match='partial periodicity'):
        op.geometry_kernels(data)
    del data[properties.pbc]
    with pytest.raises(ValueError, match='explicit'):
        op.geometry_kernels(data)


def test_periodic_path_matches_existing_ewald_and_mixed_structure_batch():
    ewald = EwaldSummation(alpha=0.4, acc_factor=8.)
    op = EwaldChargeOperator(ewald)
    periodic = _periodic_data(TRICLINIC_CELL, FRACTIONAL_POSITIONS, [0.7, -0.2],
                              ewald, neighbor_cutoff=ewald.cutoff, charge_grad=True)
    periodic[properties.pbc] = torch.ones((1, 3), dtype=torch.bool)
    periodic_kernel = op.geometry_kernels(periodic)[0]
    torch.testing.assert_close(periodic_kernel, op.kernels(periodic)[0], rtol=1e-13, atol=1e-13)
    q = periodic[properties.atomic_charge]
    torch.testing.assert_close(0.5 * q @ periodic_kernel @ q, ewald(periodic).sum(), rtol=1e-12, atol=1e-12)
    isolated_pos = torch.tensor([[0., 0., 0.], [12., 0., 0.]], dtype=DTYPE)
    isolated_kernel = op.geometry_kernels(_isolated(isolated_pos))[0]
    batch = {
        properties.positions: torch.cat((periodic[properties.positions], isolated_pos)),
        properties.n_atoms: torch.tensor([len(q), 2]),
        properties.cell: torch.cat((periodic[properties.cell].reshape(1, 3, 3), torch.zeros((1, 3, 3), dtype=DTYPE))),
        properties.pbc: torch.tensor([[True, True, True], [False, False, False]]),
    }
    kernels = op.geometry_kernels(batch)
    torch.testing.assert_close(kernels[0], periodic_kernel, rtol=0., atol=0.)
    torch.testing.assert_close(kernels[1], isolated_kernel, rtol=0., atol=0.)
    charges = torch.cat((q, q.new_tensor([0.2, 0.8])))
    torch.testing.assert_close(op.energy(batch, charges), torch.stack([
        0.5 * q @ periodic_kernel @ q, 0.5 * charges[-2:] @ isolated_kernel @ charges[-2:]]))


def test_derivative_ewald_taylor_uses_isolated_kernel_and_trains_force():
    class ScalarEnergy(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.8, dtype=DTYPE))

        def forward(self, context, state):
            return (self.weight * state + 10. * state.square()).sum()

        def state_hessian_diagonal(self, context, state, create_graph=True):
            return torch.ones_like(state) * 20.

    base = ScalarEnergy()
    wrapper = QuadraticTaylorEnergy(base, curvature_mode='diagonal_ewald')
    pos = torch.tensor([[0., 0., 0.], [3., 0., 0.]], dtype=DTYPE, requires_grad=True)
    context = wrapper.geometry_context(_isolated(pos))[0]
    # Proposal depends on geometry and a parameter, testing the training path.
    proposal_weight = torch.tensor(0.2, dtype=DTYPE, requires_grad=True)
    q = proposal_weight * (pos[:, :1] + 0.1)
    actual = wrapper(context, q)
    expected = (base.weight * q + 10. * q.square()).sum() + EwaldSummation.CONV_FACT * q[0, 0] * q[1, 0] / 3.
    torch.testing.assert_close(actual, expected)
    force = -torch.autograd.grad(actual, pos, create_graph=True)[0]
    gradients = torch.autograd.grad(force.square().sum() + actual.square(), (base.weight, proposal_weight))
    assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in gradients)
