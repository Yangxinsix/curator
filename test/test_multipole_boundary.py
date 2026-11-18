"""Physical boundary checks for charge/dipole states without state labels."""
from dataclasses import asdict

import pytest
import torch

from curator.data import properties
from curator.data.variational_data import canonical_boundary_hash
from curator.variational.multipole_boundary import MultipoleBoundary, boundary_for_geometry
from curator.variational.spec import BoundarySpec, ConstraintSpec, ControlSpec, GaugeSpec, GroupSpec, VariableSpec


DTYPE = torch.float64


def charge_spec(total=1.0):
    return BoundarySpec(
        VariableSpec('charge', 'charge', unit='e', conjugate_unit='V'),
        groups=(GroupSpec.all(),),
        equalities=(ConstraintSpec.group_sum('total', 'all', total, unit='e'),),
    )


def positions():
    return torch.tensor([[0.2, -0.4, 0.1], [1.1, 0.3, -0.2], [-0.6, 0.8, 0.5]], dtype=DTYPE)


@pytest.mark.parametrize('total', [-1.0, 1.0])
def test_nonzero_total_charge_projection_preserves_dipoles(total):
    boundary = MultipoleBoundary.compile(charge_spec(total), positions())
    state = torch.tensor([[0.3, 1., 2., 3.], [-0.2, -1., 4., 2.], [0.8, 3., -2., 5.]], dtype=DTYPE)
    projected = boundary.project(state)
    torch.testing.assert_close(projected[:, 0].sum(), state.new_tensor(total))
    torch.testing.assert_close(projected[:, 1:], state[:, 1:], rtol=0, atol=0)
    torch.testing.assert_close(boundary.project(projected), projected)
    tangent = boundary.tangent(state)
    torch.testing.assert_close(tangent[:, 0].sum(), state.new_zeros(()), atol=1.e-14, rtol=0)
    torch.testing.assert_close(tangent[:, 1:], state[:, 1:], rtol=0, atol=0)


def test_region_constraints_do_not_mix_charge_and_vector_components():
    spec = BoundarySpec(
        VariableSpec('charge', 'charge', unit='e', conjugate_unit='V'),
        groups=(GroupSpec.from_indices('left', [0, 1]), GroupSpec.from_indices('right', [2])),
        equalities=(ConstraintSpec.group_sum('qleft', 'left', 0.7, unit='e'),
                    ConstraintSpec.group_sum('qright', 'right', -0.7, unit='e')),
    )
    boundary = MultipoleBoundary.compile(spec, positions())
    state = torch.arange(12, dtype=DTYPE).reshape(3, 4) / 10
    projected = boundary.project(state)
    torch.testing.assert_close(projected[:2, 0].sum(), state.new_tensor(0.7))
    torch.testing.assert_close(projected[2, 0], state.new_tensor(-0.7))
    torch.testing.assert_close(projected[:, 1:], state[:, 1:], rtol=0, atol=0)


def test_uniform_field_energy_force_and_state_derivative_signs():
    pos = positions().requires_grad_(True)
    field = torch.tensor([0.3, -0.2, 0.7], dtype=DTYPE)
    phi = torch.tensor([0.1, -0.6, 0.2], dtype=DTYPE)
    state = torch.tensor([[0.5, 0.2, 0.1, -0.2], [-0.3, 0.4, -0.1, 0.3], [0.8, 0.2, -0.3, 0.5]], dtype=DTYPE, requires_grad=True)
    boundary = MultipoleBoundary.compile(charge_spec(), pos, potential=phi, uniform_field=field)
    energy = boundary.work(state)
    dipole = (state[:, :1] * pos + state[:, 1:]).sum(0)
    torch.testing.assert_close(energy, state[:, 0] @ phi - dipole @ field)
    grad_state, grad_pos = torch.autograd.grad(energy, (state, pos))
    torch.testing.assert_close(grad_state[:, 0], phi - pos.detach() @ field)
    torch.testing.assert_close(grad_state[:, 1:], -field.expand(3, -1))
    torch.testing.assert_close(-grad_pos, state.detach()[:, :1] * field)


def test_nonuniform_external_potential_keeps_coordinate_derivatives():
    state = torch.tensor([[0.5, 0.2, 0.1, -0.2], [-0.3, 0.4, -0.1, 0.3], [0.8, 0.2, -0.3, 0.5]], dtype=DTYPE)
    alpha = 0.17

    def energy(pos):
        phi = alpha * pos.square().sum(-1)
        electric = -2 * alpha * pos
        boundary = MultipoleBoundary.compile(charge_spec(), pos, potential=phi, local_field=electric)
        return boundary.work(state)

    pos = positions().requires_grad_(True)
    force = -torch.autograd.grad(energy(pos), pos)[0]
    torch.testing.assert_close(force, -2 * alpha * (state[:, :1] * pos.detach() + state[:, 1:]))
    eps = 1.e-6
    finite = torch.empty_like(pos)
    for index in range(pos.numel()):
        plus, minus = pos.detach().clone(), pos.detach().clone()
        plus.view(-1)[index] += eps
        minus.view(-1)[index] -= eps
        finite.view(-1)[index] = -(energy(plus) - energy(minus)) / (2 * eps)
    torch.testing.assert_close(force, finite, atol=2.e-10, rtol=1.e-8)


def test_constant_potential_gauge_changes_energy_but_not_proposal_inputs():
    pos = positions()
    phi = torch.tensor([0.4, -0.3, 0.2], dtype=DTYPE)
    original = MultipoleBoundary.compile(charge_spec(-1), pos, potential=phi)
    shifted = MultipoleBoundary.compile(charge_spec(-1), pos, potential=phi + 2.7)
    state = original.project(torch.arange(12, dtype=DTYPE).reshape(3, 4) / 20)
    torch.testing.assert_close(shifted.work(state) - original.work(state), state.new_tensor(-2.7))
    for name in ('origin', 'network_field', 'constraint_matrix', 'constraint_target'):
        torch.testing.assert_close(shifted.proposal_inputs()[name], original.proposal_inputs()[name])


def test_voltage_control_has_reservoir_work_sign():
    spec = BoundarySpec(
        VariableSpec('charge', 'charge', unit='e', conjugate_unit='V'),
        groups=(GroupSpec.all(), GroupSpec.from_indices('left', [0, 1]), GroupSpec.from_indices('right', [2])),
        equalities=(ConstraintSpec.group_sum('total', 'all', 0., unit='e'),),
        controls=(ControlSpec.group_potential('vl', 'left', 1.2, unit='V'),
                  ControlSpec.group_potential('vr', 'right', -0.4, unit='V')),
        gauge=GaugeSpec('allow'),
    )
    boundary = MultipoleBoundary.compile(spec, positions())
    state = torch.tensor([[0.3, 0., 0., 0.], [0.2, 0., 0., 0.], [-0.5, 0., 0., 0.]], dtype=DTYPE, requires_grad=True)
    expected = -1.2 * state[:2, 0].sum() + 0.4 * state[2, 0]
    torch.testing.assert_close(boundary.work(state), expected)
    grad = torch.autograd.grad(boundary.work(state), state)[0]
    torch.testing.assert_close(grad[:, 0], state.new_tensor([-1.2, -1.2, 0.4]))
    torch.testing.assert_close(grad[:, 1:], state.new_zeros((3, 3)))
    torch.testing.assert_close(boundary.proposal_inputs()['network_field'], boundary.tangent(-grad))


def test_detached_boundary_preserves_values_and_removes_external_graph():
    pos = positions().requires_grad_(True)
    field = torch.tensor([0.1, 0.2, -0.3], dtype=DTYPE, requires_grad=True)
    boundary = MultipoleBoundary.compile(charge_spec(), pos, uniform_field=field)
    detached = boundary.detached()
    state = boundary.project(torch.zeros(3, 4, dtype=DTYPE))
    torch.testing.assert_close(detached.work(state), boundary.work(state))
    assert not detached.external_conjugate.requires_grad
    assert not detached.matrix.requires_grad
    assert not detached.target.requires_grad


@pytest.mark.parametrize('total', [-1.0, 1.0])
def test_plain_geometry_uses_declared_total_charge(total):
    data = {properties.positions: positions(), properties.total_charge: torch.tensor([total], dtype=DTYPE)}
    boundary = boundary_for_geometry(data, 0, 0, 3)
    state = boundary.project(torch.zeros(3, 4, dtype=DTYPE))
    torch.testing.assert_close(state[:, 0].sum(), state.new_tensor(total))


def test_structured_charge_boundary_is_authoritative():
    raw = asdict(charge_spec(-1))
    data = {
        properties.positions: positions(), properties.total_charge: torch.tensor([1.], dtype=DTYPE),
        'variational': {
            'condition_geometry_index': torch.tensor([0]), 'boundary_spec': [raw],
            'boundary_hash': [canonical_boundary_hash(raw)],
        },
    }
    boundary = boundary_for_geometry(data, 0, 0, 3)
    torch.testing.assert_close(boundary.project(torch.zeros(3, 4, dtype=DTYPE))[:, 0].sum(), torch.tensor(-1., dtype=DTYPE))
