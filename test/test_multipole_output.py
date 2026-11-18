"""End-to-end q/dipole adapter derivative and parameter-ownership checks."""
import io

import pytest
import torch
from torch import nn

from curator.data import properties as p
from curator.layer._grad_output import GradientOutput
from curator.layer._multipole_state_energy import NeuralMultipoleStateEnergy
from curator.layer._multipole_state_proposal import MultipoleStateProposal
from curator.layer._multipole_variational_output import MultipoleVariationalOutput
from curator.model.features import IrrepsFeatureSpec
from curator.model.multipole import MultipolePotential
from curator.variational.multipole_boundary import boundary_for_geometry


class CoordinateInput(nn.Module):
    def forward(self, data):
        data = dict(data)
        data[p.positions] = data[p.positions].clone().requires_grad_(True)
        return data


class TinyGeometry(nn.Module):
    """Differentiable invariant/scalar and equivariant/vector feature supplier."""
    cutoff = 5.0

    def __init__(self):
        super().__init__()
        self.scales = nn.Parameter(torch.tensor([0.7, 0.3, 0.5, 0.2]))
        self.vector_scale = nn.Parameter(torch.tensor(0.6))
        self.readout = nn.Linear(4, 1)

    def response_feature_spec(self):
        return IrrepsFeatureSpec('test_features', '4x0e + 1x1o')

    def forward(self, data):
        data = dict(data)
        rows, energies, start = [], [], 0
        for count in data[p.n_atoms]:
            stop = start + int(count)
            xyz = data[p.positions][start:stop]
            relative = xyz - xyz.mean(0)
            r2 = relative.square().sum(-1)
            scalar = torch.stack((r2, torch.exp(-r2), data[p.Z][start:stop].to(xyz) / 10, torch.ones_like(r2)), dim=-1) * self.scales
            rows.append(torch.cat((scalar, relative * self.vector_scale), dim=-1))
            energies.append(self.readout(scalar).sum())
            start = stop
        data['test_features'] = torch.cat(rows)
        data[p.energy] = torch.stack(energies)
        return data


class ConvexReferenceEnergy(nn.Module):
    """Analytic oracle used only to exercise the explicit reference branch."""
    def __init__(self):
        super().__init__()
        self.hardness = nn.Parameter(torch.tensor(1.0))

    def bind(self, feature_spec):
        return self

    def forward(self, context, state):
        return 0.5 * self.hardness * state.square().sum()


def make_model(*, reference=False):
    torch.manual_seed(193)
    options = dict(hidden_irreps='4x0e + 2x1o + 2x2e', num_layers=2, num_heads=2, radial_features=3)
    energy = ConvexReferenceEnergy() if reference else NeuralMultipoleStateEnergy(**options, readout_scale=0.1)
    head = MultipoleVariationalOutput(energy_model=energy,
        proposal=MultipoleStateProposal(**options, readout_scale=0.3),
        state_mode='reference' if reference else 'predicted')
    return MultipolePotential(TinyGeometry(), input_modules=[CoordinateInput()],
        output_modules=[head, GradientOutput(grad_on_edge_diff=False, grad_on_positions=True)],
        model_outputs=[p.energy, p.forces]).double()


def sample(positions=None, charge=1.0, field=None):
    if positions is None:
        positions = torch.tensor([[0.1, 0.2, -0.1], [1.2, -0.3, 0.2], [-0.4, 0.8, 0.5]], dtype=torch.float64)
    return {
        p.positions: positions, p.Z: torch.tensor([1, 6, 8]),
        p.n_atoms: torch.tensor([3]), p.image_idx: torch.zeros(3, dtype=torch.long),
        p.pbc: torch.zeros(1, 3, dtype=torch.bool),
        p.total_charge: torch.tensor([charge], dtype=torch.float64),
        'external_field': torch.tensor([[0.2, -0.1, 0.15]] if field is None else [field], dtype=torch.float64),
    }


def test_full_proposal_path_forces_match_coordinate_finite_difference():
    model = make_model().eval()
    batch = sample()
    result = model(batch)
    direction = torch.tensor([[0.2, -0.1, 0.3], [-0.3, 0.2, 0.1], [0.1, 0.2, -0.2]], dtype=torch.float64)
    step = 2.e-5
    shifted = [model(sample(batch[p.positions] + sign * step * direction))[p.energy].sum() for sign in (-1, 1)]
    finite = (shifted[1] - shifted[0]) / (2 * step)
    torch.testing.assert_close(-(result[p.forces] * direction).sum(), finite, atol=2.e-8, rtol=2.e-5)
    # The exercised state actually depends on coordinates; a frozen-state test
    # would miss the proposal contribution to the predicted energy derivative.
    shift_states = [model(sample(batch[p.positions] + sign * step * direction))['multipole_state'] for sign in (-1, 1)]
    assert (shift_states[1] - shift_states[0]).abs().max() > 1.e-8


def test_energy_and_force_supervision_reaches_all_parameter_families():
    model = make_model().train()
    result = model(sample())
    loss = (result[p.energy] + 0.4).square().mean() + (result[p.forces] - 0.13).square().mean()
    loss.backward()
    for group in model.parameter_groups():
        gradients = [parameter.grad for parameter in group.params if parameter.grad is not None]
        assert gradients, group.name
        assert all(torch.isfinite(value).all() for value in gradients), group.name
        assert sum(float(value.abs().sum()) for value in gradients) > 0, group.name


def test_residual_backward_and_adamw_only_update_proposal():
    model = make_model().train()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    result = model(sample())
    loss = result['multipole_charge_residual'].square().mean() + result['multipole_dipole_residual'].square().mean()
    loss.backward()
    proposal_ids = {id(value) for value in model.output_modules[0].proposal.parameters()}
    assert any(value.grad is not None and bool(value.grad.abs().max() > 0) for value in model.output_modules[0].proposal.parameters())
    assert all(value.grad is None for value in model.parameters() if id(value) not in proposal_ids)
    optimizer.step()
    changed = [name for name, value in model.named_parameters() if not torch.equal(value.detach(), before[name])]
    assert changed
    assert all(name.startswith('output_modules.0.proposal.') for name in changed)


def test_parameter_groups_cover_unique_parameters_and_named_state_modules():
    model = make_model()
    groups = model.parameter_groups()
    ids = [id(value) for group in groups for value in group.params]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    actual = {group.name: {id(value) for value in group.params} for group in groups}
    head = model.output_modules[0]
    assert actual['state_energy'] == {id(value) for value in head.energy_model.parameters()}
    assert actual['proposal'] == {id(value) for value in head.proposal.parameters()}


def test_no_grad_evaluation_preserves_forces_and_rejects_inference_mode():
    model = make_model().eval()
    expected = model(sample())
    with torch.no_grad():
        actual = model(sample())
    for name in (p.energy, p.forces, 'multipole_state', 'multipole_charge_rms', 'multipole_dipole_rms'):
        torch.testing.assert_close(actual[name], expected[name])
    with torch.inference_mode(), pytest.raises(RuntimeError, match='inference_mode'):
        model(sample())


def test_checkpoint_state_dict_roundtrip():
    model = make_model().eval()
    payload = io.BytesIO()
    torch.save({'state_dict': model.state_dict()}, payload)
    payload.seek(0)
    restored = make_model().eval()
    restored.load_state_dict(torch.load(payload, weights_only=True)['state_dict'], strict=True)
    expected, actual = model(sample()), restored(sample())
    for name in (p.energy, p.forces, 'multipole_state', 'multipole_charge_rms', 'multipole_dipole_rms'):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_reference_branch_returns_constrained_minimum_and_envelope_force():
    model = make_model(reference=True).eval()
    batch = sample()
    result = model(batch)
    xyz, field = batch[p.positions], batch['external_field'][0]
    expected_q = 1 / 3 + (xyz - xyz.mean(0)) @ field
    expected_p = field.expand(3, -1)
    torch.testing.assert_close(result['multipole_state'][:, 0], expected_q, atol=1.e-7, rtol=1.e-7)
    torch.testing.assert_close(result['multipole_state'][:, 1:], expected_p, atol=1.e-7, rtol=1.e-7)
    assert result['multipole_charge_rms'].max() < 1.e-6
    assert result['multipole_dipole_rms'].max() < 1.e-6
    assert result['multipole_feasibility'].max() < 1.e-12
    baseline_data = model.representation(model.input_modules[0](batch))
    baseline_force = -torch.autograd.grad(baseline_data[p.energy].sum(), baseline_data[p.positions])[0]
    torch.testing.assert_close(result[p.forces], baseline_force + expected_q[:, None] * field, atol=1.e-7, rtol=1.e-7)


def test_batch_separation_preserves_individual_predictions():
    model = make_model().eval()
    first = sample(charge=1.)
    second = sample(first[p.positions] * 1.2 + 4, charge=-1., field=[-0.1, 0.05, 0.2])
    batch = {
        p.positions: torch.cat((first[p.positions], second[p.positions])),
        p.Z: torch.cat((first[p.Z], second[p.Z])), p.n_atoms: torch.tensor([3, 3]),
        p.image_idx: torch.tensor([0, 0, 0, 1, 1, 1]), p.pbc: torch.zeros(2, 3, dtype=torch.bool),
        p.total_charge: torch.tensor([1., -1.], dtype=torch.float64),
        'external_field': torch.cat((first['external_field'], second['external_field'])),
    }
    individual = [model(first), model(second)]
    actual = model(batch)
    for name in (p.energy, p.forces, 'multipole_state', 'multipole_charge_rms', 'multipole_dipole_rms'):
        torch.testing.assert_close(actual[name], torch.cat([item[name] for item in individual]), atol=2.e-10, rtol=2.e-9)
