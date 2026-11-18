"""Physical and serialization contracts for the shared EFA charge proposal."""

import copy
import io
import math

import pytest
import torch

from curator.layer.efa import (
    EFAChargeProposal, SharedGeometryChargeProposal, SharedTwoScaleQConditionedEFAEnergy,
    TwoScaleQConditionedEFAEnergy,
)
from curator.model.features import IrrepsFeatureSpec


def _case(*, fixed_charge=True):
    torch.manual_seed(73)
    spec = IrrepsFeatureSpec("features", "4x0e + 3x1o + 2x2e")
    energy = TwoScaleQConditionedEFAEnergy(
        hidden_features=4, charge_features=4, num_directions=50,
        max_frequency=math.pi, driving_scale=0.4, curvature_scale=0.3,
    ).double().bind(spec)
    shared = SharedTwoScaleQConditionedEFAEnergy.from_two_scale(energy)
    proposal = SharedGeometryChargeProposal(
        hidden_irreps="4x0e + 2x1o + 1x2e", readout_features=8,
    ).double().bind(spec)
    features = torch.randn(4, spec.irreps.dim, dtype=torch.float64, requires_grad=True)
    positions = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    field = torch.randn(4, 1, dtype=torch.float64, requires_grad=True)
    context = {
        "features": features, "positions": positions,
        "pbc": torch.zeros(3, dtype=torch.bool),
        "constraint_matrix": features.new_ones((int(fixed_charge), 4)),
        "constraint_target": features.new_ones((int(fixed_charge),)),
        "network_field": field - field.mean() if fixed_charge else field,
    }
    return energy, shared, proposal, context


@pytest.mark.parametrize("cached", [False, True])
def test_shared_energy_preserves_fixed_q_energy_force_and_parameter_gradients(cached):
    energy, shared, proposal, context = _case()
    charge = torch.randn(4, 1, dtype=torch.float64, requires_grad=True)
    reference = energy(context, charge)
    prepared = shared.prepare_proposal_context(context, proposal) if cached else context
    actual = shared(prepared, charge)
    torch.testing.assert_close(actual, reference, atol=2e-12, rtol=2e-12)
    inputs = (context["positions"], context["features"], charge)
    expected = torch.autograd.grad(reference, (*inputs, *energy.parameters()), retain_graph=True)
    observed = torch.autograd.grad(actual, (*inputs, *shared.parameters()), retain_graph=True)
    for left, right in zip(expected, observed):
        torch.testing.assert_close(left, right, atol=2e-11, rtol=2e-10)
    proposal_gradients = torch.autograd.grad(
        actual, tuple(proposal.parameters()), allow_unused=True, retain_graph=True)
    assert all(value is None or not bool(value.count_nonzero()) for value in proposal_gradients)


def test_shared_energy_remains_strongly_convex_and_boundary_independent_at_fixed_q():
    energy, shared, proposal, context = _case()
    charge = torch.randn(4, dtype=torch.float64, requires_grad=True)
    prepared = shared.prepare_proposal_context(context, proposal)
    hessian = torch.autograd.functional.hessian(lambda q: shared(prepared, q[:, None]), charge)
    torch.testing.assert_close(hessian, hessian.T, atol=2e-12, rtol=2e-12)
    assert torch.linalg.eigvalsh(hessian).min() >= shared.hardness - 2e-12
    changed = {**context, "network_field": context["network_field"] * 3 + 2}
    changed = shared.prepare_proposal_context(changed, proposal)
    torch.testing.assert_close(shared(prepared, charge[:, None]), shared(changed, charge[:, None]),
                               atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(shared(prepared, charge[:, None]), energy(context, charge[:, None]),
                               atol=2e-12, rtol=2e-12)


def test_shared_cache_preserves_total_force_residual_and_efr_parameter_gradients():
    _, shared, proposal, context = _case()
    prepared = shared.prepare_proposal_context(context, proposal)
    charge = proposal(prepared)
    results = []
    for energy_context in (context, prepared):
        energy = shared(energy_context, charge)
        force = -torch.autograd.grad(energy, context["positions"], create_graph=True, retain_graph=True)[0]
        gradient = torch.autograd.grad(energy, charge, create_graph=True, retain_graph=True)[0]
        residual = gradient - gradient.mean()
        parameters = (*shared.parameters(), *proposal.parameters())
        loss = energy.square() + force.square().sum() + residual.square().sum()
        derivatives = torch.autograd.grad(loss, parameters, retain_graph=True)
        results.append((energy, force, residual, *derivatives))
    for reference, actual in zip(*results):
        torch.testing.assert_close(actual, reference, atol=3e-11, rtol=3e-9)


def test_shared_proposal_fukui_and_residual_gradients_reach_value_and_decoder():
    _, shared, proposal, context = _case()
    prepared = shared.prepare_proposal_context(context, proposal)
    charge = proposal(prepared)
    torch.testing.assert_close(charge.sum(), context["constraint_target"].sum(), atol=1e-12, rtol=0)
    _, weights = proposal.predict_components(prepared)
    assert bool((weights > 0).all())
    gradient = torch.autograd.grad(shared(prepared, charge), charge, create_graph=True)[0]
    loss = (gradient - gradient.mean()).square().mean()
    parameters = dict(proposal.named_parameters())
    gradients = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True)
    named_gradients = dict(zip(parameters, gradients))
    for prefix in ("input.", "potential_input.", "fusion.", "readout."):
        assert any(value is not None and bool(value.abs().sum() > 0)
                   for name, value in named_gradients.items() if name.startswith(prefix)), prefix


def test_remote_potential_changes_raw_local_charge_without_fukui_projection():
    _, shared, proposal, context = _case(fixed_charge=False)
    prepared = shared.prepare_proposal_context(context, proposal)
    raw, _ = proposal.predict_components(prepared)
    derivative = torch.autograd.grad(raw[0, 0], context["network_field"])[0]
    assert derivative[3, 0].abs() > 1e-12
    # No charge projection can account for this distant dependency.
    torch.testing.assert_close(proposal(prepared), raw, atol=0, rtol=0)


def test_shared_checkpoint_round_trip_and_no_persistent_forward_cache():
    _, shared, proposal, context = _case()
    prepared = shared.prepare_proposal_context(context, proposal)
    expected_charge = proposal(prepared)
    expected_energy = shared(prepared, expected_charge)
    buffer = io.BytesIO()
    torch.save((shared, proposal), buffer)
    buffer.seek(0)
    loaded_energy, loaded_proposal = torch.load(buffer, weights_only=False)
    loaded_context = loaded_energy.prepare_proposal_context(context, loaded_proposal)
    torch.testing.assert_close(loaded_proposal(loaded_context), expected_charge, atol=0, rtol=0)
    torch.testing.assert_close(loaded_energy(loaded_context, expected_charge), expected_energy, atol=0, rtol=0)
    assert not any("cache" in key or "_shared_efa" in key for key in shared.state_dict())
    mismatched = copy.deepcopy(proposal.state_dict())
    mismatched["_extra_state"]["schema"] = "not_shared"
    with pytest.raises(ValueError, match="contract"):
        proposal.load_state_dict(mismatched)


def test_converter_keeps_energy_and_matching_local_proposal_weights_without_aliasing():
    from curator.layer import VariationalEnergyOutput
    from curator.model import NeuralNetworkPotential, Painn
    from curator.model.conversion import apply_model_transforms

    model = NeuralNetworkPotential(Painn(2, 4, 5.0, num_basis=4), output_modules=[
        VariationalEnergyOutput(
            internal_energy=TwoScaleQConditionedEFAEnergy(
                hidden_features=4, num_directions=50, max_frequency=math.pi),
            proposal=EFAChargeProposal(hidden_irreps="4x0e + 2x1o", hidden_features=4,
                                       num_directions=50, max_frequency=math.pi, readout_features=8),
            state_unit="e", conjugate_unit="V",
        ),
    ]).double()
    upgraded = apply_model_transforms(model, [{"type": "efa_shared_charge_proposal"}])
    before, after = model.output_modules[0], upgraded.output_modules[0]
    assert after.residual_evaluation == "reuse"
    for name, parameter in before.internal_energy.named_parameters():
        copied = dict(after.internal_energy.named_parameters())[name]
        torch.testing.assert_close(parameter, copied, atol=0, rtol=0)
        assert parameter is not copied
    for name in ("input", "boundary_input", "potential_input", "norm", "readout"):
        original = getattr(before.proposal, name)
        copied = getattr(after.proposal, name)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, copied.state_dict()[key], atol=0, rtol=0)
        assert original is not copied
    for key, value in before.proposal.fusion[-1].state_dict().items():
        torch.testing.assert_close(value, after.proposal.fusion.state_dict()[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="matching"):
        apply_model_transforms(model, [{"type": "efa_shared_charge_proposal",
                                       "proposal_options": {"charge_scale": 2.0}}])
    efficient = apply_model_transforms(model, [{"type": "efa_reuse"}])
    assert efficient.output_modules[0].residual_evaluation == "reuse"
    assert model.output_modules[0].residual_evaluation == "replay"
    from curator.layer.efa import EuclideanFastAttention
    assert all(module.contraction == "auto" for module in efficient.modules()
               if isinstance(module, EuclideanFastAttention))
    assert list(model.state_dict()) == list(efficient.state_dict())
    efficient.load_state_dict(model.state_dict(), strict=True)
    for name, value in model.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, efficient.state_dict()[name], atol=0, rtol=0)
        else:
            assert value == efficient.state_dict()[name]
    assert [group.name for group in model.parameter_groups()] == [
        group.name for group in efficient.parameter_groups()]
