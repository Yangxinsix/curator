"""End-to-end checks for boundary-conditioned ACVR proposal communication."""

import copy
from dataclasses import asdict

import pytest
import torch

from curator.data import properties
from curator.data.variational_data import canonical_boundary_hash
from curator.layer import PairwiseDistance, VariationalEnergyOutput
from curator.layer._charge_internal_energy import EwaldChargeEnergy
from curator.layer._variational_proposal import BoundaryConditionedProposal
from curator.model import NeuralNetworkPotential
from curator.model.base import Representation
from curator.model.features import IrrepsFeatureSpec
from curator.variational.spec import BoundarySpec, ControlSpec, GroupSpec, VariableSpec

from test_acvr_joint_model import _batch
from test_acvr_joint_lifecycle import _assert_same_state, _run
from test_acvr_response_features import _representation


def _model(communication, *, backbone="painn", kind="scalar", proposal_kwargs=None, **kwargs):
    torch.manual_seed(43)
    representation = _representation(backbone) if isinstance(backbone, str) else backbone
    return NeuralNetworkPotential(
        representation,
        input_modules=[PairwiseDistance(compute_distance_from_R=True)],
        output_modules=[VariationalEnergyOutput(
            state_kind=kind, components=3 if kind == "vector" else 1,
            hidden_features=8,
            proposal=BoundaryConditionedProposal(communication=communication, hidden_features=8,
                                                 num_layers=2, num_heads=2, **(proposal_kwargs or {})),
            **kwargs,
        )],
    ).double()


def _prediction(model, batch):
    return model(batch)["variational_prediction"]


@pytest.mark.parametrize("communication", ["pointwise", "mp", "attention"])
@pytest.mark.parametrize("backbone", ["painn", "mace", "nequip"])
@pytest.mark.parametrize("kind", ["scalar", "vector"])
def test_three_backbones_and_state_types_are_equivariant_and_trainable(communication, backbone, kind):
    model = _model(communication, backbone=backbone, kind=kind)
    positions = _batch()[properties.positions]
    rotation = positions.new_tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., -1.]])
    field = positions.new_tensor([.2, -.1, .3] if kind == "vector" else [.2])
    first = _prediction(model, _batch(positions.clone(), field=field))
    rotated_field = rotation @ field if kind == "vector" else field
    second = _prediction(model, _batch(positions @ rotation.T + 1.7, field=rotated_field))
    torch.testing.assert_close(first["labels"]["condition_energy"]["values"],
                               second["labels"]["condition_energy"]["values"], rtol=3e-6, atol=3e-7)
    for key in ("variable_value", "condition_forces"):
        original = first["labels"][key]["values"]
        expected = original.reshape(-1, 3) @ rotation.T if kind == "vector" or key == "condition_forces" else original
        torch.testing.assert_close(second["labels"][key]["values"], expected.reshape(-1),
                                   rtol=3e-6, atol=3e-7)
    residual = first["residuals"]["stationarity"]["values"]
    expected = residual.reshape(-1, 3) @ rotation.T if kind == "vector" else residual
    torch.testing.assert_close(second["residuals"]["stationarity"]["values"], expected.reshape(-1),
                               rtol=3e-6, atol=3e-7)
    loss = sum(first["labels"][key]["values"].square().mean()
               for key in ("condition_energy", "condition_forces")) + .1 * residual.square().mean()
    loss.backward()
    for group in model.parameter_groups():
        gradients = [parameter.grad for parameter in group.params if parameter.grad is not None]
        assert gradients, group.name
        assert all(torch.isfinite(gradient).all() for gradient in gradients), group.name
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0, group.name


@pytest.mark.parametrize("communication", ["pointwise", "mp", "attention"])
def test_residual_optimizer_updates_only_complete_proposal(communication):
    model = _model(communication, kind="vector")
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
    prediction = _prediction(model, _batch(field=[.2, -.1, .3]))
    prediction["residuals"]["stationarity"]["values"].square().mean().backward()
    for name, parameter in model.named_parameters():
        if not name.startswith("output_modules.0.proposal."):
            assert parameter.grad is None, name
    proposal_gradients = [parameter.grad for parameter in model.output_modules[0].proposal.parameters()
                          if parameter.grad is not None]
    assert proposal_gradients and all(torch.isfinite(value).all() for value in proposal_gradients)
    proposal = model.output_modules[0].proposal
    learned_parts = [proposal.input, proposal.boundary_input, proposal.readout]
    for block in proposal.blocks:
        learned_parts.extend((block.update, block.boundary_embed, block.state_readout))
        if communication != "pointwise":
            learned_parts.append(block.communication)
    for module in learned_parts:
        assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                   for parameter in module.parameters()), type(module).__name__
    optimizer.step()
    changed = [name for name, parameter in model.named_parameters() if not torch.equal(parameter, before[name])]
    assert changed and all(name.startswith("output_modules.0.proposal.") for name in changed)


@pytest.mark.parametrize("communication", ["pointwise", "mp", "attention"])
@pytest.mark.parametrize("backbone", ["mace", "nequip"])
def test_high_order_backbone_features_influence_proposed_state(communication, backbone):
    model = _model(communication, backbone=backbone, kind="vector", compute_forces=False)
    spec = model.representation.response_feature_spec()
    captured = []
    handle = model.representation.register_forward_hook(
        lambda _module, _inputs, output: captured.append(spec.flatten(output)))
    try:
        state = _prediction(model, _batch(field=[.2, -.1, .3]))["labels"]["variable_value"]["values"]
    finally:
        handle.remove()
    gradient = torch.autograd.grad(state.square().sum(), captured[0])[0]
    higher_parts = [gradient[:, part] for (_, irrep), part in zip(spec.irreps, spec.irreps.slices()) if irrep.l >= 2]
    assert higher_parts
    assert sum(float(part.abs().sum()) for part in higher_parts) > 1e-12


def _chain_batch(field):
    return {
        properties.Z: torch.tensor([1, 6, 8, 1]),
        properties.positions: field.new_tensor([[0., 0., 0.], [1., .1, 0.], [2., 0., .1], [3., .1, .1]]),
        properties.edge_idx: torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]),
        properties.n_atoms: torch.tensor([4]), properties.image_idx: torch.zeros(4, dtype=torch.long),
        "local_external_field": field.reshape(4, 1),
    }


@pytest.mark.parametrize("communication", ["pointwise", "mp", "attention"])
def test_field_jacobian_matches_communication_reach(communication):
    model = _model(communication, compute_forces=False)
    field = torch.tensor([.1, -.2, .3, -.1], dtype=torch.float64, requires_grad=True)
    state = _prediction(model, _chain_batch(field))["labels"]["variable_value"]["values"]
    row0 = torch.autograd.grad(state[0], field, retain_graph=True)[0]
    row2 = torch.autograd.grad(state[2], field)[0]
    assert abs(float(row0[0])) > 1e-12
    if communication == "pointwise":
        torch.testing.assert_close(row0[1:], torch.zeros_like(row0[1:]), rtol=0, atol=0)
        assert row2[3].item() == 0
    elif communication == "mp":
        assert abs(float(row2[3])) > 1e-12
        # Two proposal blocks cannot move a new field over a three-edge path.
        assert row0[3].item() == 0
    else:
        assert abs(float(row0[3])) > 1e-12


def test_attention_does_not_mix_separate_geometries():
    model = _model("attention", compute_forces=False)
    one = _batch()
    batch = {
        properties.Z: one[properties.Z].repeat(2),
        properties.positions: torch.cat((one[properties.positions], one[properties.positions] + 7.)),
        properties.edge_idx: torch.cat((one[properties.edge_idx], one[properties.edge_idx] + 3)),
        properties.n_atoms: torch.tensor([3, 3]),
        properties.image_idx: torch.tensor([0, 0, 0, 1, 1, 1]),
        "local_external_field": torch.tensor([[.1], [.2], [-.1], [.3], [-.4], [.2]],
                                              dtype=torch.float64, requires_grad=True),
    }
    state = _prediction(model, batch)["labels"]["variable_value"]["values"]
    gradient = torch.autograd.grad(state[:3].square().sum(), batch["local_external_field"])[0]
    assert bool(gradient[:3].abs().sum() > 0)
    torch.testing.assert_close(gradient[3:], torch.zeros_like(gradient[3:]), rtol=0, atol=0)


def test_attention_does_not_mix_conditions_on_the_same_geometry():
    model = _model("attention", compute_forces=False)
    groups = tuple(GroupSpec.from_indices(f"atom{i}", [i]) for i in range(3))
    variable = VariableSpec("internal", "scalar", 1, "dimensionless", "eV")

    def batch(second_field):
        result = _batch()
        boundaries = tuple(asdict(BoundarySpec(variable, groups=groups, controls=tuple(
            ControlSpec.group_potential(f"field{i}", f"atom{i}", value, unit="eV")
            for i, value in enumerate(field)))) for field in ((.1, -.2, .3), second_field))
        result["variational"] = {
            "condition_geometry_index": torch.tensor([0, 0]),
            "condition_state_ptr": torch.tensor([0, 3, 6]),
            "boundary_spec": boundaries,
            "boundary_hash": tuple(canonical_boundary_hash(boundary) for boundary in boundaries),
        }
        return result

    first = _prediction(model, batch((.3, -.1, .2)))["labels"]["variable_value"]["values"]
    second = _prediction(model, batch((-.4, .2, .7)))["labels"]["variable_value"]["values"]
    torch.testing.assert_close(first[:3], second[:3], rtol=0, atol=0)
    assert not torch.allclose(first[3:], second[3:])


def test_periodic_attention_is_invariant_to_atom_image_choice():
    model = _model("attention", kind="vector")
    batch = _batch(field=[.2, -.1, .3])
    cell = batch[properties.positions].new_tensor([[4., 0., 0.], [.3, 5., 0.], [.1, .2, 6.]])
    batch[properties.cell] = cell
    batch[properties.pbc] = torch.ones(1, 3, dtype=torch.bool)
    shifted = copy.deepcopy(batch)
    displacement = torch.zeros_like(batch[properties.positions])
    displacement[1] = cell[1]
    shifted[properties.positions] = shifted[properties.positions] + displacement
    edges = shifted[properties.edge_idx]
    shifted[properties.cell_displacements] = displacement[edges[:, 0]] - displacement[edges[:, 1]]
    original = _prediction(model, batch)
    other_image = _prediction(model, shifted)
    for key in ("condition_energy", "variable_value", "condition_forces"):
        torch.testing.assert_close(original["labels"][key]["values"], other_image["labels"][key]["values"],
                                   rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("communication", ["pointwise", "mp", "attention"])
def test_coordinate_and_field_second_derivatives_match_finite_differences(communication):
    model = _model(communication, kind="vector", retain_response_graph=True)
    positions = _batch()[properties.positions].clone().requires_grad_(True)
    field = positions.new_tensor([.2, -.1, .3], requires_grad=True)
    direction = positions.new_tensor([[.1, -.2, .3], [-.3, .2, .1], [.2, .1, -.2]])
    field_direction = positions.new_tensor([.3, -.2, .1])
    prediction = _prediction(model, _batch(positions, field=field))
    energy = prediction["labels"]["condition_energy"]["values"].sum()
    forces = prediction["labels"]["condition_forces"]["values"].reshape(3, 3)
    field_gradient = torch.autograd.grad(energy, field, retain_graph=True, create_graph=True)[0]
    field_curvature = torch.autograd.grad((field_gradient * field_direction).sum(), field, retain_graph=True)[0]
    force_directional = torch.autograd.grad((forces * direction).sum(), positions)[0]
    eps = 1e-5
    shifted = [_prediction(model, _batch(positions.detach() + sign * eps * direction,
                                        field=field.detach())) for sign in (-1, 1)]
    energy_fd = (shifted[1]["labels"]["condition_energy"]["values"].sum()
                 - shifted[0]["labels"]["condition_energy"]["values"].sum()) / (2 * eps)
    force_fd = (shifted[1]["labels"]["condition_forces"]["values"]
                - shifted[0]["labels"]["condition_forces"]["values"]) / (2 * eps)
    torch.testing.assert_close(energy_fd, -(forces * direction).sum(), rtol=3e-5, atol=2e-8)
    torch.testing.assert_close(force_fd.reshape(3, 3), force_directional, rtol=1e-4, atol=2e-8)
    shifted_gradients = []
    for sign in (-1, 1):
        shifted_field = (field.detach() + sign * eps * field_direction).requires_grad_(True)
        shifted_prediction = _prediction(model, _batch(positions.detach().clone(), field=shifted_field))
        shifted_energy = shifted_prediction["labels"]["condition_energy"]["values"].sum()
        shifted_gradients.append(torch.autograd.grad(shifted_energy, shifted_field)[0])
    torch.testing.assert_close((shifted_gradients[1] - shifted_gradients[0]) / (2 * eps),
                               field_curvature, rtol=1e-4, atol=2e-8)


class _ScalarRepresentation(Representation):
    cutoff = 5.0

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(119, 4)
        self.readout = torch.nn.Linear(4, 1)

    def response_feature_spec(self):
        return IrrepsFeatureSpec("scalar_features", "4x0e")

    def forward(self, data):
        features = self.embedding(data[properties.Z])
        data["scalar_features"] = features
        data[properties.energy] = self.readout(features).reshape(-1).sum().reshape(1)
        return data


@pytest.mark.parametrize("communication", ["pointwise", "mp", "attention"])
def test_scalar_only_backbone_supports_vector_state(communication):
    model = _model(communication, backbone=_ScalarRepresentation(), kind="vector")
    prediction = _prediction(model, _batch(field=[.2, -.1, .3]))
    state = prediction["labels"]["variable_value"]["values"]
    assert state.shape == (9,) and torch.isfinite(state).all() and bool(state.abs().sum() > 0)
    prediction["labels"]["condition_energy"]["values"].square().sum().backward()
    assert model.representation.embedding.weight.grad is not None


@pytest.mark.parametrize("communication", ["pointwise", "mp", "attention"])
def test_state_dict_restores_identical_proposal_and_derivatives(communication):
    model = _model(communication, kind="vector")
    first = _prediction(model, _batch(field=[.2, -.1, .3]))
    state = copy.deepcopy(model.state_dict())
    restored = _model(communication, kind="vector")
    restored.load_state_dict(state)
    second = _prediction(restored, _batch(field=[.2, -.1, .3]))
    for key in ("condition_energy", "condition_forces", "variable_value"):
        torch.testing.assert_close(first["labels"][key]["values"], second["labels"][key]["values"], rtol=0, atol=0)
    torch.testing.assert_close(first["residuals"]["stationarity"]["values"],
                               second["residuals"]["stationarity"]["values"], rtol=0, atol=0)


def test_four_arms_start_from_identical_physics_and_shared_proposal_parameters():
    physics, shared = [], []
    for communication in ("pointwise", "mp", "attention", "ewald"):
        model = _model(
            communication, state_unit="e", conjugate_unit="V",
            internal_energy=EwaldChargeEnergy(hidden_features=8),
            proposal_kwargs={"state_scale": .05, "field_scale": 1.},
        )
        physics.append({name: parameter.detach().clone() for name, parameter in model.named_parameters()
                        if not name.startswith("output_modules.0.proposal.")})
        shared.append({name: parameter.detach().clone()
                       for name, parameter in model.output_modules[0].proposal.named_parameters()
                       if ".communication." not in name})
    assert any(name.startswith("output_modules.0.internal_energy.") for name in physics[0])
    assert any(name.startswith("blocks.1.") for name in shared[0])
    for index in range(1, 4):
        _assert_same_state(physics[index], physics[0])
        _assert_same_state(shared[index], shared[0])


@pytest.mark.parametrize("communication", ["attention", "ewald"])
@pytest.mark.parametrize("changed_scale", ["state_scale", "field_scale"])
def test_fixed_proposal_scales_are_part_of_checkpoint_contract(communication, changed_scale):
    options = dict(state_unit="e", conjugate_unit="V",
                   proposal_kwargs={"state_scale": .05, "field_scale": 1.})
    original = _model(communication, internal_energy=EwaldChargeEnergy(hidden_features=8), **options)
    saved = copy.deepcopy(original.state_dict())
    contract = original.output_modules[0].get_extra_state()["contract"]["proposal"]
    assert contract["state_scale"] == .05 and contract["field_scale"] == 1.
    restored = _model(communication, internal_energy=EwaldChargeEnergy(hidden_features=8), **options)
    restored.load_state_dict(saved)
    _assert_same_state(restored.state_dict(), original.state_dict())
    changed = copy.deepcopy(options)
    changed["proposal_kwargs"][changed_scale] *= 2
    incompatible = _model(communication, internal_energy=EwaldChargeEnergy(hidden_features=8), **changed)
    with pytest.raises(ValueError, match="contract differs"):
        incompatible.load_state_dict(saved)


@pytest.mark.parametrize("communication", ["pointwise", "mp", "attention"])
def test_cli_two_steps_checkpoint_rebuild_and_resume(communication, tmp_path):
    from curator.utils import load_model

    proposal = (
        "+model.output_modules.0.proposal={"
        "_target_:curator.layer._variational_proposal.BoundaryConditionedProposal,"
        f"communication:{communication},hidden_features:4,num_layers:1,num_heads:2}}"
    )
    trained = _run(tmp_path, "trained", [proposal, "trainer.max_epochs=1"])
    assert trained["global_step"] == 2
    state_only = dict(trained)
    state_only.pop("model", None)
    state_only["model_save_mode"] = "state_dict"
    state_path = tmp_path / "state_only.ckpt"
    torch.save(state_only, state_path)
    rebuilt = load_model(state_path, device="cpu")
    _assert_same_state(rebuilt.state_dict(), trained["model"].state_dict())
    checkpoint = tmp_path / "trained/model_path/last.ckpt"
    resumed = _run(tmp_path, "resumed", [proposal, "trainer.max_epochs=2",
                    f"model_path={{path: {checkpoint}, mode: resume}}"])
    continuous = _run(tmp_path, "continuous", [proposal, "trainer.max_epochs=2"])
    assert resumed["global_step"] == continuous["global_step"] == 4
    _assert_same_state(resumed["state_dict"], continuous["state_dict"])
    assert len(resumed["optimizer_states"]) == len(continuous["optimizer_states"]) == 1
    actual, expected = resumed["optimizer_states"][0], continuous["optimizer_states"][0]
    assert actual["param_groups"] == expected["param_groups"]
    for index, value in expected["state"].items():
        _assert_same_state(actual["state"][index], value)
