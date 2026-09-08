"""Physical invariants and derivative tests for isolated SPICE charge models."""

import copy
import math
from types import SimpleNamespace

from ase.units import Bohr, Hartree
import pytest
import torch
from torch import nn

from curator.model import NeuralNetworkPotential
from curator.model.spice_charge import SpiceChargeModel
from curator.train import AmortizedVariationalLoss, ModelOutput


def _batch():
    return {
        "positions": torch.tensor([
            [-.4, .2, .1], [1.6, .5, -.2], [1., 1., 1.], [3.8, 2., 1.5],
            [2., -.5, .2], [9.5, .3, .8],
        ], dtype=torch.float64),
        "n_atoms": torch.tensor([2, 2, 2]),
        "atomic_numbers": torch.tensor([3, 9, 11, 17, 19, 35]),
        "total_charge": torch.tensor([0., 1., -2.], dtype=torch.float64),
    }


def _metadata():
    return {
        "schema": "spice_acvr_ion_pairs_v1", "source_sha256": "0" * 64,
        "units": {"positions": "Angstrom", "energy": "eV", "forces": "eV/Angstrom",
                  "dipole": "e*Angstrom", "total_charge": "e", "atomic_charge": "e"},
        "gauge": {"boundary": "fixed_total_charge", "dipole_origin": "source_cartesian_origin",
                  "coordinate_transform": "none"},
        "splits": {"train": [0, 1], "val": [2], "test": [3]},
    }


def test_fixed_charge_conservation_and_ordinary_curator_outputs():
    batch = _batch()
    model = NeuralNetworkPotential(SpiceChargeModel())
    predicted = model(batch)
    assert predicted["energy"].shape == (3,)
    assert predicted["forces"].shape == (6, 3)
    assert predicted["dipole"].shape == (3, 3)
    assert "amortized_training" not in predicted
    assert all(value.dtype == torch.float64 for value in predicted.values())
    torch.testing.assert_close(predicted["atomic_charge"].reshape(-1, 2).sum(1), batch["total_charge"], atol=1e-13, rtol=0)
    target = dict(batch, energy=torch.zeros(3, dtype=torch.float64),
                  forces=torch.ones((6, 3), dtype=torch.float64), dipole=torch.ones((3, 3), dtype=torch.float64))
    outputs = [ModelOutput(name, loss_fn=nn.MSELoss()) for name in ("energy", "forces", "dipole")]
    loss = sum(output.calculate_loss(predicted, target)[0] for output in outputs)
    loss.backward()
    assert all(parameter.grad is None for parameter in model.representation.proposal.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.representation.physical.parameters())
    assert sum(float(parameter.grad.square().sum()) for parameter in model.representation.physical.parameters()) > 0


def test_gaussian_self_term_pair_energy_and_positive_curvature():
    model = SpiceChargeModel(gaussian_sigma=.8)
    distance = torch.tensor([1e-6, .3, 1., 5., 30.], dtype=torch.float64)
    kernel = model.gaussian_kernel(distance)
    self_term = Hartree * Bohr / (math.sqrt(math.pi) * .8)
    torch.testing.assert_close(kernel[:, 0, 0], torch.full_like(distance, self_term))
    torch.testing.assert_close(kernel[0, 0, 1], kernel[0, 0, 0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(kernel[-1, 0, 1] * distance[-1], distance.new_tensor(Hartree * Bohr))
    assert float(torch.linalg.eigvalsh(kernel).min()) > -1e-12
    charge = torch.tensor([.3, -.6], dtype=torch.float64)
    energy = .5 * charge @ kernel[2] @ charge
    expected = .5 * self_term * charge.square().sum() + kernel[2, 0, 1] * charge.prod()
    torch.testing.assert_close(energy, expected)
    for problem in model.problems(_batch()):
        assert float(torch.linalg.eigvalsh(problem.hessian).min().detach()) >= .1


@pytest.mark.parametrize("variant", ["acvr", "short_range"])
def test_translation_rotation_and_atom_permutation_consistency(variant):
    batch = _batch()
    model = SpiceChargeModel(variant=variant).eval()
    initial = model(batch)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
    shift = torch.tensor([.4, -1.2, .7], dtype=torch.float64)
    transformed = dict(batch, positions=batch["positions"] @ rotation.T + shift)
    changed = model(transformed)
    torch.testing.assert_close(changed["energy"], initial["energy"], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(changed["atomic_charge"], initial["atomic_charge"], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(changed["forces"], initial["forces"] @ rotation.T, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(changed["dipole"], initial["dipole"] @ rotation.T + batch["total_charge"][:, None] * shift,
                               atol=1e-12, rtol=1e-12)
    swapped = dict(batch, positions=batch["positions"].reshape(-1, 2, 3).flip(1).reshape(-1, 3),
                   atomic_numbers=batch["atomic_numbers"].reshape(-1, 2).flip(1).reshape(-1))
    permuted = model(swapped)
    torch.testing.assert_close(permuted["energy"], initial["energy"], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(permuted["dipole"], initial["dipole"], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(permuted["atomic_charge"].reshape(-1, 2).flip(1).reshape(-1), initial["atomic_charge"],
                               atol=1e-12, rtol=1e-12)
    if variant == "acvr":
        torch.testing.assert_close(model.propose(swapped).reshape(-1, 2).flip(1).reshape(-1), model.propose(batch))


@pytest.mark.parametrize("variant", ["acvr", "short_range"])
def test_forces_are_full_energy_derivatives_and_force_loss_has_correct_mixed_gradient(variant):
    batch = _batch()
    model = SpiceChargeModel(variant=variant)
    initial = model(batch)
    epsilon = 1e-5
    energies = []
    for sign in (-1, 1):
        displaced = batch["positions"].clone()
        displaced[1, 0] += sign * epsilon
        energies.append(model(dict(batch, positions=displaced))["energy"].sum().detach())
    torch.testing.assert_close(initial["forces"][1, 0], -(energies[1] - energies[0]) / (2 * epsilon), rtol=2e-6, atol=2e-8)
    target = torch.full_like(initial["forces"], .1)
    force_loss = (initial["forces"] - target).square().mean()
    parameter = model.physical["local"][-1].weight
    derivative = torch.autograd.grad(force_loss, parameter)[0][2, 0]
    original = parameter[2, 0].detach().clone()
    losses = []
    for sign in (-1, 1):
        with torch.no_grad():
            parameter[2, 0] = original + sign * epsilon
        losses.append((model(batch)["forces"] - target).square().mean().detach())
    with torch.no_grad():
        parameter[2, 0] = original
    torch.testing.assert_close(derivative, (losses[1] - losses[0]) / (2 * epsilon), rtol=2e-5, atol=2e-9)


def test_short_range_baseline_has_no_distance_tail_but_acvr_has_coulomb_force():
    batch = {"positions": torch.tensor([[0., 0., 0.], [7., 0., 0.]], dtype=torch.float64),
             "n_atoms": torch.tensor([2]), "atomic_numbers": torch.tensor([11, 17]),
             "total_charge": torch.tensor([2.], dtype=torch.float64)}
    baseline = SpiceChargeModel(variant="short_range", cutoff=6.)
    near = baseline(batch)
    distant = dict(batch, positions=torch.tensor([[0., 0., 0.], [40., 0., 0.]], dtype=torch.float64))
    far = baseline(distant)
    assert torch.equal(near["energy"], far["energy"])
    assert torch.equal(near["forces"], torch.zeros_like(near["forces"]))
    assert torch.equal(far["forces"], torch.zeros_like(far["forces"]))
    physical = SpiceChargeModel(cutoff=6.)
    assert float(physical(distant)["forces"].abs().max().detach()) > 1e-6
    at_cutoff = dict(batch, positions=torch.tensor([[0., 0., 0.], [6., 0., 0.]], dtype=torch.float64))
    assert torch.equal(baseline(at_cutoff)["forces"], torch.zeros((2, 3), dtype=torch.float64))


def test_proposal_stage_freezes_all_physics_even_after_generic_unfreezing():
    physical = SpiceChargeModel()
    proposal = SpiceChargeModel(stage="proposal")
    proposal.load_state_dict(copy.deepcopy(physical.state_dict()), strict=True)
    proposal.requires_grad_(True)
    before = {name: value.clone() for name, value in proposal.named_parameters()}
    output = proposal(_batch())
    loss = AmortizedVariationalLoss()(output["amortized_training"])["loss"]
    loss = loss + output["energy"].square().mean() + output["forces"].square().mean() + output["dipole"].square().mean()
    loss.backward()
    assert all(parameter.grad is None for parameter in proposal.physical.parameters())
    assert sum(float(parameter.grad.square().sum()) for parameter in proposal.proposal.parameters()) > 0
    torch.optim.Adam(proposal.parameters(), lr=.01).step()
    assert all(torch.equal(parameter, before[name]) for name, parameter in proposal.named_parameters() if name.startswith("physical."))
    assert any(not torch.equal(parameter, before[name]) for name, parameter in proposal.named_parameters() if name.startswith("proposal."))
    torch.testing.assert_close(output["amortized_training"].projected_state.reshape(-1, 2).sum(1), _batch()["total_charge"])


def test_labels_are_never_read_and_checkpoint_manifest_is_enforced():
    batch = _batch()
    model = SpiceChargeModel()
    expected = model(batch)
    contaminated = dict(batch, energy=torch.full((3,), 999.), forces=torch.full((6, 3), -123.),
                        dipole=torch.full((3, 3), 555.), targets={"total_charge": torch.full((3,), 100.)})
    for name, value in model(contaminated).items():
        torch.testing.assert_close(value, expected[name])
    metadata = _metadata()
    model.setup_from_datamodule(SimpleNamespace(metadata=metadata))
    reconstructed = SpiceChargeModel(stage="proposal")
    reconstructed.load_state_dict(copy.deepcopy(model.state_dict()))
    reconstructed.validate_dataset_metadata(metadata)
    changed = copy.deepcopy(metadata)
    changed["source_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="metadata/splits"):
        reconstructed.validate_dataset_metadata(changed)
    with pytest.raises(ValueError, match="kernel/layout/variant"):
        SpiceChargeModel(gaussian_sigma=1.).load_state_dict(model.state_dict())
    incomplete = copy.deepcopy(model.state_dict())
    del incomplete["_extra_state"]["dataset_metadata"]
    with pytest.raises(ValueError, match="missing dataset metadata"):
        SpiceChargeModel().load_state_dict(incomplete)
    metadata["splits"]["train"].append(3)
    with pytest.raises(ValueError, match="overlap"):
        SpiceChargeModel().setup_from_datamodule(SimpleNamespace(metadata=metadata))


def test_rejects_periodicity_wrong_geometry_and_missing_charge():
    model = SpiceChargeModel()
    with pytest.raises(ValueError, match="isolated"):
        model(dict(_batch(), pbc=torch.tensor([[True, False, False]])))
    with pytest.raises(ValueError, match="exactly two atoms"):
        model(dict(_batch(), n_atoms=torch.tensor([3, 3])))
    batch = _batch()
    del batch["total_charge"]
    with pytest.raises(KeyError, match="total_charge"):
        model(batch)
