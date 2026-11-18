"""Default joint heads respect field gauge at nonzero fixed total charge."""

from dataclasses import asdict

import pytest
import torch

from curator.data import properties
from curator.data.variational_data import canonical_boundary_hash
from curator.layer import PairwiseDistance, VariationalEnergyOutput
from curator.model import NeuralNetworkPotential, Painn
from curator.variational.spec import (
    BoundarySpec, ConstraintSpec, ControlSpec, GaugeSpec, GroupSpec, VariableSpec,
)


def _batch(shift=0.0, *, live_shift=None):
    variable = VariableSpec("charge", "scalar", 1, "e", "eV/e")
    groups = (GroupSpec.all(),) + tuple(GroupSpec.from_indices(f"site{i}", [i]) for i in range(3))
    boundary = BoundarySpec(
        variable, groups=groups,
        equalities=(ConstraintSpec.group_sum("fixed_charge", "all", 0.7, unit="e"),),
        controls=tuple(ControlSpec.group_potential(f"potential{i}", f"site{i}", value + shift,
                                                  unit="eV/e")
                       for i, value in enumerate((0.2, -0.1, 0.4))),
        gauge=GaugeSpec(mode="allow"),
    )
    payload = asdict(boundary)
    batch = {
        properties.Z: torch.tensor([1, 6, 8]),
        properties.positions: torch.tensor([[0., 0., 0.], [1.1, 0.2, 0.], [0.1, 1., 0.3]], dtype=torch.float64),
        properties.edge_idx: torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]]),
        properties.n_atoms: torch.tensor([3]), properties.image_idx: torch.tensor([0, 0, 0]),
        "variational": {
            "condition_geometry_index": torch.tensor([0]), "condition_state_ptr": torch.tensor([0, 3]),
            "boundary_spec": (payload,), "boundary_hash": (canonical_boundary_hash(payload),),
            "gauge": (asdict(boundary.gauge),),
        },
    }
    if live_shift is not None:
        batch["external_field"] = live_shift.reshape(1, 1)
    return batch


def _model(mode):
    torch.manual_seed(19)
    return NeuralNetworkPotential(
        Painn(2, 8, 5.0, num_basis=4),
        input_modules=[PairwiseDistance(compute_distance_from_R=True)],
        output_modules=[VariationalEnergyOutput(
            state_unit="e", conjugate_unit="eV/e", hidden_features=8,
            state_mode=mode, compute_responses=True,
        )],
    ).double().eval()


@pytest.mark.parametrize("mode", ["predicted", "converged"])
def test_fixed_nonzero_charge_preserves_state_and_energy_gauge_shift(mode):
    model = _model(mode)
    first = model(_batch())["variational_prediction"]
    shifted = model(_batch(shift=0.31))["variational_prediction"]
    for key in ("variable_value", "condition_forces"):
        torch.testing.assert_close(shifted["labels"][key]["values"], first["labels"][key]["values"],
                                   rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(first["labels"]["variable_value"]["values"].sum(),
                               torch.tensor(0.7, dtype=torch.float64))
    difference = shifted["labels"]["condition_energy"]["values"] - first["labels"]["condition_energy"]["values"]
    torch.testing.assert_close(difference, torch.tensor([-0.7 * 0.31], dtype=torch.float64), rtol=1e-8, atol=1e-10)
    response = first["responses"]["values"].reshape(3, 3)
    torch.testing.assert_close(response @ torch.ones(3, dtype=torch.float64),
                               torch.zeros(3, dtype=torch.float64), rtol=0, atol=1e-9)
    torch.testing.assert_close(shifted["responses"]["values"], first["responses"]["values"],
                               rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("mode", ["predicted", "converged"])
def test_live_uniform_field_has_only_the_fixed_charge_energy_derivative(mode):
    model = _model(mode)
    shift = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    prediction = model(_batch(live_shift=shift))["variational_prediction"]
    energy = prediction["labels"]["condition_energy"]["values"].sum()
    derivative = torch.autograd.grad(energy, shift, retain_graph=True)[0]
    torch.testing.assert_close(derivative, torch.tensor(-0.7, dtype=torch.float64), rtol=1e-8, atol=1e-10)
    if mode == "predicted":
        state = prediction["labels"]["variable_value"]["values"]
        for value in state:
            derivative = torch.autograd.grad(value, shift, retain_graph=True)[0]
            torch.testing.assert_close(derivative, torch.tensor(0., dtype=torch.float64), rtol=0, atol=1e-10)
