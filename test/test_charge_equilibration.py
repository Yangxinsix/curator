import torch
from torch import nn

from curator.data import properties
from curator.data.properties import HeadConfig
from curator.layer import (
    AtomwiseNN,
    ChargeEquilibration,
    EwaldSummation,
    GlobalRescaleShift,
)
from curator.train import ModelOutput
from curator.utils import scatter_add


class FixedReadout(AtomwiseNN):
    def __init__(self, values):
        nn.Module.__init__(self)
        self.register_buffer("values", torch.as_tensor(values))

    def _compute(self, features):
        return self.values[: features.shape[0]].unsqueeze(-1)


class QuadraticEwald(EwaldSummation):
    """Small analytic Ewald stand-in: 1/2 sum_i (1 + x_i^2) q_i^2."""

    def forward(self, data, ewald_kernel=None):
        del ewald_kernel
        charge = data[properties.atomic_charge]
        x = data[properties.positions][:, 0]
        atomic_energy = 0.5 * (1.0 + x.square()) * charge.square()
        return scatter_add(atomic_energy, data[properties.image_idx], dim=0)


def test_inverse_hardness_charge_projection():
    raw_charge = torch.tensor([0.2, -0.1, 0.4, 0.3])
    hardness = torch.tensor([1.0, 2.0, 4.0, 0.5])
    image_idx = torch.tensor([0, 0, 0, 1])
    total_charge = torch.tensor([1.0, -0.5])

    charge = ChargeEquilibration._conserve_total_charge(
        raw_charge,
        hardness,
        image_idx,
        total_charge,
    )

    assert torch.allclose(scatter_add(charge, image_idx), total_charge)

    correction = charge[:3] - raw_charge[:3]
    expected_ratio = hardness[:3].reciprocal()
    assert torch.allclose(
        correction / correction.sum(),
        expected_ratio / expected_ratio.sum(),
    )


def test_penalty_output_does_not_require_a_target_label():
    output = ModelOutput(
        name=properties.chemical_potential_residual_normalized,
        loss_weight=2.0,
        is_penalty=True,
    )
    residual = torch.tensor([1.0, -2.0])

    loss, num_observations = output.calculate_loss(
        {properties.chemical_potential_residual_normalized: residual},
        {properties.energy: torch.tensor([0.0])},
    )

    assert torch.allclose(loss, 2.0 * residual.square().mean())
    assert num_observations == 1


def test_charge_equilibration_outputs_physical_terms_without_composing_totals():
    positions = torch.tensor(
        [[0.2, 0.0, 0.0], [0.7, 0.0, 0.0], [-0.3, 0.0, 0.0]],
        requires_grad=True,
    )
    base_charge = torch.tensor([0.1, -0.2, 0.4], requires_grad=True)
    raw_charge = base_charge + 0.1 * positions[:, 0]
    chi = torch.tensor([0.3, 0.2, 0.1])
    hardness = torch.tensor([1.0, 2.0, 4.0])
    raw_chi = chi.sqrt()
    raw_hardness = (0.5 * hardness).sqrt()
    image_idx = torch.zeros(3, dtype=torch.long)

    module = ChargeEquilibration(
        num_features=2,
        electronegativity_mlp=FixedReadout(raw_chi),
        hardness_mlp=FixedReadout(raw_hardness),
        ewald=QuadraticEwald(),
        min_hardness=0.0,
    )
    data = {
        properties.positions: positions,
        properties.node_embedding: torch.zeros(3, 2),
        properties.atomic_charge: raw_charge,
        properties.total_charge: torch.tensor([1.0]),
        properties.image_idx: image_idx,
        properties.energy: torch.zeros(1),
        properties.forces: torch.zeros_like(positions),
    }

    output = module(data, training=True)
    charge = output[properties.atomic_charge]

    inverse_hardness = hardness.reciprocal()
    weights = inverse_hardness / inverse_hardness.sum()
    expected_charge = raw_charge + weights * (1.0 - raw_charge.sum())
    assert torch.allclose(charge, expected_charge)

    ewald_curvature = 1.0 + positions[:, 0].square()
    chemical_potential = chi + hardness * charge + ewald_curvature * charge
    expected_residual = chemical_potential - chemical_potential.mean()
    assert torch.allclose(
        output[properties.chemical_potential_residual],
        expected_residual,
    )
    assert torch.allclose(
        output[properties.chemical_potential_residual].sum(),
        torch.zeros(()),
        atol=1.0e-6,
    )

    expected_ewald_force_x = -positions[:, 0] * charge.detach().square()
    assert torch.allclose(
        output[properties.ewald_forces][:, 0],
        expected_ewald_force_x,
    )
    assert torch.equal(output[properties.energy], data[properties.energy])
    assert torch.equal(output[properties.forces], data[properties.forces])
    assert torch.allclose(
        output[properties.qeq_energy],
        output[properties.onsite_energy] + output[properties.ewald_energy],
    )

    loss = output[properties.chemical_potential_residual].square().mean()
    charge_gradient = torch.autograd.grad(loss, base_charge)[0]
    assert torch.isfinite(charge_gradient).all()
    assert not torch.allclose(charge_gradient, torch.zeros_like(charge_gradient))


def test_rescale_composes_generic_physical_contributions_in_the_right_units():
    energy_scale = 4.0
    energy_shift = 10.0
    rescale = GlobalRescaleShift(
        heads=[
            HeadConfig(
                key=properties.energy,
                is_atomwise=False,
                reduction=None,
                scale_by=energy_scale,
                shift_by=energy_shift,
                atomwise_normalization=False,
            ),
            HeadConfig(
                key=properties.forces,
                is_atomwise=True,
                reduction=None,
                scale_by=energy_scale,
                shift_by=False,
            ),
        ],
        physical_contributions=[
            {
                "source": properties.qeq_energy,
                "destination": properties.energy,
                "scale_like": properties.energy,
            },
            {
                "source": properties.ewald_forces,
                "destination": properties.forces,
                "scale_like": properties.forces,
            },
        ],
        normalized_copies=[
            {
                "source": properties.chemical_potential_residual,
                "destination": properties.chemical_potential_residual_normalized,
                "scale_like": properties.energy,
            }
        ],
    )
    local_energy_normalized = torch.tensor([2.0])
    local_forces_normalized = torch.full((2, 3), 3.0)
    qeq_energy_physical = torch.tensor([8.0])
    ewald_forces_physical = torch.full((2, 3), 4.0)
    residual_physical = torch.tensor([12.0, -12.0])
    data = {
        properties.energy: local_energy_normalized,
        properties.forces: local_forces_normalized,
        properties.qeq_energy: qeq_energy_physical,
        properties.ewald_forces: ewald_forces_physical,
        properties.chemical_potential_residual: residual_physical,
    }

    rescale.train()
    normalized = rescale(data)
    torch.testing.assert_close(normalized[properties.energy], torch.tensor([4.0]))
    torch.testing.assert_close(
        normalized[properties.forces],
        torch.full((2, 3), 4.0),
    )
    torch.testing.assert_close(
        normalized[properties.chemical_potential_residual_normalized],
        residual_physical / energy_scale,
    )
    torch.testing.assert_close(
        normalized[properties.qeq_energy],
        qeq_energy_physical,
    )
    torch.testing.assert_close(
        normalized[properties.ewald_forces],
        ewald_forces_physical,
    )

    rescale.eval()
    physical = rescale(data)
    torch.testing.assert_close(physical[properties.energy], torch.tensor([26.0]))
    torch.testing.assert_close(
        physical[properties.forces],
        torch.full((2, 3), 16.0),
    )
    torch.testing.assert_close(
        physical[properties.chemical_potential_residual_normalized],
        residual_physical / energy_scale,
    )

    roundtrip = rescale.unscale(physical, force_process=True)
    torch.testing.assert_close(
        roundtrip[properties.energy],
        normalized[properties.energy],
    )
    torch.testing.assert_close(
        roundtrip[properties.forces],
        normalized[properties.forces],
    )
