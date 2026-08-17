"""Characterization tests for the legacy QEq force outputs.

These tests intentionally do not prescribe the future ACVR force API.  They
record the exact meaning of the two legacy properties so that a later adapter
can preserve compatibility without confusing either property with the force
of the complete QEq energy.
"""

import torch
from torch import nn

from curator.data import properties
from curator.layer import AtomwiseNN, ChargeEquilibration, EwaldSummation
from curator.utils import scatter_add


DTYPE = torch.float64


class _AffineFeatureReadout(AtomwiseNN):
    """Return ``bias + scale * feature[:, 0]`` without extra parameters."""

    def __init__(self, bias, scale):
        nn.Module.__init__(self)
        self.bias = bias
        self.scale = scale

    def _compute(self, features):
        return (self.bias + self.scale * features[:, :1])


class _CoordinateQuadraticEwald(EwaldSummation):
    r"""Analytic stand-in: ``E = 1/2 sum_i (1 + x_i^2) q_i^2``."""

    def forward(self, data, ewald_kernel=None):
        del ewald_kernel
        charge = data[properties.atomic_charge]
        x = data[properties.positions][:, 0]
        atomic_energy = 0.5 * (1.0 + x.square()) * charge.square()
        return scatter_add(atomic_energy, data[properties.image_idx], dim=0)


def _run_legacy_module():
    positions = torch.tensor(
        [[0.2, 0.0, 0.0], [0.7, 0.0, 0.0], [-0.4, 0.0, 0.0]],
        dtype=DTYPE,
        requires_grad=True,
    )
    x = positions[:, 0]
    node_embedding = torch.stack((x, x.square()), dim=1)
    raw_charge = torch.tensor([0.15, -0.25, 0.35], dtype=DTYPE) + torch.tensor(
        [0.12, -0.08, 0.05], dtype=DTYPE
    ) * x
    image_idx = torch.zeros(3, dtype=torch.long)
    chi_readout = _AffineFeatureReadout(bias=0.8, scale=0.25)
    hardness_readout = _AffineFeatureReadout(bias=1.1, scale=-0.15)
    module = ChargeEquilibration(
        num_features=2,
        electronegativity_mlp=chi_readout,
        hardness_mlp=hardness_readout,
        ewald=_CoordinateQuadraticEwald(),
        min_hardness=0.2,
        model_outputs=[
            properties.atomic_charge,
            properties.chemical_potential_residual,
            properties.residual_forces,
        ],
    )
    output = module(
        {
            properties.positions: positions,
            properties.node_embedding: node_embedding,
            properties.atomic_charge: raw_charge,
            properties.total_charge: torch.tensor([0.4], dtype=DTYPE),
            properties.image_idx: image_idx,
        },
        training=True,
    )
    return module, positions, node_embedding, output


def _fixed_state_energy_terms(module, positions, node_embedding, charge):
    """Rebuild the two QEq terms with charge detached from its coordinate path."""

    fixed_charge = charge.detach()
    raw_chi = module.electronegativity_mlp._compute(node_embedding).squeeze(-1)
    raw_hardness = module.hardness_mlp._compute(node_embedding).squeeze(-1)
    chi = raw_chi.square()
    hardness = 2.0 * raw_hardness.square() + module.min_hardness
    onsite = (
        chi * fixed_charge + 0.5 * hardness * fixed_charge.square()
    ).sum()
    x = positions[:, 0]
    ewald = (0.5 * (1.0 + x.square()) * fixed_charge.square()).sum()
    return onsite, ewald


def test_legacy_ewald_forces_are_only_the_fixed_charge_ewald_component():
    module, positions, node_embedding, output = _run_legacy_module()
    charge = output[properties.atomic_charge]
    onsite, ewald = _fixed_state_energy_terms(
        module, positions, node_embedding, charge
    )

    ewald_gradient = torch.autograd.grad(
        ewald, positions, retain_graph=True
    )[0]
    onsite_gradient = torch.autograd.grad(
        onsite, positions, retain_graph=True
    )[0]
    complete_envelope_gradient = torch.autograd.grad(
        onsite + ewald, positions, retain_graph=True
    )[0]

    torch.testing.assert_close(
        output[properties.ewald_forces],
        -ewald_gradient,
        atol=2.0e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        -complete_envelope_gradient,
        output[properties.ewald_forces] - onsite_gradient,
        atol=2.0e-14,
        rtol=0.0,
    )
    assert torch.linalg.vector_norm(onsite_gradient) > 1.0e-3
    assert not torch.allclose(
        output[properties.ewald_forces],
        -complete_envelope_gradient,
    )


def test_legacy_residual_forces_are_positive_energy_gradient_correction():
    module, positions, node_embedding, output = _run_legacy_module()
    charge = output[properties.atomic_charge]
    residual = output[properties.chemical_potential_residual]
    onsite, ewald = _fixed_state_energy_terms(
        module, positions, node_embedding, charge
    )

    response_energy_gradient = torch.autograd.grad(
        charge,
        positions,
        grad_outputs=residual.detach(),
        retain_graph=True,
    )[0]
    complete_envelope_gradient = torch.autograd.grad(
        onsite + ewald, positions, retain_graph=True
    )[0]
    complete_total_gradient = torch.autograd.grad(
        output[properties.qeq_energy].sum(),
        positions,
        retain_graph=True,
    )[0]

    torch.testing.assert_close(
        output[properties.residual_forces],
        response_energy_gradient,
        atol=3.0e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        complete_total_gradient - complete_envelope_gradient,
        response_energy_gradient,
        atol=3.0e-14,
        rtol=0.0,
    )

    complete_full_force = -complete_total_gradient
    complete_envelope_force = -complete_envelope_gradient
    torch.testing.assert_close(
        complete_full_force - complete_envelope_force,
        -output[properties.residual_forces],
        atol=3.0e-14,
        rtol=0.0,
    )
    assert torch.linalg.vector_norm(response_energy_gradient) > 1.0e-5
