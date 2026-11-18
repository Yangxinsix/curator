"""One-forward structured KAN for fixed Q, imposed potentials and their combination.

Curator compiles per-condition control work as G = U - field.T @ q. Each
forward builds physical coefficients from the owning energy's current weights.
Install the adapter before constructing optimizers or loading its state_dict.
"""
import torch
from torch.nn import functional as F

from architectures import ArchitectureChargeProposal, aggregate_prepared
from cache import output_head
from curator.layer.efa.geometry import structure_cells
from curator.layer.efa.shared import SharedGeometryChargeProposal, SharedTwoScaleQConditionedEFAEnergy
from operator_kan_architectures import OperatorCapacityProposal, install_operator_variant
from response_architectures import (
    SCALE_FLOOR, _drive_from_coefficients, _zero_charge_coefficient, install_response_variant,
    operator_matvec,
)


def _project_field(field, fixed_charge):
    """Remove the unobservable uniform field only where a total charge is fixed."""
    if field.ndim != 3 or field.shape[-1] != 1 or not field.shape[1]:
        raise ValueError("field must have shape (batch, nonempty atom axis, 1)")
    if (fixed_charge.dtype != torch.bool or fixed_charge.device != field.device
            or fixed_charge.shape != field.shape[:1]):
        raise ValueError("fixed_charge must be a per-structure boolean tensor on the field device")
    return torch.where(fixed_charge[:, None, None], field - field.mean(1, keepdim=True), field)


def boundary_residual(gradient, field, fixed_charge):
    """Batched diagnostic equivalent of the native compiled stationarity residual."""
    if gradient.shape != field.shape or gradient.dtype != field.dtype or gradient.device != field.device:
        raise ValueError("energy gradient and applied potential must match")
    return _project_field(gradient - field, fixed_charge)


def _readout_charge(drive, factors, total_charge, fixed_charge, epsilon):
    # M is SPD. Apply it simultaneously to d and 1, never to an atom-square basis.
    product = operator_matvec(torch.cat((drive, torch.ones_like(drive)), -1), *factors, epsilon)
    md, mone = product.split(1, dim=-1)
    denom = mone.sum(-2, keepdim=True)
    valid=(torch.isfinite(denom) & (denom > 0)).all()
    if denom.is_cuda and torch.cuda.is_current_stream_capturing():
        torch._assert_async(valid, "Nonpositive or nonfinite SPD charge response")
    elif not bool(valid):
        raise FloatingPointError("Nonpositive or nonfinite SPD charge response")
    fixed = fixed_charge[..., None, None]
    target = total_charge[..., None, None]
    constrained = -md + mone * ((target + md.sum(-2, keepdim=True)) / denom)
    constrained = constrained + (target - constrained.sum(-2, keepdim=True)) / drive.shape[-2]
    return torch.where(fixed, constrained, -md)


class LiveResponseEnergy(SharedTwoScaleQConditionedEFAEnergy):
    """Existing physical energy with a current-step coefficient handoff to the solver."""

    def _driving_force(self, offset, value, transpose_apply):
        coeff = _zero_charge_coefficient(offset, self.positive_output_weights,
                                         self.driving_scale, self.curvature_scale)
        response_weight = self.charge_affine.weight[:, self.hidden_features:]
        return _drive_from_coefficients(coeff, value, response_weight, transpose_apply)

    def _coefficients(self, features, geometry, transpose_apply):
        weights = self.charge_affine.weight[:, :self.hidden_features]
        offset = F.linear(geometry, weights, self.charge_affine.bias)
        value = self.charge_value(features)
        return self._driving_force(offset, value, transpose_apply), offset, value

    def prepare_proposal_context(self, context, proposal):
        context = super().prepare_proposal_context(context, proposal)
        cache = context["_shared_efa"]
        cache["live_physics"] = self._coefficients(
            context["features"], cache["geometry"],
            lambda x: aggregate_prepared(cache["prepared"], x, transpose=True, contraction="linear"),
        )
        return context

    def prepare_dense_batch(self, features, kernel, geometry, total_charge, field, fixed_charge):
        """Small-system batched evaluation. Supply current-step features, K and geometry."""
        if kernel.shape != (features.shape[0], features.shape[1], features.shape[1]):
            raise ValueError("dense geometry kernel and feature axes differ")
        if geometry.shape != (*features.shape[:2], self.hidden_features):
            raise ValueError("geometry must be the current energy's aggregated geometry Values")
        c, b, value = self._coefficients(features, geometry, lambda x: kernel.transpose(-1, -2) @ x)
        return self._dense_record(features, kernel, c, b, value, total_charge, field, fixed_charge)

    @staticmethod
    def _dense_record(features, kernel, c, b, value, total_charge, field, fixed_charge):
        if field.shape != (*features.shape[:2], 1) or total_charge.shape != features.shape[:1]:
            raise ValueError("field and total_charge must match structure and atom counts")
        for control in (field, total_charge):
            if (control.device != features.device or control.dtype != features.dtype
                    or not bool(torch.isfinite(control).all())):
                raise ValueError("finite charge/potential inputs must match feature dtype/device")
        potential = _project_field(field, fixed_charge)
        return dict(features=features, kernel=kernel, driving_c=c, offset=b, charge_value=value,
                    network_field=potential, fixed_charge=fixed_charge,
                    total_charge=torch.where(fixed_charge, total_charge, torch.zeros_like(total_charge)))

    def prepare_frozen_audit_batch(self, batch, field=None, fixed_charge=None):
        """Replay legacy immutable zero-field snapshots; requires matching frozen energy and data."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Frozen audit cache is restricted to eval under no_grad")
        features, kernel = batch["features"], batch["kernel"]
        if field is None:
            field = features.new_zeros(*features.shape[:2], 1)
        if fixed_charge is None:
            fixed_charge = torch.ones(len(features), device=features.device, dtype=torch.bool)
        offset, value = batch["offset"], batch["charge_value"]
        c = self._driving_force(offset, value, lambda x: kernel.transpose(-1, -2) @ x)
        return self._dense_record(features, kernel, c, offset, value,
                                  batch["total_charge"], field, fixed_charge)


class MultiboundaryKAN(OperatorCapacityProposal):
    def _periodic_condition(self, condition, volume):
        """Use charge in a fixed reference volume as the first network input.

        The second input Q/N and the physical Q in the constraint are unchanged.
        A scalar volume is for one geometry; dense replay may pass one per batch row.
        """
        if volume is None or getattr(self, "periodic_charge_condition", "total") == "total":
            return condition
        factor = (self.periodic_reference_volume / volume)[..., None, None]
        return torch.cat((condition[..., :1] * factor, condition[..., 1:]), dim=-1)

    def _encoder_condition(self, condition, open_flag):
        # Q and Q/N are unobserved at fixed potential. A neutral standardized
        # token and the existing open flag distinguish this from measured Q=0.
        return torch.where(open_flag, self.invariant_center[-2:], condition)

    def _conditions(self, context):
        condition, potential = super()._conditions(context)
        if getattr(self, "periodic_charge_condition", "total") == "density":
            cell = structure_cells(context, 1)[0]
            if cell is not None:
                condition = self._periodic_condition(condition, torch.linalg.det(cell).abs())
        return self._encoder_condition(condition, potential[..., 1:2].bool()), potential

    def checkpoint_contract(self):
        c = super().checkpoint_contract()
        c.update(schema="acvr_multiboundary_structured_kan_v1", scope="optional_total_charge_and_potential",
                 driving_coefficient="live_internal_energy_gradient_at_zero",
                 operator_constraint="fixedcharge:M_weighted_projection;open:-M(dU_dq_at_zero-V)",
                 physical_injection="live_internal_drive_offset_and_value_to_scalar_channels",
                 open_condition="calibration-center token for Q and Q/N plus explicit open flag",
                 constant_charge_condition_scale="unit scale on previously floored Q or Q/N channels",
                 projection="M_weighted_fixed_charge_or_direct_open",
                 native_extra_aggregations=["K.T_for_live_drive", "K_for_injection_delta"])
        return c

    def _native_physics(self, context):
        return context["_shared_efa"]["live_physics"]

    def predict_components(self, context):
        cache = context["_shared_efa"]
        c, offset, value = self._native_physics(context)
        local = self._inject(cache["local"], self._physical_input(c, offset, value))
        global_values = cache["global"] + aggregate_prepared(
            cache["prepared"], self.norm(local) - self.norm(cache["local"]), contraction="linear")
        first = self.fusion(local, global_values)
        final = self.extra_fusion(first, aggregate_prepared(
            cache["prepared"], self.norm(first), transpose=True, contraction="linear"))
        inv = self._invariants(context["features"], final, cache["condition"])
        field = context["network_field"]  # projected for fixed Q and absolute for open charge
        target = context["constraint_target"]
        fixed = bool(target.numel())  # _conditions validated the constraint layout when building local
        q = _readout_charge(c - field, self.response_factors(inv),
                            target.reshape(()) if fixed else c.new_zeros(()),
                            c.new_tensor(fixed, dtype=torch.bool), self.operator_epsilon)
        return q, torch.full_like(q, 1 / len(q))

    def driving_force_cached(self, batch):
        return batch["driving_c"]

    def cached_charges(self, batch):
        features, kernel, field = batch["features"], batch["kernel"], batch["network_field"]
        target, fixed = batch["total_charge"], batch["fixed_charge"]
        size, n, _ = features.shape
        condition = torch.stack((target, target / n), -1)[:, None, :].expand(-1, n, -1) / self.charge_scale
        condition = self._periodic_condition(condition, batch.get("periodic_volume"))
        flag = (~fixed)[:, None, None].expand_as(field).to(field.dtype)
        condition = self._encoder_condition(condition, flag.bool())
        potential = torch.cat((field / self.potential_scale, flag), -1)
        c = batch["driving_c"]
        local = self.input(features) + self.boundary_input(condition) + self.potential_input(potential)
        local = self._inject(local, self._physical_input(c, batch["offset"], batch["charge_value"]))
        first = self.fusion(local.flatten(0, 1), (kernel @ self.norm(local)).flatten(0, 1)).reshape(size, n, -1)
        incoming = kernel.transpose(-1, -2) @ self.norm(first)
        final = self.extra_fusion(first.flatten(0, 1), incoming.flatten(0, 1)).reshape(size, n, -1)
        factors = self.response_factors(self._invariants(features, final, condition))
        return _readout_charge(c - field, factors, target, fixed, self.operator_epsilon)


def install_multiboundary(model):
    """Upgrade a loaded final R16/KAN5 model. Preserve all trainable parameter objects."""
    head = output_head(model)
    energy, proposal = head.internal_energy, head.proposal
    if type(energy) is LiveResponseEnergy and type(proposal) is MultiboundaryKAN:
        return proposal
    if (type(energy) is not SharedTwoScaleQConditionedEFAEnergy
            or type(proposal) is not OperatorCapacityProposal
            or proposal.operator_capacity_variant != "kan5" or proposal.response_rank != 16):
        raise TypeError("Expected shared two-scale scalar charge energy and final rank16 KAN5")
    if head.proposal_mode != "absolute":
        raise ValueError("Structured charge output requires absolute proposal mode")
    energy.__class__ = LiveResponseEnergy
    proposal.__class__ = MultiboundaryKAN
    # A single-Q calibration places the Q channel at the numerical std floor.
    # A different physical Q would then look like ~1e6 standard deviations.
    # Use one model charge unit for unresolved boundary widths. Existing
    # observations at their constant training center retain exactly zero input.
    with torch.no_grad():
        charge_scales = proposal.invariant_scale[-2:]
        charge_scales.masked_fill_(charge_scales <= SCALE_FLOOR, 1.0)
    for name in tuple(proposal._buffers):
        if name.startswith("frozen_"):
            delattr(proposal, name)
    del proposal.frozen_charge_value
    energy.attention.contraction = proposal.attention_contraction = "linear"
    return proposal


def install_scratch_multiboundary(model, *, seed=3407):
    """Build the final structured solver from a newly initialized, bound shared energy/model.

    Starts with M = I/hardness, a nonzero U and a zero V so the first V update
    has a signal. Spline coefficients start at zero on the original affine base.
    Fresh physical/invariant/spline inputs use identity normalization.
    """
    head = output_head(model)
    energy, original = head.internal_energy, head.proposal
    if type(energy) is not SharedTwoScaleQConditionedEFAEnergy or type(original) is not SharedGeometryChargeProposal:
        raise TypeError("Scratch installation requires freshly bound SharedTwoScale energy and SharedGeometry proposal")
    if head.proposal_mode != "absolute" or original.feature_spec is None:
        raise ValueError("Bind the compatible absolute charge proposal before installation")
    anchor = next(energy.parameters())
    width = 1 + energy.hidden_features + energy.charge_features
    invariant_width = (sum(multiplicity for multiplicity, _ in original.feature_spec.irreps)
                       + sum(multiplicity for multiplicity, _ in original.irreps) + 2)
    def identity(count):
        return {"center": anchor.new_zeros(count), "scale": anchor.new_ones(count)}
    calibration = identity(width)
    calibration.update(invariant_center=anchor.new_zeros(invariant_width),
                       invariant_scale=anchor.new_ones(invariant_width))
    flags = [(p, p.requires_grad) for p in model.parameters()]
    contraction = energy.attention.contraction
    try:
        head.proposal = ArchitectureChargeProposal.from_shared(original, "C", seed=seed, contraction="linear")
        install_response_variant(model, "operator", calibration, seed=seed, rank=16)
        install_operator_variant(model, "kan5", {"input": identity(invariant_width), "hidden": identity(64)}, seed=seed)
        return install_multiboundary(model)
    except Exception:
        head.proposal = original
        energy.attention.contraction = contraction
        raise
    finally:
        for parameter, requires_grad in flags:
            parameter.requires_grad_(requires_grad)
