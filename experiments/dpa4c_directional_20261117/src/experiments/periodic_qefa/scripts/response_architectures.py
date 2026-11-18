"""Fixed-forward, energy-driven charge response operators.

The frozen energy derivative at q=0 is an explicit coefficient input, not a
predicted-charge residual feedback loop. The native path uses factorized
EFA contractions; only the existing small-structure training cache stores K.
No Hessian, inverse, factorization, linear solve, or charge iteration appears
in this module. Operator arms deliberately have a different initial charge
function from the warm baseline; early injection is function preserving.
"""
import copy
import math

import torch
from e3nn import o3
from torch import nn
from torch.nn import functional as F

from architectures import ArchitectureChargeProposal, aggregate_prepared
from cache import output_head
from curator.layer.efa.readout import invariant_features


VARIANTS = ("baseline", "early", "operator", "operator_nonlinear")
PHYSICAL_KEYS = {"features", "condition", "potential", "kernel", "offset",
                 "charge_value", "geometry_energy", "total_charge", "source_index"}
OPERATOR_EPSILON = 1e-8
SCALE_FLOOR = 1e-6


def _frozen_module(module):
    module = copy.deepcopy(module)
    for child in module.modules():
        for name, parameter in list(child._parameters.items()):
            if parameter is not None:
                del child._parameters[name]
                child.register_buffer(name, parameter.detach().clone())
    return module


def _zero_charge_coefficient(offset, output_weights, driving_scale, curvature_scale):
    """Derivative through native softplus, including its linear branch."""
    initial = torch.sigmoid(offset)
    derivative = torch.where(offset > 20, torch.ones_like(offset), initial)
    return output_weights * (driving_scale * initial
                             + curvature_scale * (derivative - initial))


def _drive_from_coefficients(coefficient, charge_value, response_weight, transpose_apply):
    remote = transpose_apply(F.linear(coefficient, response_weight[:, :-1].T))
    return ((remote * charge_value).sum(-1, keepdim=True)
            + (coefficient * response_weight[:, -1]).sum(-1, keepdim=True))


def _check_cached_scope(batch):
    if bool(torch.count_nonzero(batch["potential"])):
        raise ValueError("Response campaign supports fixed Q without external potentials only")
    target = batch["total_charge"]
    if target.shape != batch["features"].shape[:1] or not bool(torch.isfinite(target).all()):
        raise ValueError("Fixed total charge must be finite, one value per structure")


def operator_matvec(vector, s, u, v, eps=OPERATOR_EPSILON):
    """Apply M = eps I + L L.T, L = diag(s) + U V.T without dense N-by-N arrays.

The atom axis is -2. Inputs may be one structure [N, channels] or batched.
The caller supplies rank-normalized U/V. M is SPD for eps > 0, even if L
becomes singular. This function does not construct M or solve a system.
"""
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("operator epsilon must be finite and strictly positive")
    if (s.shape[-1] != 1 or u.shape != v.shape or s.shape[:-1] != u.shape[:-1]
            or vector.shape[:-1] != s.shape[:-1]):
        raise ValueError("Expected matching [..., N, channels] operator operands")
    transpose = s * vector + v @ (u.transpose(-1, -2) @ vector)
    return eps * vector + s * transpose + u @ (v.transpose(-1, -2) @ transpose)


def operator_charge(c, s, u, v, total_charge, eps=OPERATOR_EPSILON):
    """Explicit fixed-Q response with the M-weighted charge projection.

For fixed M this is invariant under c -> c + constant. The encoder itself
may depend on c, so this property is not asserted for the full network.
"""
    if c.shape[-1] != 1:
        raise ValueError("Driving coefficient must have one scalar per atom")
    both = operator_matvec(torch.cat((c, torch.ones_like(c)), -1), s, u, v, eps)
    mc, mone = both[..., :1], both[..., 1:]
    target = torch.as_tensor(total_charge, device=c.device, dtype=c.dtype)
    target = target.reshape(c.shape[:-2] + (1, 1))
    denominator = mone.sum(-2, keepdim=True)
    # Mathematically >= eps*N. Do not silently clamp a failed computation.
    if not bool(torch.isfinite(denominator).all() and (denominator > 0).all()):
        raise FloatingPointError("Nonpositive or nonfinite SPD projection denominator")
    charge = -mc + mone * ((target + mc.sum(-2, keepdim=True)) / denominator)
    return charge + (target - charge.sum(-2, keepdim=True)) / c.shape[-2]


class ResponseChargeProposal(ArchitectureChargeProposal):
    """Two hidden communication stages with optional learned inverse response."""

    def checkpoint_contract(self):
        if not hasattr(self, "response_variant"):
            return super().checkpoint_contract()
        variant = self.response_variant
        return {**self.response_parent_contract,
                "schema": "energy_driven_response_proposal_v1",
                "response_variant": variant,
                "response_rank": self.response_rank if variant.startswith("operator") else 0,
                "driving_coefficient": "exact_frozen_dE_dq_at_q_zero_softplus_threshold_20",
                "physical_input_order": ["driving_c", "offset_b", "charge_value_B"],
                "physical_injection": "zero_initialized_scalar_h0_projection"
                                      if variant != "baseline" else "none",
                "operator": "eps_I_plus_(diag_s_plus_UVt)(diag_s_plus_UVt)t"
                            if variant.startswith("operator") else None,
                "operator_epsilon": self.operator_epsilon if variant.startswith("operator") else None,
                "rank_normalization": "U_and_V_divided_by_sqrt_atom_count",
                "operator_feature_normalization": "fixed_current_warm_training_invariant_mean_std",
                "operator_constraint": "q=-M_c+M_1*(Q+sum(M_c))/sum(M_1)",
                "scope": "fixed_total_charge_no_external_potential",
                "readout_widths": ([self.response_readout_width, 64, 1 + 2 * self.response_rank]
                                   if variant.startswith("operator")
                                   else self.response_parent_contract.get("readout_widths")),
                "kan_layers": [] if variant.startswith("operator")
                              else self.response_parent_contract.get("kan_layers", []),
                "hidden_communications": ["K", "K.T"],
                "native_contraction": "factorized_linear",
                "native_extra_aggregations": [] if variant == "baseline"
                                             else ["K.T_for_driving_c", "K_for_early_injection_delta"],
                "inference": "one_fixed_forward_analytic_q_zero_features_no_predicted_charge_feedback_no_solve"}

    def _coefficient(self, offset):
        return _zero_charge_coefficient(offset, self.frozen_output_weights,
                                        self.frozen_driving_scale, self.frozen_curvature_scale)

    def driving_force_cached(self, batch):
        coefficient = self._coefficient(batch["offset"])
        return _drive_from_coefficients(
            coefficient, batch["charge_value"], self.frozen_response_weight,
            lambda values: batch["kernel"].transpose(-1, -2) @ values)

    def _native_physics(self, context):
        cache = context["_shared_efa"]
        offset = F.linear(cache["geometry"], self.frozen_offset_weight, self.frozen_offset_bias)
        charge_value = self.frozen_charge_value(context["features"])
        c = _drive_from_coefficients(
            self._coefficient(offset), charge_value, self.frozen_response_weight,
            lambda values: aggregate_prepared(cache["prepared"], values,
                                               transpose=True, contraction="linear"))
        return c, offset, charge_value

    def driving_force_native(self, context):
        return self._native_physics(context)[0]

    def _physical_input(self, c, offset, charge_value):
        return (torch.cat((c, offset, charge_value), -1) - self.physical_center) / self.physical_scale

    def _inject(self, local, physical):
        # Only 0e channels receive invariant physical scalars. Vector entries
        # remain unchanged, preserving the trunk's O(3) transformation law.
        correction = self.physical_input(physical)
        return local + F.linear(correction, self.scalar_embedding)

    def _invariants(self, features, hidden, condition):
        return torch.cat((invariant_features(features, self.feature_spec.irreps),
                          invariant_features(hidden, self.irreps), condition), -1)

    def response_factors(self, invariants):
        """Return s,U,V with positive s and structure-size-normalized factors."""
        raw = self.operator_head((invariants - self.invariant_center) / self.invariant_scale)
        rank = self.response_rank
        s = F.softplus(raw[..., :1]) + self.operator_s_floor
        u, v = raw[..., 1:1 + rank], raw[..., 1 + rank:]
        normalizer = math.sqrt(invariants.shape[-2])
        return s, u / normalizer, v / normalizer

    def _operator_output(self, invariants, c, total_charge):
        s, u, v = self.response_factors(invariants)
        charge = operator_charge(c, s, u, v, total_charge, self.operator_epsilon)
        if self.response_variant == "operator_nonlinear":
            correction = self.nonlinear_head((invariants - self.invariant_center) / self.invariant_scale)
            charge = charge + correction - correction.mean(-2, keepdim=True)
        target = torch.as_tensor(total_charge, device=charge.device, dtype=charge.dtype)
        target = target.reshape(charge.shape[:-2] + (1, 1))
        return charge + (target - charge.sum(-2, keepdim=True)) / charge.shape[-2]

    def predict_components(self, context):
        target = self._total_charge(context)
        if self.response_variant == "baseline":
            return super().predict_components(context)
        cache = context["_shared_efa"]
        c, offset, charge_value = self._native_physics(context)
        local = self._inject(cache["local"], self._physical_input(c, offset, charge_value))
        # Energy preparation precedes access to b. Reuse its old first pass
        # and add K(norm(new_local)-norm(old_local)); include this extra native
        # aggregation in inference timings rather than pretending it is free.
        global_values = cache["global"] + aggregate_prepared(
            cache["prepared"], self.norm(local) - self.norm(cache["local"]), contraction="linear")
        first = self.fusion(local, global_values)
        global_values = aggregate_prepared(cache["prepared"], self.norm(first),
                                            transpose=True, contraction="linear")
        final = self.extra_fusion(first, global_values)
        invariants = self._invariants(context["features"], final, cache["condition"])
        if self.response_variant == "early":
            output = self.readout(invariants)
            return self.charge_scale * output[:, :1], torch.softmax(output[:, 1:], dim=0)
        charge = self._operator_output(invariants, c, target)
        # The standard outer fixed-Q projection therefore changes at most
        # roundoff. These are feasible charges, not an intermediate state.
        return charge, torch.ones_like(charge) / len(charge)

    def cached_charges(self, batch):
        _check_cached_scope(batch)
        if self.response_variant == "baseline":
            return super().cached_charges(batch)
        features, condition = batch["features"], batch["condition"]
        structures, atoms, _ = features.shape
        c = self.driving_force_cached(batch)
        physical = self._physical_input(c, batch["offset"], batch["charge_value"])
        local = (self.input(features) + self.boundary_input(condition)
                 + self.potential_input(batch["potential"]))
        local = self._inject(local, physical)
        first_global = batch["kernel"] @ self.norm(local)
        first = self.fusion(local.flatten(0, 1), first_global.flatten(0, 1)).reshape(structures, atoms, -1)
        second_global = batch["kernel"].transpose(-1, -2) @ self.norm(first)
        final = self.extra_fusion(first.flatten(0, 1), second_global.flatten(0, 1)).reshape(structures, atoms, -1)
        invariants = self._invariants(features, final, condition)
        if self.response_variant.startswith("operator"):
            return self._operator_output(invariants, c, batch["total_charge"])
        output = self.readout(invariants)
        raw = self.charge_scale * output[..., :1]
        weights = torch.softmax(output[..., 1:], dim=1)
        target = batch["total_charge"][:, None, None]
        charge = raw + weights * (target - raw.sum(1, keepdim=True))
        return charge + (target - charge.sum(1, keepdim=True)) / atoms


@torch.no_grad()
def collect_response_calibration(model, batches):
    """Training-only population moments of [c,b,B], without teacher inputs."""
    head = output_head(model)
    energy, proposal = head.internal_energy, head.proposal
    weight = energy.charge_affine.weight[:, energy.hidden_features:]
    count, structures, mean, m2 = 0, 0, None, None
    invariant_mean, invariant_m2 = None, None
    for batch in batches:
        if set(batch) != PHYSICAL_KEYS:
            raise ValueError("Calibration accepts exact physical cache fields only")
        _check_cached_scope(batch)
        coefficient = _zero_charge_coefficient(batch["offset"], energy.positive_output_weights,
                                               energy.driving_scale, energy.curvature_scale)
        c = _drive_from_coefficients(coefficient, batch["charge_value"], weight,
                                    lambda values: batch["kernel"].transpose(-1, -2) @ values)
        values = torch.cat((c, batch["offset"], batch["charge_value"]), -1).flatten(0, 1)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Nonfinite physical coefficient calibration")
        n, current = len(values), values.mean(0)
        current_m2 = (values - current).square().sum(0)
        features, condition = batch["features"], batch["condition"]
        bs, atoms, _ = features.shape
        local = (proposal.input(features) + proposal.boundary_input(condition)
                 + proposal.potential_input(batch["potential"]))
        global_values = batch["kernel"] @ proposal.norm(local)
        first = proposal.fusion(local.flatten(0, 1), global_values.flatten(0, 1)).reshape(bs, atoms, -1)
        global_values = batch["kernel"].transpose(-1, -2) @ proposal.norm(first)
        final = proposal.extra_fusion(first.flatten(0, 1), global_values.flatten(0, 1)).reshape(bs, atoms, -1)
        invariant = torch.cat((invariant_features(features, proposal.feature_spec.irreps),
                               invariant_features(final, proposal.irreps), condition), -1).flatten(0, 1)
        if not bool(torch.isfinite(invariant).all()):
            raise ValueError("Nonfinite warm proposal invariant calibration")
        current_invariant = invariant.mean(0)
        current_invariant_m2 = (invariant - current_invariant).square().sum(0)
        if count:
            delta, total = current - mean, count + n
            mean, m2 = (mean + delta * n / total,
                        m2 + current_m2 + delta.square() * count * n / total)
            invariant_delta = current_invariant - invariant_mean
            invariant_mean, invariant_m2 = (
                invariant_mean + invariant_delta * n / total,
                invariant_m2 + current_invariant_m2 + invariant_delta.square() * count * n / total)
        else:
            mean, m2 = current, current_m2
            invariant_mean, invariant_m2 = current_invariant, current_invariant_m2
        count += n
        structures += batch["features"].shape[0]
    if not count:
        raise ValueError("Calibration requires nonempty training batches")
    std = (m2 / count).sqrt()
    invariant_std = (invariant_m2 / count).sqrt()
    return {"center": mean.detach().clone(), "scale": std.clamp_min(SCALE_FLOOR),
            "std": std, "atom_count": count, "structure_count": structures,
            "floored_channels": int((std < SCALE_FLOOR).sum()), "scale_floor": SCALE_FLOOR,
            "source": "training_physical_inputs_only_frozen_dE_dq_at_zero",
            "invariant_center": invariant_mean.detach().clone(),
            "invariant_scale": invariant_std.clamp_min(SCALE_FLOOR),
            "invariant_std": invariant_std,
            "invariant_floored_channels": int((invariant_std < SCALE_FLOOR).sum()),
            "input_order": ["driving_c", "offset_b", "charge_value_B"]}


def install_response_variant(model, variant, calibration, seed=3407, rank=8):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank not in (4, 8, 16):
        raise ValueError("Response rank must be 4, 8, or 16")
    head = output_head(model)
    original, energy = head.proposal, head.internal_energy
    if (not isinstance(original, ArchitectureChargeProposal) or original.variant != "C"
            or hasattr(original, "response_variant") or hasattr(original, "physics_variant")):
        raise TypeError("Install once into an original C or capacity-C proposal")
    if not hasattr(original, "extra_fusion"):
        raise ValueError("Both K and K.T hidden communication stages are required")
    converted = copy.deepcopy(original)
    converted.__class__ = ResponseChargeProposal
    converted.response_parent_contract = copy.deepcopy(original.checkpoint_contract())
    converted.response_variant, converted.response_rank = variant, rank
    converted.attention_contraction = "linear"
    converted.operator_epsilon = OPERATOR_EPSILON
    converted.operator_s_floor = 1e-6
    converted.response_readout_width = (sum(m for m, _ in original.feature_spec.irreps)
                                        + sum(m for m, _ in original.irreps) + 2)
    reference = next(original.parameters())
    # Frozen energy copies are buffers, so requires_grad_(True) on the proposal
    # cannot accidentally optimize any physical energy coefficient.
    for name, value in (
        ("frozen_response_weight", energy.charge_affine.weight[:, energy.hidden_features:]),
        ("frozen_offset_weight", energy.charge_affine.weight[:, :energy.hidden_features]),
        ("frozen_offset_bias", energy.charge_affine.bias),
        ("frozen_output_weights", energy.positive_output_weights),
        ("frozen_driving_scale", energy.driving_scale),
        ("frozen_curvature_scale", energy.curvature_scale),
        ("frozen_hardness", energy.hardness),
    ):
        converted.register_buffer(name, value.detach().clone())
    converted.frozen_charge_value = _frozen_module(energy.charge_value)
    width = 1 + energy.charge_features + energy.hidden_features
    for name in ("center", "scale"):
        value = calibration[name].detach().clone().to(reference)
        if value.shape != (width,) or not bool(torch.isfinite(value).all()):
            raise ValueError("Calibration dimensions or values are invalid")
        if name == "scale" and not bool((value > 0).all()):
            raise ValueError("Calibration scales must be strictly positive")
        converted.register_buffer("physical_" + name, value)
    if variant != "baseline":
        scalar_indices = []
        for (multiplicity, irrep), section in zip(converted.irreps, converted.irreps.slices()):
            if irrep == o3.Irrep("0e"):
                scalar_indices.extend(range(section.start, section.stop))
        if not scalar_indices:
            raise ValueError("Early physical injection requires even scalar channels")
        embedding = reference.new_zeros(converted.irreps.dim, len(scalar_indices))
        embedding[scalar_indices, torch.arange(len(scalar_indices), device=reference.device)] = 1
        converted.register_buffer("scalar_embedding", embedding)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            converted.physical_input = nn.Linear(width, len(scalar_indices), bias=False,
                                                  device="cpu", dtype=torch.float64).to(reference)
        with torch.no_grad():
            converted.physical_input.weight.zero_()
    if variant.startswith("operator"):
        for name in ("center", "scale"):
            value = calibration["invariant_" + name].detach().clone().to(reference)
            if (value.shape != (converted.response_readout_width,)
                    or not bool(torch.isfinite(value).all())
                    or (name == "scale" and not bool((value > 0).all()))):
                raise ValueError("Operator invariant calibration is invalid")
            converted.register_buffer("invariant_" + name, value)
        # The warm charge readout is not part of the operator prediction; do
        # not retain dead trainable parameters or zero-gradient spline heads.
        del converted.readout
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            converted.operator_head = nn.Sequential(
                nn.Linear(converted.response_readout_width, 64, device="cpu", dtype=torch.float64),
                nn.SiLU(), nn.Linear(64, 1 + 2 * rank, device="cpu", dtype=torch.float64)).to(reference)
            if variant == "operator_nonlinear":
                converted.nonlinear_head = nn.Sequential(
                    nn.Linear(converted.response_readout_width, 16, device="cpu", dtype=torch.float64),
                    nn.SiLU(), nn.Linear(16, 1, bias=False, device="cpu", dtype=torch.float64)).to(reference)
        desired = math.sqrt(1 / float(energy.hardness.detach()) - OPERATOR_EPSILON)
        raw_s = desired - converted.operator_s_floor
        if raw_s <= 0:
            raise ValueError("Hardness is incompatible with the positive diagonal initialization")
        inverse_softplus = raw_s + math.log(-math.expm1(-raw_s))
        with torch.no_grad():
            last = converted.operator_head[-1]
            last.weight[0].zero_()
            last.bias[0].fill_(inverse_softplus)
            last.weight[1 + rank:].zero_()
            last.bias[1 + rank:].zero_()
            # U retains its nonzero deterministic random draw. V, rather than
            # both factors, starts at zero so the V output has a gradient.
            if variant == "operator_nonlinear":
                converted.nonlinear_head[-1].weight.zero_()
    converted.train(original.training)
    head.proposal = converted
    # Pure contraction scheduling: same frozen kernel and energy weights,
    # including the geometry/context stage before predict_components.
    energy.attention.contraction = "linear"
    model.requires_grad_(False)
    converted.requires_grad_(True)
    return converted


def architecture_metadata(proposal):
    variant = proposal.response_variant
    return {"variant": variant, "class": type(proposal).__name__,
            "rank": proposal.response_rank if variant.startswith("operator") else 0,
            "proposal_parameters": sum(p.numel() for p in proposal.parameters()),
            "readout_parameters": sum(p.numel() for p in proposal.readout.parameters())
                                  if hasattr(proposal, "readout") else 0,
            "spline_parameters": sum(p.numel() for n, p in proposal.named_parameters()
                                     if n.endswith("spline_weight")),
            "operator_parameters": sum(p.numel() for p in proposal.operator_head.parameters())
                                   if hasattr(proposal, "operator_head") else 0,
            "frozen_coefficient_buffer_elements": sum(v.numel() for n, v in proposal.named_buffers()
                                                       if n.startswith("frozen_")),
            "local_fusion_blocks": 2, "proposal_query_key_trainable": False,
            "initialization": "warm_checkpoint_function_preserved_up_to_roundoff"
                              if variant in ("baseline", "early")
                              else "M_initially_hardness_inverse_times_I_not_warm_charge_function",
            "operator_nonlinear_initial_match": "same_seed_and_rank_operator_branch_plus_zero_branch",
            "native_driving_cost_included": variant != "baseline",
            "constraint": "fixed_Q", "contract": proposal.checkpoint_contract()}
