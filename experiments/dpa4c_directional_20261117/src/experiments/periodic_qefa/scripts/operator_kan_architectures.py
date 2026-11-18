"""Function-preserving capacity controls for the structured response head.

All original Parameter objects survive installation, so their complete SOAP
states can be migrated by name. Only scalar invariants enter these heads; the
two communications, response factors, fixed-Q projection and rank are unchanged.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from cache import output_head
from kan_proposal import bspline_basis
from response_architectures import PHYSICAL_KEYS, ResponseChargeProposal


VARIANTS = ("baseline", "kan5", "matched_mlp")
SPLINE_ORDER = 3
GRID_SIZE = 2
BASIS_COUNT = GRID_SIZE + SPLINE_ORDER


class OperatorSplineLayer(nn.Module):
    """Keep the affine/SiLU path and add initially zero cubic spline edges."""

    def __init__(self, base, calibration, activation):
        super().__init__()
        if activation not in ("identity", "silu"):
            raise ValueError("Expected identity or SiLU base activation")
        self.base = base
        self.activation = activation
        for name in ("center", "scale"):
            value = calibration[name].detach().clone().to(base.weight)
            if value.shape != (base.in_features,) or not bool(torch.isfinite(value).all()):
                raise ValueError("Calibration must be finite and match the input width")
            if name == "scale" and not bool((value > 0).all()):
                raise ValueError("Spline normalization scale must be positive")
            self.register_buffer(name, value)
        knots = torch.arange(-SPLINE_ORDER, GRID_SIZE + SPLINE_ORDER + 1,
                             dtype=base.weight.dtype, device=base.weight.device)
        self.register_buffer("knots", knots * (2. / GRID_SIZE) - 1.)
        self.spline_weight = nn.Parameter(base.weight.new_zeros(
            base.out_features, base.in_features, BASIS_COUNT))

    def forward(self, values):
        base_input = F.silu(values) if self.activation == "silu" else values
        bounded = torch.tanh((values - self.center) / self.scale)
        basis = bspline_basis(bounded, self.knots, SPLINE_ORDER)
        return self.base(base_input) + F.linear(basis.flatten(-2), self.spline_weight.flatten(1))


class OperatorKANHead(nn.Module):
    def __init__(self, original, calibration):
        super().__init__()
        self.first = OperatorSplineLayer(original[0], calibration["input"], "identity")
        self.last = OperatorSplineLayer(original[2], calibration["hidden"], "silu")

    def forward(self, values):
        return self.last(self.first(values))


class ExpandedOperatorMLP(nn.Module):
    """A wider one-hidden-layer MLP, partitioned to retain old SOAP coordinates.

The added hidden units have zero outgoing weights. This is algebraically the
usual concatenated wide MLP, with one shared output bias in the original path.
"""

    def __init__(self, original, extra_width, seed):
        super().__init__()
        self.base = original
        self.extra_width = int(extra_width)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.extra_input = nn.Linear(original[0].in_features, extra_width,
                                         dtype=torch.float64, device="cpu")
            self.extra_output = nn.Linear(extra_width, original[2].out_features,
                                          bias=False, dtype=torch.float64, device="cpu")
        self.extra_input.to(original[0].weight)
        self.extra_output.to(original[0].weight)
        with torch.no_grad():
            self.extra_output.weight.zero_()

    def forward(self, values):
        return self.base(values) + self.extra_output(F.silu(self.extra_input(values)))


class OperatorCapacityProposal(ResponseChargeProposal):
    def checkpoint_contract(self):
        contract = super().checkpoint_contract()
        if not hasattr(self, "operator_capacity_variant"):
            return contract
        variant = self.operator_capacity_variant
        return {**contract, "schema": "structured_response_operator_capacity_v1",
                "operator_capacity_variant": variant,
                "readout_widths": [self.response_readout_width, self.operator_capacity_width,
                                   1 + 2 * self.response_rank],
                "kan_layers": ["first", "last"] if variant == "kan5" else [],
                "kan_grid_size": GRID_SIZE if variant == "kan5" else None,
                "kan_spline_order": SPLINE_ORDER if variant == "kan5" else None,
                "kan_basis_count": BASIS_COUNT if variant == "kan5" else 0,
                "kan_input_map": "tanh((x-training_mean)/max(training_std,scale_floor))"
                                 if variant == "kan5" else None,
                "kan_scale_floor": self.operator_spline_scale_floor if variant == "kan5" else None,
                "kan_grid_update": "none",
                "initialization": "original_parameter_objects_plus_zero_output_increment"}


@torch.no_grad()
def collect_operator_calibration(model, batches, scale_floor=1e-3):
    """Stream initial head input/preactivation moments from training inputs only.

Normalization is applied only to the new spline path. Existing invariant
normalization and the old linear/SiLU path are left untouched.
"""
    if not math.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale_floor must be finite and positive")
    proposal = output_head(model).proposal
    if not isinstance(proposal.operator_head, nn.Sequential):
        raise TypeError("Calibrate the original response MLP before installing a capacity head")
    moments, source_indices = {}, []

    def update(name, values):
        values = values.detach().reshape(-1, values.shape[-1])
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Nonfinite training values in spline calibration")
        count, mean = len(values), values.mean(0)
        m2 = (values - mean).square().sum(0)
        if name in moments:
            old_count, old_mean, old_m2 = moments[name]
            total, delta = old_count + count, mean - old_mean
            moments[name] = (total, old_mean + delta * count / total,
                             old_m2 + m2 + delta.square() * old_count * count / total)
        else:
            moments[name] = (count, mean, m2)

    def observe(module, args):
        update("input", args[0])
        update("hidden", module[0](args[0]))

    hook = proposal.operator_head.register_forward_pre_hook(observe)
    try:
        for batch in batches:
            if set(batch) != PHYSICAL_KEYS:
                raise ValueError("Calibration requires only the exact physical input fields")
            source_indices.extend(int(v) for v in batch["source_index"].reshape(-1).cpu())
            proposal.cached_charges(batch)
    finally:
        hook.remove()
    if not moments or len(source_indices) != len(set(source_indices)):
        raise ValueError("Calibration requires nonempty, unique training structures")
    result = {"source": "current_parent_training_physical_inputs_only",
              "scale_floor": float(scale_floor), "source_indices": source_indices,
              "structure_count": len(source_indices)}
    for name, (count, mean, m2) in moments.items():
        std = (m2 / count).sqrt()
        result[name] = {"center": mean.cpu(), "scale": std.clamp_min(scale_floor).cpu(),
                        "std": std.cpu(), "atom_count": count,
                        "floored_channels": int((std < scale_floor).sum())}
    return result


def install_operator_variant(model, variant, calibration=None, seed=3407):
    """Install once; return exact old-to-new parameter names and audit metadata."""
    if variant not in VARIANTS:
        raise ValueError(f"Expected one of {VARIANTS}")
    proposal = output_head(model).proposal
    if (not isinstance(proposal, ResponseChargeProposal)
            or proposal.response_variant != "operator"
            or hasattr(proposal, "operator_capacity_variant")):
        raise TypeError("Expected an unmodified structured response operator proposal")
    head = proposal.operator_head
    if (not isinstance(head, nn.Sequential) or len(head) != 3
            or not isinstance(head[0], nn.Linear) or not isinstance(head[1], nn.SiLU)
            or not isinstance(head[2], nn.Linear)):
        raise TypeError("Expected the original Linear-SiLU-Linear response head")
    if (head[0].in_features, head[0].out_features, head[2].out_features) != (
            proposal.response_readout_width, 64, 33):
        raise ValueError("The declared experiment requires the bound invariant width, hidden64 and rank16")
    original = dict(proposal.named_parameters())
    original_values = {name: value.detach().clone() for name, value in original.items()}
    added_target = BASIS_COUNT * (head[0].weight.numel() + head[2].weight.numel())
    per_hidden = head[0].in_features + 1 + head[2].out_features
    extra_width = round(added_target / per_hidden)
    if variant != "baseline":
        if variant == "kan5":
            if calibration is None:
                raise ValueError("KAN requires training-only spline calibration")
            proposal.operator_head = OperatorKANHead(head, calibration)
            proposal.operator_spline_scale_floor = calibration.get("scale_floor", 1e-3)
        else:
            proposal.operator_head = ExpandedOperatorMLP(head, extra_width, seed)
        proposal.__class__ = OperatorCapacityProposal
        proposal.operator_capacity_variant = variant
        proposal.operator_capacity_width = 64 + extra_width if variant == "matched_mlp" else 64
        proposal.operator_head.train(head.training)
    current = dict(proposal.named_parameters())
    names_by_id = {id(parameter): name for name, parameter in current.items()}
    mapping = {}
    for name, parameter in original.items():
        if id(parameter) not in names_by_id or not torch.equal(parameter, original_values[name]):
            raise AssertionError("Installation changed an original parameter: " + name)
        mapping[name] = names_by_id[id(parameter)]
    old_ids = {id(parameter) for parameter in original.values()}
    new_names = [name for name, parameter in current.items() if id(parameter) not in old_ids]
    model.requires_grad_(False)
    proposal.requires_grad_(True)
    return {"variant": variant, "seed": int(seed),
            "old_to_new_parameter_names": mapping, "new_parameter_names": new_names,
            "original_parameters": sum(p.numel() for p in original.values()),
            "proposal_parameters": sum(p.numel() for p in current.values()),
            "added_parameters": sum(current[name].numel() for name in new_names),
            "spline_parameters": sum(p.numel() for name, p in current.items()
                                     if name.endswith("spline_weight")),
            "kan_added_parameter_target": added_target,
            "matched_added_parameter_relative_difference":
                (extra_width * per_hidden - added_target) / added_target if variant == "matched_mlp" else None,
            "original_parameter_objects_preserved": True,
            "initialization": "unchanged_parent_function_up_to_roundoff",
            "inference": "one_native_forward_fixed_rank_no_solve",
            "contract": proposal.checkpoint_contract()}
