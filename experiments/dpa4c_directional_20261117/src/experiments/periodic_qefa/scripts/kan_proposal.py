"""Function-preserving spline KAN readouts for the native C proposal.

Only scalar invariants receive scalar nonlinear functions. The original C
communications, equivariant value maps and feasible charge projection remain
unchanged. Cubic B-splines use a fixed grid and training-only frozen affine
normalization followed by tanh; no solver or adaptive grid runs at inference.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from architectures import ArchitectureChargeProposal
from cache import output_head


VARIANTS = ("c", "kan_output", "kan_readout", "mlp_wide")
GRID_SIZE = 5
SPLINE_ORDER = 3
WIDE_WIDTH = 576


def bspline_basis(x, knots, order=SPLINE_ORDER):
    """Return [*, input_channels, grid_size + order] cardinal B-splines.

    Extended, equally spaced knots provide a partition of unity throughout
    [-1, 1], including both endpoints. The recurrence is C2 for cubic splines;
    interval membership itself requires no derivative.
    """
    x = x.unsqueeze(-1)
    bases = ((x >= knots[:-1]) & (x < knots[1:])).to(x.dtype)
    for degree in range(1, order + 1):
        left = (x - knots[:-(degree + 1)]) / (knots[degree:-1] - knots[:-(degree + 1)])
        right = (knots[degree + 1:] - x) / (knots[degree + 1:] - knots[1:-degree])
        bases = left * bases[..., :-1] + right * bases[..., 1:]
    return bases


class SplineEdgeLayer(nn.Module):
    """Existing affine/base-activation edges plus independent cubic splines."""

    def __init__(self, base, calibration, activation):
        super().__init__()
        if activation not in ("identity", "silu"):
            raise ValueError("Unsupported base activation")
        self.base = copy.deepcopy(base)
        self.activation = activation
        self.register_buffer("center", calibration["center"].detach().clone().to(base.weight))
        self.register_buffer("scale", calibration["scale"].detach().clone().to(base.weight))
        if self.center.shape != (base.in_features,) or self.scale.shape != self.center.shape:
            raise ValueError("Calibration does not match input width")
        if not bool(torch.isfinite(self.center).all() and torch.isfinite(self.scale).all()
                    and (self.scale > 0).all()):
            raise ValueError("Calibration must be finite with strictly positive scales")
        knots = torch.arange(-SPLINE_ORDER, GRID_SIZE + SPLINE_ORDER + 1,
                             device=base.weight.device, dtype=base.weight.dtype)
        self.register_buffer("knots", knots * (2.0 / GRID_SIZE) - 1.0)
        # Keep the tensor in [output, input, basis] form for SOAP preconditioning.
        self.spline_weight = nn.Parameter(base.weight.new_zeros(
            base.out_features, base.in_features, GRID_SIZE + SPLINE_ORDER))

    def bounded_input(self, x):
        return torch.tanh((x - self.center) / self.scale)

    def forward(self, x):
        base_input = F.silu(x) if self.activation == "silu" else x
        bases = bspline_basis(self.bounded_input(x), self.knots)
        return self.base(base_input) + F.linear(bases.flatten(-2), self.spline_weight.flatten(1))


class KANReadout(nn.Module):
    """338 -> 64 -> 2 map, initially identical to the trained MLP readout."""

    def __init__(self, readout, calibration, both_layers):
        super().__init__()
        self.first = (SplineEdgeLayer(readout[0], calibration["input"], "identity")
                      if both_layers else copy.deepcopy(readout[0]))
        self.last = SplineEdgeLayer(readout[2], calibration["hidden"], "silu")
        self.both_layers = bool(both_layers)

    def forward(self, x):
        return self.last(self.first(x))


class KANChargeProposal(ArchitectureChargeProposal):
    """C's native implementation with an explicitly versioned readout contract."""

    def checkpoint_contract(self):
        contract = super().checkpoint_contract()
        if not hasattr(self, "kan_variant"):
            return contract
        return {**contract, "schema": "shared_geometry_charge_proposal_kan_v1",
                "kan_variant": self.kan_variant,
                "kan_grid_size": GRID_SIZE, "kan_spline_order": SPLINE_ORDER,
                "kan_input_map": "tanh((x-training_mean)/max(training_std,scale_floor))",
                "kan_grid_update": "none",
                "readout_widths": [338, WIDE_WIDTH if self.kan_variant == "mlp_wide" else 64, 2],
                "first_layer_base": "identity",
                "last_layer_base": "silu"}


@torch.no_grad()
def collect_calibration(proposal, batches, scale_floor=1e-3):
    """Stream moments of the initial C TRAIN readout inputs/preactivations.

    ``batches`` must contain physical inputs only. No target charges, residuals,
    validation structures, or optimized states enter these statistics.
    """
    if type(proposal) is not ArchitectureChargeProposal or proposal.variant != "C":
        raise TypeError("Calibrate the unchanged C proposal before installing a variant")
    if not math.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale_floor must be finite and positive")
    moments = {}

    def update(name, values):
        values = values.detach().reshape(-1, values.shape[-1])
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Nonfinite training features in calibration")
        count = len(values)
        mean = values.mean(0)
        m2 = (values - mean).square().sum(0)
        if name not in moments:
            moments[name] = [count, mean, m2]
        else:
            old_count, old_mean, old_m2 = moments[name]
            delta = mean - old_mean
            total = old_count + count
            moments[name] = [total, old_mean + delta * count / total,
                             old_m2 + m2 + delta.square() * old_count * count / total]

    def observe(module, args):
        values = args[0]
        update("input", values)
        update("hidden", module[0](values))

    hook = proposal.readout.register_forward_pre_hook(observe)
    batch_count = 0
    allowed = {"features", "condition", "potential", "kernel", "offset",
               "charge_value", "geometry_energy", "total_charge", "source_index"}
    try:
        for batch in batches:
            if set(batch) != allowed:
                raise ValueError("Calibration accepts only the physical-input whitelist")
            proposal.cached_charges(batch)
            batch_count += 1
    finally:
        hook.remove()
    if not batch_count:
        raise ValueError("At least one training batch is required for calibration")
    result = {"source": "initial_C_training_physical_features_only",
              "scale_floor": float(scale_floor), "batches": batch_count}
    for name, (count, mean, m2) in moments.items():
        std = (m2 / count).sqrt()
        result[name] = {"center": mean.detach().clone(),
                        "scale": std.clamp_min(scale_floor).detach().clone(),
                        "std": std.detach().clone(), "atom_count": count,
                        "floored_channels": int((std < scale_floor).sum())}
    return result


def install_variant(model, name, calibration=None, seed=20261027):
    """Install once into a copied C model; train every proposal parameter only."""
    if name not in VARIANTS:
        raise ValueError(f"name must be one of {VARIANTS}")
    head = output_head(model)
    original = head.proposal
    if type(original) is not ArchitectureChargeProposal or original.variant != "C":
        raise TypeError("Expected the original C proposal; install a variant only once")
    if original.readout[0].in_features != 338 or original.readout[0].out_features != 64:
        raise ValueError("This experiment requires the current 338 -> 64 -> 2 readout")
    if name == "c":
        model.requires_grad_(False)
        original.requires_grad_(True)
        return original
    converted = copy.deepcopy(original)
    # All native methods are inherited unchanged. Python class reassignment on
    # this copied Module avoids reconstructing e3nn code-generated submodules.
    converted.__class__ = KANChargeProposal
    converted.kan_variant = name
    if name in ("kan_output", "kan_readout"):
        if calibration is None:
            raise ValueError("Spline variants require training-only calibration")
        converted.readout = KANReadout(original.readout, calibration, name == "kan_readout")
    else:
        # Explicit CPU initialization isolates the random stream across GPUs.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            wide = nn.Sequential(nn.Linear(338, WIDE_WIDTH, device="cpu", dtype=torch.float64),
                                 nn.SiLU(), nn.Linear(WIDE_WIDTH, 2, device="cpu", dtype=torch.float64))
        wide = wide.to(original.readout[0].weight)
        with torch.no_grad():
            wide[0].weight[:64].copy_(original.readout[0].weight)
            wide[0].bias[:64].copy_(original.readout[0].bias)
            wide[2].weight.zero_()
            wide[2].weight[:, :64].copy_(original.readout[2].weight)
            wide[2].bias.copy_(original.readout[2].bias)
        converted.readout = wide
    converted.train(original.training)
    head.proposal = converted
    model.requires_grad_(False)
    converted.requires_grad_(True)
    return converted


def architecture_metadata(proposal):
    name = getattr(proposal, "kan_variant", "c")
    return {"variant": name,
            "proposal_parameters": sum(p.numel() for p in proposal.parameters()),
            "readout_parameters": sum(p.numel() for p in proposal.readout.parameters()),
            "spline_parameters": sum(p.numel() for n, p in proposal.named_parameters()
                                     if n.endswith("spline_weight")),
            "fixed_normalization": name in ("kan_output", "kan_readout"),
            "initialization": "exact_C_function_up_to_roundoff",
            "inference": "one_native_C_forward_fixed_hidden_communications_K_then_K_transpose",
            "contract": proposal.checkpoint_contract()}
