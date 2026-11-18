"""Equivariant global processing of finite-system variational states."""
import math
import torch
from e3nn import o3
from torch import nn
from torch.nn import functional as F

STATE_IRREPS = o3.Irreps("1x0e + 1x1o")
HARMONIC_IRREPS = o3.Irreps.spherical_harmonics(2)


def prepare_state_geometry(context):
    """Cache parameter-free geometry without detaching coordinate derivatives."""
    if bool(context["pbc"].any()):
        raise ValueError("global multipole modules currently support finite, nonperiodic systems only")
    if "_state_squared_distance" in context and "_state_harmonics" in context:
        return context
    positions = context["positions"]
    if positions.ndim != 2 or positions.shape[-1] != 3 or not positions.shape[0]:
        raise ValueError("positions must have shape [n_atoms, 3] with n_atoms > 0")
    difference = positions[None, :, :] - positions[:, None, :]
    squared = difference.square().sum(-1)
    # Self pairs have finite scalar and zero higher harmonics.
    directions = difference / torch.sqrt(squared[..., None] + 1e-12)
    angular = o3.spherical_harmonics(HARMONIC_IRREPS, directions,
                                     normalize=False, normalization="component")
    return {**context, "_state_squared_distance": squared, "_state_harmonics": angular}


def invariant_features(values, irreps):
    parts = []
    for (mul, ir), part in zip(irreps, irreps.slices()):
        block = values[..., part].reshape(*values.shape[:-1], mul, ir.dim)
        parts.append(block[..., 0] if ir == o3.Irrep("0e") else block.square().mean(-1))
    return torch.cat(parts, dim=-1)


def smooth_irrep_scale(values, irreps):
    """Scale only a nonlinear branch; the residual preserves raw amplitude."""
    parts = []
    for (mul, ir), part in zip(irreps, irreps.slices()):
        block = values[..., part].reshape(*values.shape[:-1], mul, ir.dim)
        scale = torch.rsqrt(1 + block.square().mean(dim=(-2, -1), keepdim=True))
        parts.append((block * scale).flatten(-2))
    return torch.cat(parts, dim=-1)


class EquivariantStateGate(nn.Module):
    """Smooth scalar activations and invariant gates for tensor channels."""
    def __init__(self, irreps):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.gates = nn.Linear(self.irreps.num_irreps, self.irreps.num_irreps)

    def forward(self, values):
        gates = torch.sigmoid(self.gates(invariant_features(values, self.irreps)))
        output, offset = [], 0
        for (mul, ir), part in zip(self.irreps, self.irreps.slices()):
            block = values[..., part].reshape(*values.shape[:-1], mul, ir.dim)
            block = F.silu(block) if ir == o3.Irrep("0e") else block * gates[..., offset:offset + mul, None]
            output.append(block.flatten(-2))
            offset += mul
        return torch.cat(output, dim=-1)


class GlobalStateBlock(nn.Module):
    """Geometry attention, depthwise angular tensor products, channel mixing.

    All ordered atom pairs participate, without a neighbor cutoff or prescribed
    Coulomb law. There is no guarantee of asymptotic fragment size consistency.
    Depthwise TP paths avoid quadratic channel scaling in per-edge weights.
    """
    def __init__(self, irreps, num_heads=4, radial_features=8, max_length=32.0):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        if not self.irreps or num_heads < 1 or radial_features < 2 or max_length <= 0.5:
            raise ValueError("positive irreps/heads, radial_features >= 2, max_length > 0.5 required")
        if any(mul % num_heads for mul, _ in self.irreps):
            raise ValueError("each hidden multiplicity must be divisible by num_heads")
        allowed = (o3.Irrep("0e"), o3.Irrep("1o"), o3.Irrep("2e"))
        if any(ir not in allowed for _, ir in self.irreps):
            raise ValueError("hidden irreps must be 0e, 1o or 2e")
        scalar_channels = sum(mul for mul, ir in self.irreps if ir == o3.Irrep("0e"))
        if not scalar_channels:
            raise ValueError("invariant hidden channels are required")
        self.num_heads = int(num_heads)
        self.head_channels = scalar_channels // num_heads
        self.geometry_scalars = o3.Linear(self.irreps, f"{scalar_channels}x0e")
        self.query = nn.Linear(scalar_channels, scalar_channels, bias=False)
        self.key = nn.Linear(scalar_channels, scalar_channels, bias=False)
        self.distance_bias = nn.Sequential(nn.Linear(radial_features, 16), nn.SiLU(), nn.Linear(16, num_heads))
        self.register_buffer("radial_lengths", torch.logspace(math.log10(0.5), math.log10(max_length), radial_features))
        paths, instructions = [], []
        target_types = {ir for _, ir in self.irreps}
        for source, (mul, source_ir) in enumerate(self.irreps):
            for angular, (_, angular_ir) in enumerate(HARMONIC_IRREPS):
                for target_ir in source_ir * angular_ir:
                    if target_ir in target_types:
                        destination = len(paths)
                        paths.append((mul, target_ir))
                        instructions.append((source, angular, destination, "uvu", True))
        self.path_irreps = o3.Irreps(paths)
        self.product = o3.TensorProduct(self.irreps, HARMONIC_IRREPS, self.path_irreps, instructions,
                                       internal_weights=False, shared_weights=False)
        self.radial = nn.Sequential(nn.Linear(radial_features, 32), nn.SiLU(),
                                    nn.Linear(32, self.product.weight_numel))
        self.message_mix = o3.Linear(self.path_irreps, self.irreps)
        self.update_mix = o3.Linear(self.irreps, self.irreps, biases=True)
        self.gate = EquivariantStateGate(self.irreps)

    def forward(self, values, geometry_features, context):
        n = values.shape[0]
        radial = torch.exp(-context["_state_squared_distance"][..., None] / self.radial_lengths.square())
        scalar = self.geometry_scalars(geometry_features)
        query = self.query(scalar).reshape(n, self.num_heads, self.head_channels)
        key = self.key(scalar).reshape_as(query)
        query = query * torch.rsqrt(1 + query.square().mean(-1, keepdim=True))
        key = key * torch.rsqrt(1 + key.square().mean(-1, keepdim=True))
        score = torch.einsum("ihd,jhd->ijh", query, key) / math.sqrt(self.head_channels)
        attention = torch.softmax(score + self.distance_bias(radial), dim=1)
        source = values[None, :, :].expand(n, -1, -1).reshape(-1, values.shape[-1])
        messages = self.product(source, context["_state_harmonics"].reshape(-1, HARMONIC_IRREPS.dim),
                                self.radial(radial).reshape(n * n, -1))
        pooled = []
        for (mul, ir), part in zip(self.path_irreps, self.path_irreps.slices()):
            block = messages[:, part].reshape(n, n, self.num_heads, mul // self.num_heads, ir.dim)
            pooled.append((block * attention[..., None, None]).sum(1).flatten(1))
        message = self.message_mix(torch.cat(pooled, dim=-1))
        update = self.update_mix(values + message + geometry_features)
        update = self.gate(smooth_irrep_scale(update, self.irreps))
        return values + update / math.sqrt(2.0)


class GlobalStateNetwork(nn.Module):
    """Shared feature adapter and state stack; no energy or boundary policy."""
    def __init__(self, hidden_irreps="64x0e + 16x1o + 8x2e", num_layers=2, num_heads=4,
                 radial_features=8, max_length=32.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.irreps = o3.Irreps(hidden_irreps)
        self.num_layers, self.num_heads = int(num_layers), int(num_heads)
        self.radial_features, self.max_length = int(radial_features), float(max_length)
        self.input_irreps = None
        self.geometry_input = None
        self.state_input = o3.Linear(STATE_IRREPS, self.irreps)
        self.blocks = nn.ModuleList(GlobalStateBlock(self.irreps, num_heads, radial_features, max_length)
                                    for _ in range(num_layers))

    def bind(self, feature_spec):
        irreps = o3.Irreps(feature_spec.irreps)
        if self.input_irreps is not None:
            if irreps != self.input_irreps:
                raise ValueError("state network already bound to different feature irreps")
            return self
        self.input_irreps = irreps
        self.geometry_input = o3.Linear(irreps, self.irreps, biases=True).to(self.state_input.weight)
        return self

    def geometry_features(self, context):
        if self.geometry_input is None:
            raise RuntimeError("bind the global state network before use")
        return smooth_irrep_scale(self.geometry_input(context["features"]), self.irreps)

    def forward(self, context, state, extra=None):
        context = prepare_state_geometry(context)
        geometry = self.geometry_features(context)
        values = geometry + self.state_input(state)
        if extra is not None:
            values = values + extra
        for block in self.blocks:
            values = block(values, geometry, context)
        return values


__all__ = ["STATE_IRREPS", "GlobalStateBlock", "GlobalStateNetwork", "prepare_state_geometry"]
