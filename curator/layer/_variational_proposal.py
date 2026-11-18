"""Boundary-conditioned, equivariant proposals with interchangeable communication.

These modules predict a state; none runs an equilibrium solver. Equalities use
only their small row-space solve. The legacy VariationalStateHead is separate.
"""

import math

import torch
from torch import nn
from e3nn import o3

from ._ewald import apply_ewald


def _normal(context, values):
    """Constraint-normal action on [n_state, channels], without a dense projector."""
    matrix = context["constraint_matrix"]
    if not matrix.shape[0]:
        return torch.zeros_like(values)
    with torch.profiler.record_function("acvr.constraint_projection"):
        return matrix.T @ torch.linalg.solve(matrix @ matrix.T, matrix @ values)


def _invariants(values, irreps):
    parts = []
    for (channels, irrep), part in zip(irreps, irreps.slices()):
        block = values[:, part].reshape(values.shape[0], channels, irrep.dim)
        parts.append(block[..., 0] if irrep == o3.Irrep("0e") else block.square().mean(-1))
    return torch.cat(parts, dim=-1)


class EquivariantFeatureNorm(nn.Module):
    """Smooth per-atom scaling, grouping all copies of each orthogonal irrep.

    No batch statistics or affine parameters. The identity-like behavior at
    zero avoids small-variance amplification and supports higher derivatives.
    """

    def __init__(self, irreps, scale=1.0):
        super().__init__()
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("feature normalization scale must be positive and finite")
        self.irreps = o3.Irreps(irreps)
        self.scale = float(scale)
        self.groups = {}
        for (_, irrep), part in zip(self.irreps, self.irreps.slices()):
            self.groups.setdefault(irrep, []).append(part)

    def forward(self, values):
        factors = {}
        for irrep, parts in self.groups.items():
            group = torch.cat([values[..., part] for part in parts], dim=-1)
            factors[irrep] = torch.rsqrt(
                1 + group.square().mean(-1, keepdim=True) / self.scale ** 2)
        return torch.cat([values[..., part] * factors[irrep]
                          for (_, irrep), part in zip(self.irreps, self.irreps.slices())], dim=-1)


class TensorProductBranchGain(nn.Module):
    """Fixed nonnegative gain; introduces no parameters, buffers or RNG draws."""

    def __init__(self, gain=1.0):
        super().__init__()
        if not math.isfinite(gain) or gain < 0:
            raise ValueError("product gain must be nonnegative and finite")
        self.gain = float(gain)

    def forward(self, values):
        return values if self.gain == 1.0 else self.gain * values


class AttentionQueryKeyNorm(nn.Module):
    """Smooth normalization over the channel dimension of each atom/head."""

    def forward(self, values):
        return values * torch.rsqrt(1 + values.square().mean(-1, keepdim=True))


class EquivariantUpdate(nn.Module):
    """Channel-wise angular products, channel mixing, and invariant gates."""

    def __init__(self, irreps, product_gain=1.0, product_normalization="legacy",
                 product_norm_scale=1.0):
        super().__init__()
        if product_normalization not in {"legacy", "smooth_per_irrep"}:
            raise ValueError("product_normalization must be legacy or smooth_per_irrep")
        if not math.isfinite(product_norm_scale) or product_norm_scale <= 0:
            raise ValueError("product_norm_scale must be positive and finite")
        self.product_gain = TensorProductBranchGain(product_gain)
        self.product_norm = (EquivariantFeatureNorm(irreps, product_norm_scale)
                             if product_normalization == "smooth_per_irrep" else None)
        self.irreps = irreps
        instructions = [(i, j, k, "uuu", True)
                        for i, (_, left) in enumerate(irreps)
                        for j, (_, right) in enumerate(irreps)
                        for k, (_, result) in enumerate(irreps)
                        if result in left * right]
        self.product = o3.TensorProduct(irreps, irreps, irreps, instructions)
        self.mix = o3.Linear(irreps, irreps, biases=True)
        self.gates = nn.Linear(irreps.num_irreps, irreps.num_irreps)

    def forward(self, values):
        # Bounded invariant scaling keeps polynomial products well behaved even
        # for a backbone whose raw feature scale differs from the default one.
        normalization = getattr(self, "product_norm", None)
        if normalization is None:
            invariants = _invariants(values, self.irreps)
            scale = torch.rsqrt(1 + invariants.square().mean(-1, keepdim=True))
            product = self.product(values * scale, values * scale)
        else:
            scaled = normalization(values)
            product = self.product(scaled, scaled)
        gain = getattr(self, "product_gain", None)
        if gain is not None:
            product = gain(product)
        values = self.mix(values + product)
        gates = torch.sigmoid(self.gates(_invariants(values, self.irreps)))
        blocks, offset = [], 0
        for (channels, irrep), part in zip(self.irreps, self.irreps.slices()):
            block = values[:, part].reshape(values.shape[0], channels, irrep.dim)
            if irrep == o3.Irrep("0e"):
                block = nn.functional.silu(block)
            else:
                block = block * gates[:, offset:offset + channels, None]
            blocks.append(block.flatten(1))
            offset += channels
        return torch.cat(blocks, dim=-1)


class PointwiseCommunication(nn.Module):
    def forward(self, features, state, context):
        return torch.zeros_like(features)


class MessagePassingCommunication(nn.Module):
    """Finite-cutoff, directional messages; no pooling across disconnected nodes."""

    def __init__(self, irreps, radial_features=8):
        super().__init__()
        self.irreps = irreps
        self.harmonics = o3.Irreps.spherical_harmonics(max(ir.l for _, ir in irreps))
        instructions = [(i, j, k, "uvu", True)
                        for i, (_, source) in enumerate(irreps)
                        for j, (_, angular) in enumerate(self.harmonics)
                        for k, (_, result) in enumerate(irreps)
                        if result in source * angular]
        self.product = o3.TensorProduct(irreps, self.harmonics, irreps, instructions,
                                       internal_weights=False, shared_weights=False)
        self.register_buffer("centers", torch.linspace(0, 1, radial_features))
        self.radial = nn.Sequential(nn.Linear(radial_features, radial_features), nn.SiLU(),
                                    nn.Linear(radial_features, self.product.weight_numel))
        self.mix = o3.Linear(irreps, irreps)

    def edge_values(self, features, sources, differences, squared_distance, length):
        scaled_squared = squared_distance / length.square()
        radial = torch.exp(-8 * (scaled_squared[:, None] - self.centers).square())
        directions = differences / torch.sqrt(squared_distance[:, None] + 1e-12)
        angular = o3.spherical_harmonics(self.harmonics, directions, normalize=False,
                                         normalization="component")
        return self.product(features[sources], angular, self.radial(radial))

    def forward(self, features, state, context):
        edges = context["edge_index"]
        if not edges.numel():
            return torch.zeros_like(features)
        distance = context["edge_distance"]
        messages = self.edge_values(features, edges[:, 1], context["edge_diff"],
                                     distance.square(), context["cutoff"])
        # Six zero derivatives at the cutoff, including response/force training.
        envelope = (1 - (distance / context["cutoff"]).square()).clamp_min(0).pow(6)
        messages = messages * envelope[:, None]
        result = torch.zeros_like(features).index_add(0, edges[:, 0], messages)
        return self.mix(result / math.sqrt(16))


class AttentionCommunication(MessagePassingCommunication):
    """All-pairs scalar attention weights and equivariant directional values.

    Periodic edges use smooth sine/cosine lattice phases, not discontinuous
    minimum-image vectors. This is a learned geometric encoding, not an Ewald
    approximation. Explicit torch operations support higher-order derivatives.
    """

    def __init__(self, irreps, num_heads=4, normalization="legacy"):
        if normalization not in {"legacy", "smooth"}:
            raise ValueError("attention normalization must be legacy or smooth")
        super().__init__(irreps)
        self.query_key_norm = AttentionQueryKeyNorm() if normalization == "smooth" else None
        self.num_heads = num_heads
        self.head_channels = irreps[0].mul // num_heads
        self.query = nn.Linear(irreps.num_irreps, num_heads * self.head_channels)
        self.key = nn.Linear(irreps.num_irreps, num_heads * self.head_channels)
        self.distance_bias = nn.Sequential(nn.Linear(1, 8), nn.SiLU(), nn.Linear(8, num_heads))

    def forward(self, features, state, context):
        n_atoms = features.shape[0]
        positions = context["positions"]
        differences = positions[None, :, :] - positions[:, None, :]
        pbc = context["pbc"]
        if bool(pbc.any()):
            cell = context["cell"]
            fractional = torch.linalg.solve(cell.T, differences.reshape(-1, 3).T).T.reshape(n_atoms, n_atoms, 3)
            phase = 2 * math.pi * fractional
            periodic = torch.where(pbc, torch.sin(phase) / (2 * math.pi), fractional)
            differences = periodic @ cell
            # A positive smooth periodic scalar metric complements the odd
            # directional encoding, which vanishes at a half-cell displacement.
            lengths = cell.square().sum(-1)
            metric = torch.where(pbc, 2 * (1 - torch.cos(phase)) / (2 * math.pi) ** 2,
                                 fractional.square())
            squared_distance = (metric * lengths).sum(-1)
        else:
            squared_distance = differences.square().sum(-1)
        invariants = _invariants(features, self.irreps)
        query = self.query(invariants).reshape(n_atoms, self.num_heads, self.head_channels)
        key = self.key(invariants).reshape_as(query)
        normalization = getattr(self, "query_key_norm", None)
        if normalization is not None:
            query, key = normalization(query), normalization(key)
        # beta=sqrt(head_channels): preserve the legacy small-input scale.
        scores = torch.einsum("ihd,jhd->ijh", query, key) / math.sqrt(self.head_channels)
        scores = scores + self.distance_bias((squared_distance / context["cutoff"].square())[..., None])
        weights = torch.softmax(scores, dim=1)
        sources = torch.arange(n_atoms, device=features.device).repeat(n_atoms)
        messages = self.edge_values(features, sources, differences.reshape(-1, 3),
                                     squared_distance.reshape(-1), context["cutoff"])
        blocks = []
        for (channels, irrep), part in zip(self.irreps, self.irreps.slices()):
            values = messages[:, part].reshape(n_atoms, n_atoms, self.num_heads,
                                                channels // self.num_heads, irrep.dim)
            blocks.append((values * weights[..., None, None]).sum(1).flatten(1))
        return self.mix(torch.cat(blocks, dim=-1))


class EwaldCommunication(nn.Module):
    """Apply the energy's scalar charge kernel to a feasible trial charge."""

    def __init__(self, irreps, state_scale=1.0, field_scale=1.0):
        super().__init__()
        self.state_scale, self.field_scale = state_scale, field_scale
        self.embed = o3.Linear("2x0e", irreps, biases=True)

    def forward(self, features, state, context):
        if not bool(context["pbc"].all()) or not (
                "ewald_kernel" in context or "ewald_cos" in context):
            raise ValueError("Ewald proposal requires 3D periodic EwaldChargeEnergy context")
        with torch.profiler.record_function("acvr.ewald_apply"):
            potential = apply_ewald(context, state)
        return self.embed(torch.cat((state / self.state_scale, potential / self.field_scale), dim=-1))


class ProposalBlock(nn.Module):
    """Shared state/constraint/environment update followed by one communication."""

    def __init__(self, irreps, state_irrep, components, communication, num_heads,
                 state_scale=1.0, field_scale=1.0, product_gain=1.0,
                 product_normalization="legacy", product_norm_scale=1.0,
                 attention_normalization="legacy"):
        super().__init__()
        self.state_scale, self.field_scale = state_scale, field_scale
        self.channels = irreps[0].mul
        self.components = components
        self.state_readout = o3.Linear(irreps, o3.Irreps([(self.channels * (components if state_irrep.l == 0 else 1), state_irrep)]))
        boundary_irreps = o3.Irreps([(3 * self.channels * (components if state_irrep.l == 0 else 1), state_irrep)])
        self.boundary_embed = o3.Linear(boundary_irreps, irreps)
        self.update = EquivariantUpdate(irreps, product_gain, product_normalization, product_norm_scale)
        # Communication initialization must not change subsequent shared-block
        # initial weights when an experiment changes only the communication arm.
        with torch.random.fork_rng(devices=[]):
            if communication == "attention":
                self.communication = AttentionCommunication(irreps, num_heads, attention_normalization)
            elif communication == "mp":
                self.communication = MessagePassingCommunication(irreps)
            elif communication == "ewald":
                self.communication = EwaldCommunication(irreps, state_scale, field_scale)
            else:
                self.communication = PointwiseCommunication()

    def forward(self, features, context):
        n_atoms = features.shape[0]
        latent = self.state_readout(features).reshape(n_atoms, self.channels, self.components)
        flat = latent.transpose(1, 2).reshape(-1, self.channels)
        normal = _normal(context, flat)
        tangent = flat - normal
        matrix = context["environment_matrix"]
        environment = (self.state_scale / self.field_scale) * (
            matrix.T @ (context["environment_stiffness"] @ (matrix @ flat)))
        boundary = torch.cat([part.reshape(n_atoms, self.components, self.channels).transpose(1, 2).flatten(1)
                              for part in (tangent, normal, environment)], dim=-1)
        state = context["origin"] + self.state_scale * tangent[:, 0].reshape(n_atoms, self.components)
        messages = self.communication(features, state, context)
        return self.update(features + self.boundary_embed(boundary) + messages)


class BoundaryConditionedProposal(nn.Module):
    """Backbone-independent proposal; select pointwise, mp, attention, or ewald.

    Construction is lazy until the representation supplies its complete feature
    types. Every input irrep is retained, including higher angular orders. The
    output head applies the final hard projection just as for the MLP baseline.
    """

    def __init__(self, communication="pointwise", hidden_features=48, num_layers=2, num_heads=4,
                 state_scale=1.0, field_scale=1.0, use_physics_features=True,
                 product_gain=1.0, product_normalization="legacy", product_norm_scale=1.0,
                 attention_normalization="legacy"):
        super().__init__()
        self.use_physics_features = bool(use_physics_features)
        if not math.isfinite(product_gain) or product_gain < 0:
            raise ValueError("product_gain must be nonnegative and finite")
        if product_normalization not in {"legacy", "smooth_per_irrep"}:
            raise ValueError("product_normalization must be legacy or smooth_per_irrep")
        if not math.isfinite(product_norm_scale) or product_norm_scale <= 0:
            raise ValueError("product_norm_scale must be positive and finite")
        if attention_normalization not in {"legacy", "smooth"}:
            raise ValueError("attention_normalization must be legacy or smooth")
        if attention_normalization != "legacy" and communication != "attention":
            raise ValueError("attention normalization requires attention communication")
        self.product_gain = float(product_gain)
        self.product_normalization = product_normalization
        self.product_norm_scale = float(product_norm_scale)
        self.attention_normalization = attention_normalization
        if communication not in {"pointwise", "mp", "attention", "ewald"}:
            raise ValueError("communication must be pointwise, mp, attention or ewald")
        if hidden_features < 1 or num_layers < 1 or num_heads < 1:
            raise ValueError("proposal widths, layers and heads must be positive")
        if communication == "attention" and hidden_features % num_heads:
            raise ValueError("attention hidden_features must be divisible by num_heads")
        if not all(math.isfinite(value) and value > 0 for value in (state_scale, field_scale)):
            raise ValueError("fixed proposal state_scale and field_scale must be positive and finite")
        self.state_scale, self.field_scale = float(state_scale), float(field_scale)
        self.communication = communication
        self.hidden_features, self.num_layers, self.num_heads = hidden_features, num_layers, num_heads
        self.is_bound = False

    def bind(self, feature_spec, components, state_kind, cutoff, **unused):
        if self.is_bound:
            return
        if self.communication == "ewald" and (state_kind != "scalar" or components != 1):
            raise ValueError("Ewald communication supports only one scalar charge per atom")
        state_irrep = o3.Irrep("1o" if state_kind == "vector" else "0e")
        types = {ir for _, ir in feature_spec.irreps} | {o3.Irrep("0e"), state_irrep}
        # Directional modules can create vectors even from a scalar backbone.
        if self.communication in {"mp", "attention"}:
            types.add(o3.Irrep("1o"))
        self.irreps = o3.Irreps([(self.hidden_features, ir) for ir in sorted(types)])
        self.input = o3.Linear(feature_spec.irreps, self.irreps, biases=True)
        self.boundary_input = o3.Linear(o3.Irreps([(2 * (components if state_kind == "scalar" else 1), state_irrep)]), self.irreps)
        self.physics_input = o3.Linear("2x0e", self.irreps)
        self.blocks = nn.ModuleList(ProposalBlock(self.irreps, state_irrep, components,
                                                 self.communication, self.num_heads,
                                                 self.state_scale, self.field_scale,
                                                 getattr(self, "product_gain", 1.0),
                                                 getattr(self, "product_normalization", "legacy"),
                                                 getattr(self, "product_norm_scale", 1.0),
                                                 getattr(self, "attention_normalization", "legacy"))
                                    for _ in range(self.num_layers))
        self.readout = o3.Linear(self.irreps, o3.Irreps([(components if state_kind == "scalar" else 1, state_irrep)]))
        self.is_bound = True

    def checkpoint_contract(self):
        contract = {"class": type(self).__name__, "version": 1, "communication": self.communication,
                "hidden_features": self.hidden_features, "num_layers": self.num_layers,
                "num_heads": self.num_heads, "irreps": str(self.irreps),
                "state_scale": self.state_scale, "field_scale": self.field_scale}
        if not getattr(self, "use_physics_features", True):
            contract["use_physics_features"] = False
        stability = {
            "product_gain": getattr(self, "product_gain", 1.0),
            "product_normalization": getattr(self, "product_normalization", "legacy"),
            "product_norm_scale": getattr(self, "product_norm_scale", 1.0),
            "attention_normalization": getattr(self, "attention_normalization", "legacy"),
        }
        if stability != {"product_gain": 1.0, "product_normalization": "legacy",
                         "product_norm_scale": 1.0, "attention_normalization": "legacy"}:
            contract["version"] = 2
            contract["stability"] = stability
        return contract

    def forward(self, context):
        if not self.is_bound:
            raise RuntimeError("bind the proposal to a representation before use")
        features = self.input(context["features"])
        features = features + self.boundary_input(torch.cat((context["network_field"] / self.field_scale,
                                                             context["origin"] / self.state_scale), dim=-1))
        if getattr(self, "use_physics_features", True) and "electronegativity" in context and "hardness" in context:
            features = features + self.physics_input(torch.cat((context["electronegativity"] / self.field_scale,
                                      context["hardness"] * self.state_scale / self.field_scale), dim=-1))
        for block in self.blocks:
            features = block(features, context)
        return self.state_scale * self.readout(features)


__all__ = ["BoundaryConditionedProposal", "PointwiseCommunication", "MessagePassingCommunication",
           "AttentionCommunication", "EwaldCommunication", "EquivariantFeatureNorm",
           "TensorProductBranchGain", "AttentionQueryKeyNorm"]
