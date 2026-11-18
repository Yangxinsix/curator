"""Controlled, single-forward proposal extensions of the frozen shared model.

All variants start at the exact original proposal function (up to rounding).
A enlarges only the invariant readout. B/C add one residual hidden-feature
block using K/K.T. Neither block predicts charges or evaluates a residual.
The native contraction has the same linear large-N path as the energy; dense
kernels are used only by the existing small-system training cache.
"""
import copy

import torch
from torch import nn

from curator.layer.efa.readout import invariant_features
from curator.layer.efa.reciprocal import NUFFTPreparedAttention
from curator.layer.efa.shared import SharedGeometryChargeProposal
from curator.layer.efa.state import EquivariantStateFusion


VARIANTS = ("baseline", "A", "B", "C")


def aggregate_prepared(prepared, values, transpose=False, contraction="auto",
                       pairwise_max_atoms=256):
    """Apply the native EFA kernel or its transpose without storing the kernel."""
    if isinstance(prepared, NUFFTPreparedAttention) or getattr(prepared, "is_molecular_factorization", False):
        return prepared.apply(values, transpose=transpose)
    query, key, weights = prepared
    if transpose:
        query, key = key, query
    n, directions, channels = query.shape
    if contraction == "auto":
        width = values.shape[-1]
        pairwise = (n <= pairwise_max_atoms and
                    n * (directions * channels + width)
                    < 2 * directions * channels * width)
    elif contraction in ("linear", "pairwise"):
        pairwise = contraction == "pairwise"
    else:
        raise ValueError("Unknown EFA contraction: " + contraction)
    if pairwise:
        if n > pairwise_max_atoms:
            raise ValueError("pairwise atom count exceeds pairwise_max_atoms")
        kernel = (query * weights[None, :, None]).flatten(1) @ key.flatten(1).T
        return kernel @ values
    projected = torch.einsum("nmd,nv->mdv", key, values)
    return torch.einsum("nmd,mdv,m->nv", query, projected, weights)


class ArchitectureChargeProposal(SharedGeometryChargeProposal):
    """Native proposal whose only output state is its final feasible charge."""

    @classmethod
    def from_shared(cls, proposal, variant, seed=20261026,
                    contraction="auto", pairwise_max_atoms=256):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if type(proposal) is not SharedGeometryChargeProposal:
            raise TypeError("Install once into the unmodified shared proposal")
        invariant_width = (sum(multiplicity for multiplicity, _ in proposal.feature_spec.irreps)
                           + sum(multiplicity for multiplicity, _ in proposal.irreps) + 2)
        if proposal.readout_features != 64 or proposal.readout[0].in_features != invariant_width:
            raise ValueError("A/B/C require the bound invariant features and a 64-wide hidden readout")
        # Preserve the checkpoint's trained parameters, dtype, and device. Only
        # the newly introduced parameters consume this isolated random stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            converted = cls(
                feature_spec=copy.deepcopy(proposal.feature_spec),
                hidden_irreps=str(proposal.irreps), readout_features=64,
                charge_scale=proposal.charge_scale,
                potential_scale=proposal.potential_scale,
            ).to(proposal._dtype_anchor)
            converted.load_state_dict(proposal.state_dict())
            converted.variant = variant
            converted.attention_contraction = contraction
            converted.pairwise_max_atoms = pairwise_max_atoms
            if variant == "A":
                converted.readout = nn.Sequential(
                    nn.Linear(invariant_width, 128), nn.SiLU(), nn.Linear(128, 128),
                    nn.SiLU(), nn.Linear(128, 2),
                ).to(proposal._dtype_anchor)
                with torch.no_grad():
                    first, second, last = (converted.readout[i] for i in (0, 2, 4))
                    first.weight[:64].copy_(proposal.readout[0].weight)
                    first.bias[:64].copy_(proposal.readout[0].bias)
                    second.weight.zero_()
                    second.bias.zero_()
                    identity = torch.eye(64, device=second.weight.device,
                                         dtype=second.weight.dtype)
                    second.weight[:64, :64].copy_(identity)
                    second.weight[64:, :64].copy_(-identity)
                    last.weight[:, :64].copy_(proposal.readout[2].weight)
                    last.weight[:, 64:].copy_(-proposal.readout[2].weight)
                    last.bias.copy_(proposal.readout[2].bias)
                # SiLU(x)-SiLU(-x)=x, so this deeper readout is exactly the old
                # function. Random extra first-layer features are initially
                # disconnected; their second-layer columns get gradients.
            elif variant in ("B", "C"):
                converted.extra_fusion = EquivariantStateFusion(
                    proposal.irreps, proposal.readout_features,
                ).to(proposal._dtype_anchor)
                with torch.no_grad():
                    for parameter in converted.extra_fusion.mix.parameters():
                        parameter.zero_()
                # The fusion already returns local + gated_update. Zero mix
                # makes it identity without introducing an additional skip.
        converted.train(proposal.training)
        return converted

    def checkpoint_contract(self):
        contract = super().checkpoint_contract()
        # Construction first loads the unchanged checkpoint, before variant
        # modules are installed. Afterwards state_dict enforces this contract.
        if not hasattr(self, "variant"):
            return contract
        return {
            **contract, "schema": "shared_geometry_charge_proposal_abc_v1",
            "variant": self.variant,
            "readout_widths": ([self.readout[0].in_features, 128, 128, 2] if self.variant == "A"
                               else [self.readout[0].in_features, 64, 2]),
            "hidden_communications": (["K", "K.T"] if self.variant == "C"
                                      else ["K", "K"] if self.variant == "B"
                                      else ["K"]),
            "inference": "one_fixed_forward_final_charge_only",
        }

    def _readout_features(self, features, values, condition):
        return self.readout(torch.cat((
            invariant_features(features, self.feature_spec.irreps),
            invariant_features(values, self.irreps), condition,
        ), dim=-1))

    def predict_components(self, context):
        cache = context["_shared_efa"]
        values = self.fusion(cache["local"], cache["global"])
        if self.variant in ("B", "C"):
            global_values = aggregate_prepared(
                cache["prepared"], self.norm(values),
                transpose=self.variant == "C",
                contraction=self.attention_contraction,
                pairwise_max_atoms=self.pairwise_max_atoms,
            )
            values = self.extra_fusion(values, global_values)
        output = self._readout_features(context["features"], values, cache["condition"])
        return self.charge_scale * output[:, :1], torch.softmax(output[:, 1:], dim=0)

    def cached_charges(self, batch):
        """Identical proposal computation with frozen small-system geometry."""
        features, condition = batch["features"], batch["condition"]
        b, n, _ = features.shape
        local = (self.input(features) + self.boundary_input(condition)
                 + self.potential_input(batch["potential"]))
        global_values = batch["kernel"] @ self.norm(local)
        values = self.fusion(local.flatten(0, 1), global_values.flatten(0, 1))
        if self.variant in ("B", "C"):
            kernel = (batch["kernel"].transpose(-1, -2) if self.variant == "C"
                      else batch["kernel"])
            global_values = kernel @ self.norm(values).reshape(b, n, -1)
            values = self.extra_fusion(values, global_values.flatten(0, 1))
        output = self._readout_features(
            features.flatten(0, 1), values, condition.flatten(0, 1),
        ).reshape(b, n, 2)
        raw = self.charge_scale * output[..., :1]
        weights = torch.softmax(output[..., 1:], dim=1)
        target = batch["total_charge"][:, None, None]
        charge = raw + weights * (target - raw.sum(dim=1, keepdim=True))
        # Same numerical feasibility guard as VariationalEnergyOutput.
        return charge + (target - charge.sum(dim=1, keepdim=True)) / n


def install_architecture(model, variant, seed=20261026):
    """Replace only head.proposal; the frozen energy and backbone are untouched."""
    from cache import output_head

    head = output_head(model)
    attention = head.internal_energy.attention
    head.proposal = ArchitectureChargeProposal.from_shared(
        head.proposal, variant, seed=seed,
        contraction=getattr(attention, "contraction", "linear"),
        pairwise_max_atoms=getattr(attention, "pairwise_max_atoms", 256),
    )
    model.requires_grad_(False)
    head.proposal.requires_grad_(True)
    return head.proposal


# These adapters retain the existing energy, residual formula, normalization,
# and objective. Only the proposal charge calculation changes.
from cache import FrozenSharedProblem, output_head
from problem import ResidualModel


class FrozenArchitectureProblem(FrozenSharedProblem):
    def __init__(self, model):
        head = output_head(model)
        self.energy, self.proposal = head.internal_energy, head.proposal
        if not isinstance(self.proposal, ArchitectureChargeProposal):
            raise TypeError("Install an ArchitectureChargeProposal first")
        if head.proposal_mode != "absolute":
            raise ValueError("This experiment requires absolute charge prediction")

    def charges(self, batch):
        return self.proposal.cached_charges(batch)


class ArchitectureResidualModel(ResidualModel):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.problem = FrozenArchitectureProblem(model)
        self.proposal = self.problem.proposal
        self.names = list(dict(self.proposal.named_parameters()))
        self.shapes = [parameter.shape for parameter in self.proposal.parameters()]
        self.sizes = [parameter.numel() for parameter in self.proposal.parameters()]
