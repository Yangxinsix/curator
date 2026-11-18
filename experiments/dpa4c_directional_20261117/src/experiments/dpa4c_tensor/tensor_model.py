"""DPA4C/ACVR arms: change angular attention *scores* or transmitted Values.

Build the historical model (including A3/B2), load its checkpoint strictly,
then install a variant before constructing the optimizers. To reload a saved
variant install that same variant first and load its full state strictly.
"""
from pathlib import Path
import sys
from math import sqrt

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dpa4c_acvr"))
from model import RMSChargeEnergy  # selects the same Curator as the historical runner
from curator.data import properties as p
from curator.layer.efa.geometry import structure_cells


ARMS = {"old": (), "qk2": (), "orient1": (1,), "orient2": (2,), "orient12": (1, 2)}


class TensorBranch(nn.Module):
    def __init__(self, backbone, arm, width, exclude_self):
        super().__init__()
        self.arm = arm
        self.ranks = ARMS[arm]
        self.exclude_self = exclude_self
        self.offsets = tuple(backbone.moment_offsets)
        self.widths = tuple(backbone.degree_channels)
        if backbone.l_max < 2 or len(self.widths) < 3:
            raise ValueError("DPA4C must provide native rank-two moments")
        if arm == "qk2":
            self.query2 = nn.Linear(self.widths[2], width, bias=False)
            self.key2 = nn.Linear(self.widths[2], width, bias=False)
            # Match the original kernel without making both new factors dead.
            nn.init.zeros_(self.query2.weight)
        else:
            self.projections = nn.ModuleDict()
            self.readout = nn.ParameterDict()
            for rank in self.ranks:
                name = str(rank)
                w = self.widths[rank]
                self.projections[name] = nn.Linear(w, w, bias=False)
                nn.init.orthogonal_(self.projections[name].weight)
                self.readout[name] = nn.Parameter(torch.zeros(w))

    def moments(self, raw, rank):
        """DPA packs [component, channel]. Its five STF coefficients are isometric."""
        lo, hi = self.offsets[rank:rank + 2]
        return raw[..., lo:hi].reshape(*raw.shape[:-1], 2 * rank + 1, self.widths[rank])

    def extend_qk(self, q, k, raw):
        if self.arm != "qk2":
            return q, k
        if q.shape[-2] != 4:
            raise ValueError("Expected the historical 0e+1o angular components")
        T = self.moments(raw, 2)
        # prepare() divides by sqrt(number of components * hidden width).
        # Preserve exactly the old old-old bilinear scale when growing 4 -> 9.
        return (torch.cat((q * sqrt(9 / 4), self.query2(T)), dim=-2),
                torch.cat((k, self.key2(T)), dim=-2))

    def orient_energy(self, raw, left, right):
        """One factorized application for all tensor components, no atom-pair matrix."""
        # raw, left, right: [batch, atoms, width]; left already includes all weights.
        source = [self.projections[str(l)](self.moments(raw, l)) for l in self.ranks]
        columns = [t.flatten(-2) for t in source]
        x = torch.cat(columns, dim=-1)
        remote = left @ (right.mT @ x)
        if self.exclude_self:
            diagonal = (left * right).sum(-1, keepdim=True)
            remote = remote - diagonal * x
        messages = remote.split([c.shape[-1] for c in columns], dim=-1)
        # Dot products in Cartesian rank one, Frobenius inner products in the
        # orthonormal five-component symmetric traceless rank-two basis.
        return sum(((t * msg.reshape_as(t)).sum(-2) * self.readout[str(l)]).sum((-2, -1))
                   for t, msg, l in zip(source, messages, self.ranks))


def _factors(prepared):
    # Fused molecular path: quadrature/scaling already packed on the left.
    if hasattr(prepared, "q"):
        return prepared.q, prepared.k
    # Reference molecular and periodic A3 paths represent exactly the same
    # bilinear operator with an explicit direction axis and quadrature weights.
    q, k, w = prepared
    return (q * w[None, :, None]).flatten(1)[None], k.flatten(1)[None]


class TensorEnergy(RMSChargeEnergy):
    def geometry_context(self, data):
        chunks = data[p.node_embedding].split(data[p.n_atoms].reshape(-1).tolist())
        return [{"_dpa_moments": raw} for raw in chunks]

    def _project_features(self, features, context):
        q, k, value = super()._project_features(features, context)
        q, k = self.tensor_branch.extend_qk(q, k, context["_dpa_moments"])
        return q, k, value

    def _with_tensor_energy(self, context):
        if self.tensor_branch.ranks:
            left, right = _factors(context["_shared_efa"]["prepared"])
            extra = {"normalized_features": self._energy_features(context)[None]} if hasattr(
                self.tensor_branch, "chemical_gate"
            ) else {}
            orient = self.tensor_branch.orient_energy(context["_dpa_moments"][None], left, right, **extra)
            return {**context, "_orient_energy": orient.squeeze(0)}
        return context

    def prepare_proposal_context(self, context, proposal):
        return self._with_tensor_energy(super().prepare_proposal_context(context, proposal))

    def response_terms(self, context, state):
        if "_shared_efa" not in context:
            # A direct U(R,q) call has no proposal/geometry cache. Build exactly
            # the same q-independent factors including the new Q/K block.
            q, k, v = self._project_features(context["features"], context)
            prepared = self.attention.prepare(q, k, context["positions"],
                                             cell=structure_cells(context, 1)[0])
            left, right = _factors(prepared)
            geometry = (left @ (right.mT @ v[None])).squeeze(0)
            context = self._with_tensor_energy({
                **context, "_shared_efa": {"geometry": geometry, "prepared": prepared}
            })
        terms = super().response_terms(context, state)
        if self.tensor_branch.ranks:
            terms["geometry_energy"] = context["_orient_energy"]
        return terms

    def get_extra_state(self):
        return {**super().get_extra_state(), "tensor_arm": self.tensor_branch.arm,
                "exclude_self": self.tensor_branch.exclude_self}


def install_variant(model, arm, *, seed=3407, exclude_self=True):
    """In-place installation, deterministic without changing the training RNG."""
    if arm not in ARMS:
        raise ValueError(f"unknown tensor variant {arm}")
    energy = model.output_modules[0].internal_energy
    if not isinstance(energy, RMSChargeEnergy) or hasattr(energy, "tensor_branch"):
        raise ValueError("Install once into the historical RMSNorm charge-only model")
    if arm == "old":
        return model
    reference = next(energy.parameters())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        branch = TensorBranch(model.representation, arm, energy.hidden_features, exclude_self)
    energy.tensor_branch = branch.to(reference)
    energy.__class__ = TensorEnergy
    return model
