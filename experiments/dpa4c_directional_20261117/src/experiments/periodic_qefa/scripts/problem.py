"""Pure differentiable original stationarity residual for frozen shared EFA."""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F
from cache import FrozenSharedProblem, collate_cached, output_head


def frozen_digest(model):
    ids = {id(p) for p in output_head(model).proposal.parameters()}
    digest = hashlib.sha256()
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if id(value) not in ids:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gpu_batches(rows, chunk=256):
    return [{k: v.cuda() for k, v in batch.items()}
            for start in range(0, len(rows), chunk)
            for batch in collate_cached(rows[start:start + chunk])]


class ResidualModel(nn.Module):
    """Native proposal with the analytic derivative of the unchanged energy.

    Functional transforms see only ordinary tensor operations, avoiding nested
    autograd.grad inside JVP. Preflight verifies values and parameter gradients
    against the native energy derivative.
    """

    def __init__(self, model):
        super().__init__()
        self.problem = FrozenSharedProblem(model)
        self.proposal = self.problem.proposal
        self.names = list(dict(self.proposal.named_parameters()))
        self.shapes = [p.shape for p in self.proposal.parameters()]
        self.sizes = [p.numel() for p in self.proposal.parameters()]

    def physical(self, batch):
        q = self.problem.charges(batch)
        energy = self.problem.energy
        weight = energy.charge_affine.weight[:, energy.hidden_features:]
        response = F.linear(torch.cat((batch['kernel'] @ (q * batch['charge_value']), q), -1), weight)
        initial = torch.sigmoid(batch['offset'])
        argument = batch['offset'] + response
        derivative = torch.where(argument > 20, torch.ones_like(argument), torch.sigmoid(argument))
        coefficient = energy.positive_output_weights * (
            energy.driving_scale * initial
            + energy.curvature_scale * (derivative - initial))
        remote = (batch['kernel'].transpose(-1, -2) @ (coefficient @ weight[:, :-1]))
        g = (remote * batch['charge_value']).sum(-1, keepdim=True)
        g = g + (coefficient * weight[:, -1]).sum(-1, keepdim=True) + energy.hardness * q
        return g - g.mean(1, keepdim=True), q

    def forward(self, batches):
        atoms = sum(batch['features'].shape[0] * batch['features'].shape[1] for batch in batches)
        # 0.5 ||r_normalized||² equals the existing 30 * physical residual MSE.
        return torch.cat([self.physical(b)[0].flatten() for b in batches]) * math.sqrt(60 / atoms)

    def flat(self):
        return torch.cat([p.detach().flatten() for p in self.proposal.parameters()])

    def assign(self, vector):
        with torch.no_grad():
            for p, part in zip(self.proposal.parameters(), vector.split(self.sizes)):
                p.copy_(part.reshape_as(p))

    def function(self, batches):
        def residual(vector):
            params = {'proposal.' + name: part.reshape(shape)
                      for name, part, shape in zip(self.names, vector.split(self.sizes), self.shapes)}
            return torch.func.functional_call(self, params, (batches,), strict=False)
        return residual


@torch.no_grad()
def metrics(problem, batches):
    squared, count, constraint = 0., 0, 0.
    frames = []
    for batch in batches:
        r, q = problem.physical(batch)
        squared += float(r.square().sum())
        count += r.numel()
        frames.append(r.square().mean((1, 2)).sqrt())
        constraint = max(constraint, float((q.sum((1, 2)) - batch['total_charge']).abs().max()))
    frames = torch.cat(frames)
    return {'stationarity_rms_V': math.sqrt(squared / count), 'atoms': count,
            'structures': len(frames), 'frame_rms_p95_V': float(torch.quantile(frames, .95)),
            'frame_rms_max_V': float(frames.max()), 'constraint_max_e': constraint}
