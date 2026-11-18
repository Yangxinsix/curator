"""Exact frozen-geometry batching of the existing shared charge proposal.

Only the small NaCl geometries are supported here. Their frozen EFA kernels
are cached explicitly; no proposal Values, gates or readouts are cached.
Physical forces must always be evaluated through the original full model.
"""
from collections import defaultdict
import time

from common import ROOT, write_json
import torch
from torch.nn import functional as F
from curator.data import properties as p
from curator.data.collate_atoms_data import collate_atoms_data
from curator.layer import VariationalEnergyOutput
from curator.layer.efa.readout import invariant_features
from equilibrium import model_inputs


def output_head(model):
    heads = [m for m in model.output_modules if isinstance(m, VariationalEnergyOutput)]
    if len(heads) != 1:
        raise ValueError("Expected one variational output")
    return heads[0]


def collate_cached(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[len(row['features'])].append(row)
    return [{key: torch.stack([r[key] for r in group]) for key in group[0]}
            for group in groups.values()]


class FrozenSharedProblem:
    """Reuse native proposal modules and the unchanged two-scale energy."""

    def __init__(self, model):
        head = output_head(model)
        self.energy, self.proposal = head.internal_energy, head.proposal
        if type(self.proposal).__name__ != 'SharedGeometryChargeProposal':
            raise TypeError('Expected the shared proposal without conversion')
        if head.proposal_mode != 'absolute':
            raise ValueError('This experiment requires absolute charge prediction')

    def pack(self, context, source_index):
        if context['features'].shape[0] > 256:
            raise ValueError('Dense frozen kernels are restricted to small diagnostic systems')
        self.proposal._total_charge(context)
        condition, potential = self.proposal._conditions(context)
        cache = context['_shared_efa']
        query, key, weights = cache['prepared']
        kernel = (query * weights[None, :, None]).flatten(1) @ key.flatten(1).T
        geometry = cache['geometry']
        width = self.energy.hidden_features
        row = {
            'features': context['features'], 'condition': condition,
            'potential': potential, 'kernel': kernel,
            'offset': F.linear(geometry, self.energy.charge_affine.weight[:, :width],
                               self.energy.charge_affine.bias),
            'charge_value': self.energy.charge_value(context['features']),
            'geometry_energy': self.energy.energy_scale * self.energy.readout(geometry).sum(),
            'total_charge': context['constraint_target'].reshape(()),
            'source_index': torch.tensor(source_index, device=kernel.device),
        }
        return {k: v.detach() for k, v in row.items()}

    def charges(self, batch):
        proposal = self.proposal
        features, condition = batch['features'], batch['condition']
        b, n, _ = features.shape
        local = (proposal.input(features) + proposal.boundary_input(condition)
                 + proposal.potential_input(batch['potential']))
        global_values = batch['kernel'] @ proposal.norm(local)
        mixed = proposal.fusion(local.flatten(0, 1), global_values.flatten(0, 1))
        invariants = torch.cat((
            invariant_features(features, proposal.feature_spec.irreps),
            invariant_features(mixed, proposal.irreps).reshape(b, n, -1), condition,
        ), dim=-1)
        output = proposal.readout(invariants)
        raw = proposal.charge_scale * output[..., :1]
        weights = torch.softmax(output[..., 1:], dim=1)
        charge = raw + weights * (batch['total_charge'][:, None, None]
                                  - raw.sum(dim=1, keepdim=True))
        # Same ordinary fixed-total-charge feasibility guard as the native head.
        return charge + (batch['total_charge'][:, None, None]
                         - charge.sum(dim=1, keepdim=True)) / n

    def energies(self, batch, charge):
        energy = self.energy
        response = F.linear(torch.cat((batch['kernel'] @ (charge * batch['charge_value']),
                                       charge), dim=-1),
                            energy.charge_affine.weight[:, energy.hidden_features:])
        offset, weights = batch['offset'], energy.positive_output_weights
        linear = torch.sigmoid(offset) * response
        curvature = F.softplus(offset + response) - F.softplus(offset) - linear
        return (batch['geometry_energy'] + energy.driving_scale * (weights * linear).sum((1, 2))
                + energy.curvature_scale * (weights * curvature).sum((1, 2))
                + .5 * energy.hardness * charge.square().sum((1, 2)))

    def residual(self, batch, create_graph=False):
        charge = self.charges(batch)
        if not charge.requires_grad:
            charge.requires_grad_(True)
        energy = self.energies(batch, charge)
        gradient, = torch.autograd.grad(energy.sum(), charge, create_graph=create_graph)
        return gradient - gradient.mean(dim=1, keepdim=True), charge, energy


def build_cache(model, data, checkpoint_sha):
    path = ROOT / 'cache/geometry.pt'
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['checkpoint_sha256'] != checkpoint_sha:
            raise ValueError('Cache belongs to a different energy model')
        return saved
    head, problem = output_head(model), FrozenSharedProblem(model)
    model.eval().requires_grad_(False)
    head.compute_forces = head.retain_response_graph = head.compute_responses = False
    rows, checks = {}, []
    started = time.monotonic()
    for name, dataset in (('train', data.train_dataset), ('validation', data.val_dataset)):
        rows[name] = []
        for start in range(0, len(dataset), 32):
            raw = collate_atoms_data([dataset[i] for i in range(start, min(start + 32, len(dataset)))])
            calls = []
            hook = head.internal_energy.register_forward_pre_hook(
                lambda module, args: calls.append(args))
            try:
                with torch.enable_grad():
                    result = model(model_inputs(raw, 'cuda'))['variational_prediction']
            finally:
                hook.remove()
            if len(calls) != len(raw['meta']):
                raise RuntimeError('Expected one native charge functional call per structure')
            native_r = result['residuals']['stationarity']
            rs = native_r['values'].detach().reshape(-1) * native_r['normalization']['scale']
            offset = 0
            for meta, (context, state) in zip(raw['meta'], calls):
                row = problem.pack(context, int(meta['source_index']))
                n = len(row['features'])
                if len(checks) < 8:
                    batch = collate_cached([row])[0]
                    with torch.enable_grad():
                        residual, q, value = problem.residual(batch)
                    torch.testing.assert_close(q[0], state, atol=3e-7, rtol=3e-5)
                    torch.testing.assert_close(residual.flatten(), rs[offset:offset+n],
                                               atol=3e-7, rtol=3e-4)
                    expected = head.internal_energy(context, state).detach()
                    torch.testing.assert_close(value[0], expected, atol=2e-6, rtol=2e-5)
                    checks.append({'source_index': int(meta['source_index']),
                                   'q_max_error_e': float((q[0]-state).abs().max()),
                                   'residual_max_error_V': float((residual.flatten()-rs[offset:offset+n]).abs().max()),
                                   'energy_error_eV': float((value[0]-expected).abs())})
                rows[name].append({k: v.cpu() for k, v in row.items()})
                offset += n
            del calls, result
            if start % 512 == 0:
                print('CACHE_PROGRESS', name, start, len(dataset), flush=True)
    saved = {'checkpoint_sha256': checkpoint_sha, 'splits': rows,
             'seconds': time.monotonic()-started, 'checks': checks}
    temporary = path.with_suffix('.tmp')
    torch.save(saved, temporary)
    temporary.replace(path)
    write_json(ROOT/'results/cache.json', {k:v for k,v in saved.items() if k != 'splits'} | {
        'counts': {name: len(v) for name,v in rows.items()}, 'bytes': path.stat().st_size})
    return saved
