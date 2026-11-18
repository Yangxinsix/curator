"""Build the RMSNorm / KAN charge model with periodic reciprocal communication."""
import hashlib
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ACVR_SOURCE", str(ROOT / "curator_source"))
# The sealed training adapter imports its helpers as top-level modules. This
# edition of architectures dispatches both directions to the selected backend.
os.environ["ACVR_SCRIPTS"] = str(ROOT / "scripts")
sys.path[:0] = [os.environ["ACVR_SOURCE"], str(ROOT / "scripts"), str(ROOT / "adapter")]
from rms_parent import RMSChargeEnergy, SCRIPTS, ScratchTrainer, calibrate_training_inputs
from rms_parent import build_scratch_model as build_rms_model
from cache import output_head
from architectures import aggregate_prepared
from curator.layer.efa.geometry import structure_cells
from curator.layer.efa.reciprocal import ReciprocalFastAttention
from linear_neighbors import install_linear_neighbors


class LocalOffsetEnergy(RMSChargeEnergy):
    """Store the local Value in the geometry slot used by both energy and proposal."""

    def _local_context(self, context):
        normalized = self._energy_features(context)
        context = {**context, '_charge_efa_input': normalized}
        query, key, value = self._project_features(context['features'], context)
        prepared = self.attention.prepare(query, key, context['positions'],
                                          cell=structure_cells(context, 1)[0])
        return {**context, '_shared_efa': {'prepared': prepared, 'geometry': value}}

    def prepare_proposal_context(self, context, proposal):
        context = self._local_context(context)
        cache = context['_shared_efa']
        local, condition = proposal.local_features(context)
        cache.update(local=local, condition=condition,
                     **{'global': self.attention.aggregate(cache['prepared'], proposal.norm(local))})
        cache['live_physics'] = self._coefficients(
            context['_charge_efa_input'], cache['geometry'],
            lambda x: aggregate_prepared(cache['prepared'], x, transpose=True, contraction='linear'))
        return context

    def _attention_terms(self, context, state):
        if '_shared_efa' not in context:
            context = self._local_context(context)
        return super()._attention_terms(context, state)

    def prepare_dense_batch(self, features, kernel, geometry, total_charge, field, fixed_charge):
        local_value = self.value(self.feature_norm(features))
        return super().prepare_dense_batch(features, kernel, local_value, total_charge, field, fixed_charge)

    def prepare_frozen_audit_batch(self, batch, field=None, fixed_charge=None):
        normalized = self.feature_norm(batch['features'])
        local_value = self.value(normalized)
        weights = self.charge_affine.weight[:, :self.hidden_features]
        offset = torch.nn.functional.linear(local_value, weights, self.charge_affine.bias)
        return super().prepare_frozen_audit_batch(
            {**batch, 'offset': offset, 'charge_value': self.charge_value(normalized)}, field, fixed_charge)

    def get_extra_state(self):
        return {**super().get_extra_state(), 'offset_readout': 'local_projected_geometry_value'}


def build_scratch_model(*, offset_readout, charge_feature_norm, periodic_backend="explicit",
                        periodic_cutoff=0.86, periodic_reference_volume=45.,
                        periodic_charge_condition="density", **options):
    if offset_readout not in ('global', 'local') or charge_feature_norm != 'rmsnorm':
        raise ValueError('Expected RMSNorm with a global or local geometry offset')
    model = build_rms_model(charge_feature_norm=charge_feature_norm, **options)
    head = output_head(model)
    energy = head.internal_energy
    if periodic_charge_condition not in ("density", "total"):
        raise ValueError("periodic_charge_condition must be density or total")
    if offset_readout == 'local':
        energy.__class__ = LocalOffsetEnergy
    if periodic_backend != "legacy":
        energy.attention = ReciprocalFastAttention.from_attention(
            energy.attention, periodic_backend=periodic_backend, periodic_cutoff=periodic_cutoff,
            periodic_reference_volume=periodic_reference_volume)
        head.proposal.periodic_charge_condition = periodic_charge_condition
        head.proposal.periodic_reference_volume = energy.attention.periodic_reference_volume
    return install_linear_neighbors(model)


def load_checkpoint(path, device='cpu', *, periodic_backend=None, periodic_charge_condition=None):
    """Replay the saved model; pass a backend explicitly to convert old weights."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    options = dict(state['model_options'])
    if "periodic_backend" in options:
        # Early reciprocal checkpoints used the absolute total-charge input.
        options.setdefault("periodic_charge_condition", "total")
    options.setdefault("periodic_backend", "legacy")
    if periodic_backend is not None:
        options["periodic_backend"] = periodic_backend
    if periodic_charge_condition is not None:
        options["periodic_charge_condition"] = periodic_charge_condition
    model = build_scratch_model(**options)
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval()


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(repr((name, tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
