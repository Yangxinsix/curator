"""Charge-only MATPES-MACE with an optional trainable scalar input norm."""
import torch
from torch import nn

import charge_only
from charge_only import SCRIPTS, ScratchTrainer, calibrate_training_inputs, load_foundation
from cache import output_head
from architectures import aggregate_prepared
from curator.layer.efa.shared import SharedTwoScaleQConditionedEFAEnergy


class ScalarRMSNorm(nn.Module):
    """Normalize all even scalars jointly per atom. Copy other irreps unchanged."""

    def __init__(self, irreps, anchor):
        super().__init__()
        channels = [i for (_, irrep), block in zip(irreps, irreps.slices())
                    if str(irrep) == '0e' for i in range(block.start, block.stop)]
        if not channels:
            raise ValueError('The charge interface needs at least one even scalar channel')
        self.register_buffer('scalar_indices', torch.tensor(channels, device=anchor.device),
                             persistent=False)
        self.gain = nn.Parameter(anchor.new_ones(len(channels)))
        self.eps = 1e-8

    def forward(self, features):
        scalars = features.index_select(-1, self.scalar_indices)
        scale = torch.rsqrt(scalars.square().mean(dim=-1, keepdim=True) + self.eps)
        return features.index_copy(-1, self.scalar_indices, scalars * scale * self.gain)


class RMSChargeEnergy(charge_only.ChargeOnlyEnergy):
    """The energy projections share one normalized input; the proposal keeps raw h."""

    def _energy_features(self, context):
        if '_charge_efa_input' in context:
            return context['_charge_efa_input']
        return self.feature_norm(context['features'])

    def _project_features(self, features, context):
        # The native preparation passes raw h to the proposal's local bypass.
        energy_h = (context['_charge_efa_input'] if '_charge_efa_input' in context
                    else self.feature_norm(features))
        return super()._project_features(energy_h, context)

    def prepare_proposal_context(self, context, proposal):
        context = {**context, '_charge_efa_input': self.feature_norm(context['features'])}
        context = SharedTwoScaleQConditionedEFAEnergy.prepare_proposal_context(self, context, proposal)
        cache = context['_shared_efa']
        cache['live_physics'] = self._coefficients(
            context['_charge_efa_input'], cache['geometry'],
            lambda x: aggregate_prepared(cache['prepared'], x, transpose=True, contraction='linear'),
        )
        return context

    def _attention_terms(self, context, state):
        # The cached charge contraction and standalone fixed-q functional both
        # read B from context['features']. The charge q itself is never normalized.
        energy_h = self._energy_features(context)
        return super()._attention_terms({**context, '_charge_efa_input': energy_h,
                                         'features': energy_h}, state)

    def prepare_dense_batch(self, features, kernel, geometry, total_charge, field, fixed_charge):
        batch = super().prepare_dense_batch(self.feature_norm(features), kernel, geometry,
                                            total_charge, field, fixed_charge)
        batch['features'] = features
        return batch

    def get_extra_state(self):
        return {**super().get_extra_state(), 'schema': 'acvr_charge_only_efa_rmsnorm_v1',
                'normalization': 'per_atom_0e_before_q_k_v_b', 'epsilon': self.feature_norm.eps}


def build_scratch_model(*, charge_feature_norm, **options):
    if charge_feature_norm not in ('raw', 'rmsnorm'):
        raise ValueError('charge_feature_norm must be raw or rmsnorm')
    model = charge_only.build_scratch_model(**options)
    if charge_feature_norm == 'rmsnorm':
        energy = output_head(model).internal_energy
        energy.feature_norm = ScalarRMSNorm(energy.feature_spec.irreps, next(energy.parameters()))
        energy.__class__ = RMSChargeEnergy
    return model


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if 'charge_feature_norm' not in state['model_options']:
        return charge_only.load_checkpoint(path, device)
    model = build_scratch_model(**state['model_options'])
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval()
