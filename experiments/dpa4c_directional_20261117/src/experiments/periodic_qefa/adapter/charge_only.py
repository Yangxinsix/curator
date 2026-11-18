"""Charge-dependent EFA on PaiNN or MATPES-MACE, sharing the original one-forward solver."""
import os
from pathlib import Path

import torch
from torch.nn import functional as F

DOCS = Path(__file__).resolve().parents[2]
os.environ.setdefault('ACVR_SOURCE', str(DOCS / 'efa-proposal-speed-20261022/source_snapshot'))
SCRIPTS = os.environ.setdefault('ACVR_SCRIPTS', str(DOCS / 'proposal-mace-campaign-20261106/scripts'))

from legacy_mace import ScratchTrainer, calibrate_training_inputs, load_foundation
from legacy_mace import build_scratch_model as _mace
from legacy_painn import build_scratch_model as _painn
from cache import output_head
from multiboundary import LiveResponseEnergy


class ChargeOnlyEnergy(LiveResponseEnergy):
    """Keep the geometric descriptor in the charge offset; F(R,0) is identically zero."""

    def response_terms(self, context, state):
        _, offset, response = self._attention_terms(context, state)
        linear = torch.sigmoid(offset) * response
        nonlinear = F.softplus(offset + response) - F.softplus(offset)
        weights = self.positive_output_weights
        return {
            'geometry_energy': state.new_zeros(()),
            'linear': (weights * linear).sum(),
            'curvature': (weights * (nonlinear - linear)).sum(),
            'offset': offset,
            'response': response,
        }

    def get_extra_state(self):
        return {**super().get_extra_state(), 'schema': 'acvr_charge_only_efa_v1'}


def install_charge_only(model):
    """Convert an assembled model in place before creating its optimizers."""
    energy = output_head(model).internal_energy
    if isinstance(energy, ChargeOnlyEnergy):
        return model
    if type(energy) is not LiveResponseEnergy or energy.feature_spec is None:
        raise TypeError('Expected a bound multi-boundary structured-response EFA energy')
    del energy.readout
    energy.__class__ = ChargeOnlyEnergy
    return model


def _assemble(options):
    builder = _mace if options.get('foundation_checkpoint') is not None else _painn
    return builder(**options)


def build_scratch_model(**model_options):
    """Build the new version with the original backbone and charge-head initialization."""
    return install_charge_only(_assemble(model_options))


def load_checkpoint(path, device='cpu'):
    """Strictly restore a new state or remove G from an existing campaign model state."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    weights = checkpoint['model']
    metadata = [v for k, v in weights.items() if k.endswith('.internal_energy._extra_state')]
    if len(metadata) != 1:
        raise ValueError('Expected one checkpointed EFA energy contract')
    model = _assemble(checkpoint['model_options'])
    if metadata[0].get('schema') == 'acvr_charge_only_efa_v1':
        install_charge_only(model)
    model.load_state_dict(weights, strict=True)
    return install_charge_only(model).to(device).eval()
