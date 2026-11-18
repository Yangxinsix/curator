"""Physical ACVR validation diagnostics without an additional model evaluation."""
import json
import math
import time
from pathlib import Path

import torch
from pytorch_lightning import Callback

from curator.data import properties


class ACVRPhysicalMonitor(Callback):
    """Single-device physical metrics at initialization and every validation.

    The temporary hook retains only Python scalar sums, never autograd tensors.
    It is removed before checkpoint callbacks can serialize the complete model.
    Loss components come from Curator's existing structured accumulators.
    """
    def __init__(self, output_path):
        self.output_path = str(output_path)
        self._hook = None

    def _remove_hook(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.world_size != 1:
            raise ValueError('ACVRPhysicalMonitor currently supports one device')
        self._remove_hook()
        self._started = time.monotonic()
        self._sums = {key: 0.0 for key in ('energy', 'forces', 'stationarity', 'charge')}
        self._counts = {key: 0 for key in self._sums}
        self._constraint_max = 0.0
        self._hook = pl_module.model.register_forward_hook(self._observe)

    def _add(self, name, values):
        values = values.detach().to(device='cpu', dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError(f'nonfinite physical validation values: {name}')
        self._sums[name] += float(values.square().sum())
        self._counts[name] += values.numel()

    def _observe(self, module, inputs, output):
        del module
        batch = inputs[0]
        predicted = output['variational_prediction']
        if predicted.get('value_space') != 'physical':
            raise ValueError('physical monitor requires physical variational predictions')
        target = batch['variational']
        labels = predicted['labels']
        geometry = target['condition_geometry_index'].detach().cpu()
        counts = batch[properties.n_atoms].detach().cpu().reshape(-1)
        for name, metric in (('condition_energy', 'energy'), ('condition_forces', 'forces')):
            truth, estimate = target['labels'][name], labels[name]
            present = truth['present'] & truth.get('eligible', torch.ones_like(truth['present']))
            for condition in range(present.numel()):
                if not bool(present[condition]):
                    continue
                left, right = int(truth['ptr'][condition]), int(truth['ptr'][condition + 1])
                pred_left, pred_right = int(estimate['ptr'][condition]), int(estimate['ptr'][condition + 1])
                value = estimate['values'][pred_left:pred_right].detach().to('cpu', torch.float64)
                reference = truth['values'][left:right].detach().to('cpu', torch.float64)
                if value.numel() != reference.numel():
                    raise ValueError('physical prediction and label shapes differ')
                error = value - reference
                if metric == 'energy':
                    error = error / int(counts[int(geometry[condition])])
                self._add(metric, error)
        residual = predicted['residuals'].get('stationarity')
        if residual is None:
            if predicted.get('metadata', {}).get('state_semantics') != 'predicted_auxiliary':
                raise ValueError('variational prediction is missing stationarity residuals')
        else:
            normalization = residual['normalization']
            if normalization.get('kind') != 'fixed_physical_scale' or normalization.get('unit') not in {'V', 'eV/e'}:
                raise ValueError('stationarity monitor requires a fixed physical voltage scale')
            self._add('stationarity', residual['values'] * float(normalization['scale']))
        self._add('charge', labels['variable_value']['values'])
        violation = predicted['constraint_violation']['values'].detach()
        if violation.numel():
            if not torch.isfinite(violation).all():
                raise ValueError('nonfinite constraint violation')
            self._constraint_max = max(self._constraint_max, float(violation.abs().max()))

    def on_validation_epoch_end(self, trainer, pl_module):
        self._remove_hook()
        if not self._counts['energy'] or not self._counts['forces']:
            raise ValueError('physical validation has no energy or force observations')
        rms = {key: math.sqrt(self._sums[key] / count) if count else None for key, count in self._counts.items()}
        losses = {}
        for subset in ('train', 'val'):
            losses[subset] = {}
            for output in pl_module.outputs:
                if hasattr(output, 'accumulate_metrics'):
                    values = output.accumulate_metrics(subset)
                    losses[subset].update({f'{output.name}/{key}': float(value) for key, value in values.items()})
            losses[subset]['total'] = sum(losses[subset].values())
        report = {
            'stage': 'initial' if trainer.sanity_checking else 'validation',
            'epoch': int(trainer.current_epoch), 'global_step': int(trainer.global_step),
            'energy_rmse_meV_per_atom': 1000 * rms['energy'],
            'force_component_rmse_eV_per_Angstrom': rms['forces'],
            'stationarity_rms_V': rms['stationarity'], 'charge_rms_e': rms['charge'],
            'constraint_max_e': self._constraint_max, 'counts': dict(self._counts),
            'loss_components': losses, 'validation_seconds': time.monotonic() - self._started,
        }
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as handle:
            handle.write(json.dumps(report, allow_nan=False) + '\n')
        if not trainer.sanity_checking:
            for key in ('energy_rmse_meV_per_atom', 'force_component_rmse_eV_per_Angstrom',
                        'stationarity_rms_V', 'charge_rms_e', 'constraint_max_e'):
                if report[key] is not None:
                    pl_module.log('val_physical_' + key, report[key], on_step=False, on_epoch=True,
                                  sync_dist=False, add_dataloader_idx=False)

    def on_exception(self, trainer, pl_module, exception):
        self._remove_hook()

    def teardown(self, trainer, pl_module, stage):
        self._remove_hook()
