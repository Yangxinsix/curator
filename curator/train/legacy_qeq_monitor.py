"""Physical monitoring of the isolated historical direct-charge QEq baseline."""
from types import SimpleNamespace
import torch
from curator.data import properties
from curator.utils import scatter_add
from .acvr_monitor import ACVRPhysicalMonitor


class LegacyQEqPhysicalMonitor(ACVRPhysicalMonitor):
    def on_validation_epoch_start(self, trainer, pl_module):
        self._legacy_losses = {'energy_loss': 0., 'forces_loss': 0., 'residual_forces_loss': 0.}
        self._legacy_geometries = 0
        super().on_validation_epoch_start(trainer, pl_module)

    def _observe(self, module, inputs, output):
        batch = inputs[0]
        counts = batch[properties.n_atoms].reshape(-1)
        pointer = torch.nn.functional.pad(counts.cumsum(0), (1, 0))
        n = counts.numel()
        charge = output[properties.atomic_charge]
        constraint = scatter_add(charge, batch[properties.image_idx], dim=0) - batch[properties.total_charge].reshape(-1)
        def pack(values, ptr):
            return {'values': values.reshape(-1), 'ptr': ptr}
        adapted = {'variational_prediction': {'value_space': 'physical', 'labels': {
            'condition_energy': pack(output[properties.energy], torch.arange(n + 1, device=counts.device)),
            'condition_forces': pack(output[properties.forces], 3 * pointer),
            'variable_value': pack(charge, pointer)},
            'residuals': {'stationarity': {'values': output[properties.chemical_potential_residual],
                'normalization': {'kind': 'fixed_physical_scale', 'scale': 1., 'unit': 'V'}}},
            'constraint_violation': pack(constraint, torch.arange(n + 1, device=counts.device))}}
        super()._observe(module, inputs, adapted)
        for i in range(n):
            start, stop = int(pointer[i]), int(pointer[i + 1])
            self._legacy_losses['energy_loss'] += 200 * float(((output[properties.energy].reshape(-1)[i] - batch[properties.energy].reshape(-1)[i]) / counts[i]).detach().square())
            self._legacy_losses['forces_loss'] += 200 * float((output[properties.forces][start:stop] - batch[properties.forces][start:stop]).detach().square().mean())
            self._legacy_losses['residual_forces_loss'] += 200000 * float(output[properties.residual_forces][start:stop].detach().square().mean())
            self._legacy_geometries += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        values = {key: value / self._legacy_geometries for key, value in self._legacy_losses.items()}
        output = SimpleNamespace(name='legacy', accumulate_metrics=lambda subset: values if subset == 'val' else {})
        proxy = SimpleNamespace(outputs=[output], log=pl_module.log)
        super().on_validation_epoch_end(trainer, proxy)
        if not trainer.sanity_checking:
            pl_module.log('val_legacy_total_loss', sum(values.values()), on_step=False, on_epoch=True,
                          sync_dist=False, add_dataloader_idx=False)
