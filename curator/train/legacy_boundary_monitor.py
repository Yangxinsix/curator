"""Matched physical EF and read-only KKT monitoring for historical QEq."""
import json, math
from pathlib import Path
from curator.train.ef_monitor import EFPhysicalMonitor
from curator.data import properties


class LegacyBoundaryMonitor(EFPhysicalMonitor):
    def on_validation_epoch_start(self, trainer, pl_module):
        self.rsq=self.nr=self.response_loss=0.
        super().on_validation_epoch_start(trainer, pl_module)

    def observe(self, module, inputs, output):
        super().observe(module, inputs, output)
        residual=output[properties.chemical_potential_residual].detach().double()
        self.rsq+=float(residual.square().sum()); self.nr+=residual.numel()
        sizes=inputs[0][properties.n_atoms].tolist()
        for part in output[properties.residual_forces].detach().double().split(sizes):
            self.response_loss+=200000.*float(part.square().mean())

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        path=Path(self.output_path)
        lines=path.read_text().splitlines()
        record=json.loads(lines[-1])
        record.update(stationarity_rms_V=math.sqrt(self.rsq/self.nr),
            residual_force_weighted_loss=self.response_loss/self.ne,
            val_legacy_total_loss=(self.loss+self.response_loss)/self.ne)
        lines[-1]=json.dumps(record)
        path.write_text('\n'.join(lines)+'\n')
        if not trainer.sanity_checking:
            pl_module.log('val_legacy_total_loss',record['val_legacy_total_loss'],
                on_epoch=True,on_step=False)
            pl_module.log('val_physical_stationarity_rms_V',record['stationarity_rms_V'],
                on_epoch=True,on_step=False)
