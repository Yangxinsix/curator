"""Explicit matched-objective control; separate from historical QEq loss."""
import torch
from torch import nn
from curator.data import properties
from curator.train.legacy_boundary_monitor import LegacyBoundaryMonitor
from curator.train.ef_monitor import EFPhysicalMonitor


class PerStructureStationarityMSE(nn.Module):
    requires_batch=True
    def forward(self,prediction,target,batch):
        if prediction.shape != target.shape:raise ValueError('state shape mismatch')
        return torch.stack([part.square().mean() for part in (prediction-target).split(batch[properties.n_atoms].tolist())]).mean()


class ConservativeQEqMonitor(LegacyBoundaryMonitor):
    def observe(self,module,inputs,output):
        EFPhysicalMonitor.observe(self,module,inputs,output)
        r=output[properties.chemical_potential_residual].detach().double()
        self.rsq+=float(r.square().sum());self.nr+=r.numel()
        for part in r.split(inputs[0][properties.n_atoms].tolist()):
            self.response_loss+=float(part.square().mean())

    def on_validation_epoch_end(self,trainer,pl_module):
        super().on_validation_epoch_end(trainer,pl_module)
        import json
        from pathlib import Path
        path=Path(self.output_path);lines=path.read_text().splitlines();r=json.loads(lines[-1])
        r['stationarity_weighted_loss']=r.pop('residual_force_weighted_loss')
        r['method']='legacy_parameterization_conservative_stationarity_control'
        lines[-1]=json.dumps(r);path.write_text('\n'.join(lines)+'\n')
