"""Physical energy/force diagnostics for the matched single-device benchmarks."""
import json, math, time
from pathlib import Path
import torch
from pytorch_lightning import Callback
from curator.data import properties

class EFPhysicalMonitor(Callback):
    def __init__(self,output_path,energy_weight=200.,force_weight=200.):
        self.output_path=output_path
        self.energy_weight=float(energy_weight)
        self.force_weight=float(force_weight)
        self.hook=None
    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook=None
    def on_validation_epoch_start(self,trainer,pl_module):
        if trainer.world_size != 1:
            raise ValueError("EFPhysicalMonitor supports one device")
        self.remove()
        self.started=time.monotonic()
        self.esq=self.fsq=self.loss=0.
        self.ne=self.nf=0
        self.hook=pl_module.model.register_forward_hook(self.observe)
    def observe(self,module,inputs,output):
        batch=inputs[0]
        n=batch[properties.n_atoms].reshape(-1).detach().cpu()
        de=(output[properties.energy].detach().cpu().double().reshape(-1)-batch[properties.energy].detach().cpu().double().reshape(-1))/n
        df=output[properties.forces].detach().cpu().double()-batch[properties.forces].detach().cpu().double()
        assert torch.isfinite(de).all() and torch.isfinite(df).all()
        self.esq+=float(de.square().sum());self.ne+=de.numel()
        self.fsq+=float(df.square().sum());self.nf+=df.numel()
        start=0
        for i,count in enumerate(n):
            stop=start+int(count)
            self.loss+=self.energy_weight*float(de[i].square())+self.force_weight*float(df[start:stop].square().mean())
            start=stop
    def on_validation_epoch_end(self,trainer,pl_module):
        self.remove()
        r={'stage':'initial' if trainer.sanity_checking else 'validation','epoch':int(trainer.current_epoch),'global_step':int(trainer.global_step),'energy_rmse_meV_per_atom':1000*math.sqrt(self.esq/self.ne),'force_component_rmse_eV_per_Angstrom':math.sqrt(self.fsq/self.nf),'counts':{'energy':self.ne,'forces':self.nf},'val_physical_total_loss':self.loss/self.ne,'validation_seconds':time.monotonic()-self.started}
        p=Path(self.output_path);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('a') as f:f.write(json.dumps(r)+'\n')
        if not trainer.sanity_checking:
            pl_module.log('val_physical_total_loss',r['val_physical_total_loss'],on_epoch=True,on_step=False)
            pl_module.log('val_physical_energy_rmse_meV_per_atom',r['energy_rmse_meV_per_atom'],on_epoch=True,on_step=False)
            pl_module.log('val_physical_force_component_rmse_eV_per_Angstrom',r['force_component_rmse_eV_per_Angstrom'],on_epoch=True,on_step=False)
    def on_exception(self,trainer,pl_module,exception):self.remove()
    def teardown(self,trainer,pl_module,stage):self.remove()
