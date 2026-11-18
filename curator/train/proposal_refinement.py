"""Frozen-functional proposal diagnostics, used through the native Curator trainer."""
import hashlib
import json
from pathlib import Path
import torch
from .acvr_monitor import ACVRPhysicalMonitor

class ProposalRefinementMonitor(ACVRPhysicalMonitor):
    """Freeze everything except proposal, audit tensor state, retain EF diagnostics."""
    def __init__(self, output_path, audit_path):
        super().__init__(output_path)
        self.audit_path = str(audit_path)

    def setup(self, trainer, pl_module, stage):
        model = pl_module.model
        heads = [(n, m) for n, m in model.named_modules() if hasattr(m, 'proposal_mode') and hasattr(m, 'proposal')]
        if len(heads) != 1:
            raise ValueError('diagnostic requires exactly one variational state head')
        name, head = heads[0]
        self.proposal_prefix = name + '.proposal.'
        model.requires_grad_(False)
        head.proposal.requires_grad_(True)
        self.before = self._hashes(model)
        self.trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        if not self.trainable or any(not n.startswith(self.proposal_prefix) for n in self.trainable):
            raise ValueError('unexpected trainable parameters')
        self._audit(model, 'initial')

    def _hashes(self, model):
        result = {}
        for name, value in model.state_dict().items():
            if isinstance(value, torch.Tensor):
                result[name] = hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        return result

    def _audit(self, model, stage):
        hashes = self._hashes(model)
        changed = [n for n, h in hashes.items() if self.before.get(n) != h]
        outside = [n for n in changed if not n.startswith(self.proposal_prefix)]
        row = dict(stage=stage, only_proposal_trainable=True, trainable_parameters=self.trainable,
                   changed_tensors=changed, frozen_tensors_unchanged=not outside,
                   frozen_hashes={n:h for n,h in hashes.items() if not n.startswith(self.proposal_prefix)})
        path = Path(self.audit_path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as f: f.write(json.dumps(row) + '\n')
        if outside: raise RuntimeError('frozen tensors changed: ' + str(outside))

    def on_validation_epoch_start(self, trainer, pl_module):
        self.frame_residual_rms = []
        super().on_validation_epoch_start(trainer, pl_module)

    def _observe(self, module, inputs, output):
        super()._observe(module, inputs, output)
        value = output['variational_prediction']['residuals']['stationarity']
        r = value['values'].detach().double().cpu() * float(value['normalization']['scale'])
        for left, right in zip(value['ptr'][:-1], value['ptr'][1:]):
            self.frame_residual_rms.append(float(r[int(left):int(right)].square().mean().sqrt()))

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        r = torch.tensor(self.frame_residual_rms, dtype=torch.float64)
        row = dict(stage='initial' if trainer.sanity_checking else 'validation', epoch=int(trainer.current_epoch),
                   global_step=int(trainer.global_step), n_frames=len(r),
                   residual_rms_V_quantiles=dict(zip(('median','p95','p99','max'), torch.quantile(r, torch.tensor([.5,.95,.99,1.],dtype=r.dtype)).tolist())))
        with Path(self.output_path).with_name('residual_distribution.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        self._audit(pl_module.model, row['stage'] + '_' + str(trainer.current_epoch))

    def on_train_end(self, trainer, pl_module):
        self._audit(pl_module.model, 'train_end')
