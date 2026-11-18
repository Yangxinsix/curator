"""Convert official MATPES to native Curator, verify E/F, fit train-only offsets."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ase.io import read
from curator.data import properties
from curator.data._data_reader import AseDataReader
from curator.model.conversion import create_model_from_mace
from curator.layer import GlobalRescaleShift

p=argparse.ArgumentParser()
p.add_argument('--source',required=True)
p.add_argument('--data',required=True)
p.add_argument('--output',required=True)
p.add_argument('--seed',type=int,default=3407)
a=p.parse_args()
out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
torch.set_default_dtype(torch.float64)
torch.use_deterministic_algorithms(True)
device=torch.device('cuda')
raw=torch.load(a.source,map_location='cpu',weights_only=False).double().eval()
model=create_model_from_mace(raw,head='default',mode='single').double().eval()
# Conversion's scalar transforms initialize float32; restore exact source dtype.
rescale = next(m for m in model.output_modules if isinstance(m,GlobalRescaleShift))
with torch.no_grad():
 rescale.scales[0].scale.copy_(raw.scale_shift.scale.reshape_as(rescale.scales[0].scale))
 rescale.shifts[0].shift.copy_(raw.scale_shift.shift.reshape_as(rescale.shifts[0].shift))
 for z, value in zip(raw.atomic_numbers.tolist(),raw.atomic_energies_fn.atomic_energies.reshape(-1)):
  rescale.atomic_shifts[0].values[z] = value

# Exact mapping audit: learned tensors in every interaction/product and embedding.
audit=[]
pairs=[('embedding',raw.node_embedding.linear,model.representation.embeddings.chemical_embedding.linear)]
pairs += [(f'interaction_{i}',x,model.representation.interactions[i]) for i,x in enumerate(raw.interactions)]
pairs += [(f'product_{i}',x,model.representation.products[i]) for i,x in enumerate(raw.products)]
for name,source,target in pairs:
 td=dict(target.named_parameters())
 for key,value in source.named_parameters():
  ok=key in td and value.shape==td[key].shape and torch.equal(value.cpu(),td[key].cpu())
  audit.append({'group':name,'key':key,'numel':value.numel(),'exact':ok})
assert all(x['exact'] for x in audit),[x for x in audit if not x['exact']]
model.to(device)
frames=read(a.data,':')
reader=AseDataReader(float(raw.r_max),return_cell_displacements=True,default_dtype=torch.float64)
def predict(atoms):
 d=reader(atoms)
 for key in (properties.energy,properties.forces,properties.stress,properties.virial): d.pop(key,None)
 d={k:v.to(device) for k,v in d.items()}
 return model(d)
from mace.calculators import MACECalculator
calc=MACECalculator(models=[raw.to(device)],device='cuda',default_dtype='float64')
checks=[]
for i in [0,1,2,3]:
 pred=predict(frames[i]); at=frames[i].copy(); at.calc=calc
 de=abs(float(pred[properties.energy].detach().cpu())-at.get_potential_energy())
 df=float(np.max(np.abs(pred[properties.forces].detach().cpu().numpy()-at.get_forces(apply_constraint=False))))
 checks.append({'index':i,'energy_abs_eV':de,'force_max_abs_eV_A':df})
 print('equivalence',checks[-1],flush=True)
 assert de < 1e-5 and df < 1e-6,checks[-1]
del calc,raw
model.cpu(); torch.save(model,out/'matpes_curator_uncalibrated.pt'); model.to(device)
train=torch.randperm(len(frames),generator=torch.Generator().manual_seed(a.seed))[:270].tolist()
elements=sorted(set(int(z) for at in frames for z in at.numbers))
rows=[]; counts=[]
for n,i in enumerate(train):
 pred=predict(frames[i]); pe=float(pred[properties.energy].detach().cpu()); te=frames[i].get_potential_energy()
 pf=pred[properties.forces].detach().cpu().numpy(); tf=frames[i].get_forces(apply_constraint=False)
 rows.append({'index':i,'n_atoms':len(frames[i]),'pred_energy':pe,'target_energy':te,'force_sq_sum':float(((pf-tf)**2).sum()),'force_count':tf.size})
 counts.append([(frames[i].numbers==z).sum() for z in elements])
 if n%30==0: print('train predictions',n,rows[-1],flush=True)
np_counts=np.asarray(counts,dtype=float)
y=np.array([r['target_energy']-r['pred_energy'] for r in rows])
# Fit per-atom E errors with minimum-norm composition offsets; all fitting is train-only.
natoms=np.array([r['n_atoms'] for r in rows])
offsets,_,rank,singular=np.linalg.lstsq(np_counts/natoms[:,None],y/natoms,rcond=None)
rescale=next(m for m in model.output_modules if isinstance(m,GlobalRescaleShift))
old=rescale.atomic_shifts[0].values.detach().cpu().clone()
with torch.no_grad():
 for z,value in zip(elements,offsets): rescale.atomic_shifts[0].values[z] += float(value)
calibration_delta=np_counts@offsets
stats={'source':a.source,'source_sha256':hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),'curator_class':str(type(model)),'hidden_irreps':str(model.representation.hidden_irreps),'parameters':sum(p.numel() for p in model.parameters()),'cutoff':float(model.representation.cutoff),'conversion_checks':checks,'learned_tensor_mapping':audit,'mapped_learned_numel':sum(x['numel'] for x in audit),'train_indices':train,'seed':a.seed,'calibration':'train-only minimum-norm composition energy offset; forces unchanged','elements':elements,'offset_eV_per_element':offsets.tolist(),'composition_rank':int(rank),'composition_singular_values':singular.tolist(),'train_energy_RMSE_eV_atom_before':float(np.sqrt(np.mean((y/natoms)**2))),'train_energy_RMSE_eV_atom_after':float(np.sqrt(np.mean(((y-calibration_delta)/natoms)**2))),'train_force_RMSE_eV_A':float(np.sqrt(sum(r['force_sq_sum'] for r in rows)/sum(r['force_count'] for r in rows))),'output_modules':[str(type(m)) for m in model.output_modules],'scale_buffers':{k:v.detach().cpu().tolist() for k,v in rescale.named_buffers() if 'values' not in k},'original_atomic_shifts':{str(z):float(old[z]) for z in elements}}
model.cpu(); torch.save(model,out/'matpes_curator_calibrated.pt')
(out/'initialization.json').write_text(json.dumps(stats,indent=2))
(out/'train_predictions.json').write_text(json.dumps(rows,indent=2))
print('COMPLETE',json.dumps({k:v for k,v in stats.items() if k not in ('learned_tensor_mapping','train_indices','scale_buffers')}),flush=True)
