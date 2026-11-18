"""Same-checkpoint physical scoring. Never use synthetic coordinates as E/F labels."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
from ase.io.trajectory import Trajectory

HERE=Path(__file__).resolve().parent
for name in ('dpa4c_tensor','dpa4c_fragment_sa','dpa4c_fragment_dimers','dpa4c_sa_scan'):
    sys.path.insert(0,str(HERE.parent/name))
from diagnose import geometry, mechanics
from rigid import data, describe
from train_runner import _read_only, _forward, primary_paths, inputs, sha, ROOT
from importlib.util import spec_from_file_location, module_from_spec
def imported(name, path):
    spec=spec_from_file_location(name,path); m=module_from_spec(spec); spec.loader.exec_module(m);return m
dimer=imported('dimer_physical',HERE.parent/'dpa4c_fragment_dimers/score.py')
revised=imported('sa_partition',HERE.parent/'dpa4c_fragment_sa/report.py').revised

def write(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def rmse(rows,k):
    return float(np.sqrt(np.mean([r[k]**2 for r in rows])))

def score_fragments(system, model, ck, out):
    out.mkdir(exist_ok=True,parents=True)
    src,sp,ref=(primary_paths if system=='sasa' else inputs)(system)
    split=json.loads(sp.read_text()); ident=json.loads(ref.read_text())
    assert sha(sp)==ident['split_sha256'] and sha(src)==ident['source_sha256']
    summary=dict(system=system,checkpoint_sha256=sha(ck),split_sha256=sha(sp),units='eV and Angstrom')
    rows=[]; curves=[]
    with _read_only(model,'cuda') as head:
        def calc(at,x,charge):
            y=_forward(model,head,data(at,x,charge),'cuda','proposal')
            return float(y['energy']),y['forces'].numpy()
        if system=='dimers':
            from ase import Atoms
            records,manifest=dimer.frames(ROOT/'datasets/longrange_benchmarks_20261009/raw',sp)
            summary['reference_manifest']=manifest
            for meta,x,z,mask,axis,e,f in records:
                at=Atoms(numbers=z,positions=x)
                ep,fp=calc(at,x,0.)
                pr=dimer.projections(x,z,mask,axis,fp)
                row=dict(**meta,E_pred=ep,F_radial=pr['radial_COM_eV_A'])
                if axis is not None:row['F_path']=pr['path_force_eV_A']
                if f is not None:
                    gt=dimer.projections(x,z,mask,axis,f)
                    row.update(E_ref=e,F_radial_ref=gt['radial_COM_eV_A'],radial_error=row['F_radial']-gt['radial_COM_eV_A'])
                    if axis is not None:row['F_path_ref']=gt['path_force_eV_A']
                (rows if meta['which']=='test' else curves).append(row)
            summary['curves']={}
            for number in (9,76):
                refs=[r for r in curves if r['scan']==number and r['which']=='reference']
                far=max(refs,key=lambda r:r['dij_A'])
                for r in curves:
                    if r['scan']!=number:continue
                    r['relative_E_pred']=r['E_pred']-far['E_pred']
                    if 'E_ref' in r:
                        r['relative_E_ref']=r['E_ref']-far['E_ref']
                        r['curve_error']=r['relative_E_pred']-r['relative_E_ref']
                        r['path_error']=r['F_path']-r['F_path_ref']
                summary['curves'][number]=dict(anchors=len(refs),relative_E_RMSE_eV=rmse(refs,'curve_error'),
                    F_path_RMSE_eV_A=rmse(refs,'path_error'),
                    well_depth_pred_eV=-min(r['relative_E_pred'] for r in refs),
                    well_depth_ref_eV=-min(r['relative_E_ref'] for r in refs))
        else:
            with Trajectory(str(src)) as tr:
                candidates=[]
                for i in sorted(split['test']):
                    at=tr[i]; x=at.positions; z=at.numbers; charge=float(at.info['total_charge'])
                    assert charge=={'sasa':0.,'sabs':-1.,'sadma':0.}[system]
                    _,fp=calc(at,x,charge); f=at.get_forces(apply_constraint=False)
                    g=geometry(z,x); rad=mechanics(fp,x,g)[2]; target=mechanics(f,x,g)[2]
                    row=dict(source_index=i,R_A=g[3],H_flag=g[-1],F_radial=rad,F_radial_ref=target,radial_error=rad-target,
                             outside_cutoff=bool(np.linalg.norm(x[:7,None]-x[None,7:],axis=-1).min()>5.5))
                    g2=revised(z,x)
                    if g2 is not None:row['reassigned_radial_error']=mechanics(fp,x,g2)[2]-mechanics(f,x,g2)[2]
                    rows.append(row)
                    desc=describe(at)
                    if desc is not None:candidates.append((float(desc[0]),i))
                candidates.sort(); summary['rigid_reference']='Only delta=0 is labelled; relative E uses delta=8 Angstrom.'
                for frac in (.25,.75):
                    _,idx=candidates[round(frac*(len(candidates)-1))];at=tr[idx];sep,axis=describe(at)
                    rr=[]
                    for delta in np.linspace(0,8,33):
                        x=at.positions.copy();x[7:]+=delta*axis
                        ep,fp=calc(at,x,float(at.info['total_charge']))
                        row=dict(source_index=idx,R_A=sep+delta,delta_A=delta,E_pred=ep,F_radial=mechanics(fp,x,geometry(at.numbers,x))[2])
                        if delta==0:row['F_radial_ref']=mechanics(at.get_forces(),x,geometry(at.numbers,x))[2]
                        rr.append(row)
                    for r in rr:r['relative_E_pred']=r['E_pred']-rr[-1]['E_pred']
                    curves.extend(rr)
            for key,subset,col in [('outside', [r for r in rows if r['outside_cutoff']],'radial_error'),
                                   ('reassigned',[r for r in rows if 'reassigned_radial_error' in r],'reassigned_radial_error')]:
                if subset:summary[key]=dict(n=len(subset),radial_RMSE_eV_A=rmse(subset,col))
            bins=[]
            for lo,hi in ((0,4),(4,6),(6,8),(8,12),(12,20),(20,1000)):
                a=[r for r in rows if lo<=r['R_A']<hi]
                if a:bins.append(dict(lo_A=lo,hi_A=hi,n=len(a),R_mean=np.mean([r['R_A'] for r in a]),
                         F_model=np.mean([r['F_radial'] for r in a]),F_reference=np.mean([r['F_radial_ref'] for r in a]),
                         radial_RMSE_eV_A=rmse(a,'radial_error')))
            write(out/'heldout_distance_bins.csv',bins)
    assert {r['source_index'] for r in rows}==set(split['test']) and len(rows)==len(split['test'])
    summary.update(heldout_frames=len(rows),radial_RMSE_eV_A=rmse(rows,'radial_error'))
    write(out/'heldout_radial.csv',rows);write(out/'binding_curves.csv',curves)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
