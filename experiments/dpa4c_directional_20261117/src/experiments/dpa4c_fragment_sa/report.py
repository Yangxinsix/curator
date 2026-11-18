"""Summarize paired SA results and expose sensitivity to the definition of a fragment."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ase.data import atomic_masses

from diagnose import geometry, mechanics

ROOT = Path(__file__).resolve().parent / 'observations/40k'
SYSTEMS = ('sasa','sabs','sadma')
ARMS = ('old','qk2','orient2')
EDGES = (0,4,6,8,12,20,np.inf)


def revised(z,x):
    m,c,u,r,flag = geometry(z,x)
    if flag=='shared_or_unassigned':
        return None
    sulfur=np.flatnonzero(z==16); oc=np.flatnonzero(z==8)
    if len(sulfur)==2:
        ds=np.linalg.norm(x[oc,None]-x[sulfur],axis=-1)
        if np.any(ds.argmin(1)!=(oc>=7)):return None
    elif np.any(np.linalg.norm(x[oc]-x[sulfur[0]],axis=1)>2.2):
        return None
    h=np.flatnonzero(z==1)
    distances=np.stack([np.linalg.norm(x[h,None]-x[a & np.isin(z,[6,7,8])],axis=-1).min(1) for a in m],axis=1)
    first=m[0].copy();first[h]=distances[:,0]<distances[:,1]
    masks=(first,~first)
    cen=np.stack([np.average(x[a],axis=0,weights=atomic_masses[z[a]]) for a in masks])
    axis=cen[1]-cen[0]; rr=np.linalg.norm(axis)
    return masks,cen,axis/rr,float(rr),flag


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def sensitivity(path):
    z=np.load(path);out={}
    geom=[revised(z['z'],p) for p in z['positions']]
    fixed=[geometry(z['z'],p) for p in z['positions']]
    for mode in ('proposal','uniform'):
        sums={};f0=z['forces_ref'];fp=z['F_'+mode];positions=z['positions']
        for i,g in enumerate(geom):
            if g is None:continue
            a=mechanics(f0[i],positions[i],g); b=mechanics(fp[i],positions[i],g)
            err=[np.mean((v-w)**2) for v,w in zip(a,b)]
            for key in ('all_confident',g[-1]):
                sums.setdefault(key,[]).append(err)
            a=mechanics(f0[i],positions[i],fixed[i]); b=mechanics(fp[i],positions[i],fixed[i])
            err=[np.mean((v-w)**2) for v,w in zip(a,b)]
            for key in ('fixed_same_confident','fixed_'+g[-1]):
                sums.setdefault(key,[]).append(err)
        out[mode]={k:dict(n=len(v),**dict(zip(('net_rmse','torque_rmse','radial_rmse'),np.sqrt(np.mean(v,axis=0))))) for k,v in sums.items()}
    return out


def main():
    run=json.loads((ROOT.parents[2]/'dpa4c_tensor/observations/astra_review_1116.json').read_text())
    table=[];bins=[];checks={};sens={}
    fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    for icol,s in enumerate(SYSTEMS):
        for arm in ARMS:
            stem=ROOT/(s+'-'+arm)
            pred=json.loads(stem.with_suffix('.json').read_text());source=run['results'][s+'/'+arm]['test_final']
            checks[s+'/'+arm]={}
            data=list(csv.DictReader(stem.with_suffix('.csv').open()))
            for mode in ('proposal','uniform'):
                rows=[r for r in data if r['mode']==mode]
                assert len(rows)==len({r['source_index'] for r in rows})==2000
                truth=source[mode];meta=pred['stats'][mode]['all']
                matches=dict(energy=abs(rms([float(r['dE_per_atom_meV']) for r in rows])-truth['energy_RMSE_meV_atom']),
                             force=abs(1000*meta['atom_rmse']-truth['force_RMSE_meV_A']))
                if max(matches.values())>1e-4:raise ValueError((stem,mode,matches))
                checks[s+'/'+arm][mode]=matches
                table.append(dict(system=s,arm=arm,mode=mode,n=len(rows),net_meV_A=1000*meta['net_rmse'],
                                  torque_meV=1000*meta['torque_rmse'],radial_meV_A=1000*meta['radial_rmse']))
                rr=np.array([float(r['R_A']) for r in rows]); f=np.array([float(r['radial_ref']) for r in rows])*1000
                v=np.array([float(r['radial_pred']) for r in rows])*1000
                y=[(rr>=lo)&(rr<hi) for lo,hi in zip(EDGES,EDGES[1:])]
                for lo,hi,cut in zip(EDGES,EDGES[1:],y):
                    if cut.sum()==0:continue
                    bins.append(dict(system=s,arm=arm,mode=mode,lo_A=lo,hi_A=hi,count=int(cut.sum()),
                                     R_mean_A=rr[cut].mean(),radial_ref_meV_A=f[cut].mean(),radial_pred_meV_A=v[cut].mean(),
                                     radial_reference_rms_meV_A=rms(f[cut]),radial_error_rmse_meV_A=rms((v-f)[cut])))
                if (arm,mode) not in (('old','proposal'),('old','uniform'),('orient2','proposal')):continue
                # Singleton and low-occupancy tails stay in bins.csv without being connected to well-sampled bins.
                keep=[c for c in y if c.sum()>=10]
                xx=[rr[c].mean() for c in keep]
                color={'old-proposal':'#265a80','old-uniform':'#ae6f13','orient2-proposal':'#157f78'}[arm+'-'+mode]
                label=arm if mode=='proposal' else 'old: Q/N'
                style= '-' if arm=='old' and mode=='proposal' else ('--' if mode=='uniform' else ':')
                axs[0,icol].plot(xx,[v[c].mean() for c in keep],ls=style,marker='.',label=label,color=color)
                axs[1,icol].plot(xx,[rms((v-f)[c]) for c in keep],ls=style,marker='.',label=label,color=color)
                if arm=='old' and mode=='proposal':
                    axs[0,icol].plot(xx,[f[c].mean() for c in keep],color='black',marker='x',label='DFT',lw=1)
                    axs[1,icol].plot(xx,[rms(f[c]) for c in keep],color='black',marker='x',label='zero-force error',lw=1)
            raw=stem.with_suffix('.npz')
            if raw.exists():sens[s+'/'+arm]=sensitivity(raw)
        axs[0,icol].set_title(('SA-SA','SA-BS','SA-DMA')[icol])
        axs[0,icol].set_yscale('symlog',linthresh=1)
        axs[1,icol].set_yscale('log')
        axs[1,icol].set_xlabel('Fragment COM separation (Å)')
        for ax in axs[:,icol]:ax.grid(axis='y',alpha=.17)
    axs[0,0].set_ylabel('Mean signed radial force (meV/Å)')
    axs[1,0].set_ylabel('Radial force RMSE (meV/Å)')
    for a in axs[:,0]:a.legend(frameon=False,fontsize=8)
    for nm,rows in [('summary',table),('distance_bins',bins)]:
        with (ROOT/(nm+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    fig.savefig(ROOT/'radial.png',dpi=190);fig.savefig(ROOT/'radial.pdf')
    (ROOT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    (ROOT/'proton_partition.json').write_text(json.dumps(sens,indent=2)+'\n')
    text=[
        "# SA fragment force and torque at joint update 40,000",
        "",
        "All 2,000 distinct held-out DFT geometries per system; seed 3407; paired checkpoints. "
        "The uniform column is the counterfactual Q/N of the same old ACVR, including Q=-1 for SA-BS. "
        "Positive radial force pushes the two fragments apart. The split follows the source atom order: "
        "fragment A is the first seven (H2SO4) atoms. Net forces and torques about each fragment's own mass center "
        "are scored per Cartesian component over both fragments; radial force is half the difference of "
        "the two net forces projected onto the A-to-B center-of-mass axis. No charge labels were used.",
        "",
        "| System | Arm | Mode | Net force RMSE (meV/Å) | Radial RMSE (meV/Å) | Torque RMSE (meV) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in table:
        if row['arm']=='old' or row['mode']=='proposal':
            text.append("| {system} | {arm} | {mode} | {net_meV_A:.2f} | {radial_meV_A:.2f} | {torque_meV:.2f} |".format(**row))
    text+=["", "Natural configurations are neither matched in angle nor identical in intramolecular geometry. "
           "distance_bins.csv and radial.png show conditional force statistics, not a rigid or relaxed binding-energy scan. "
           "The zero-force baseline is the RMS reference radial force within each bin; a small absolute residual alone is not evidence of a correct tail.",
           "", "proton_partition.json repeats the mechanical projection after assigning unambiguous H to the nearest "
           "C/N/O heavy-atom fragment. Exclude a frame for this sensitivity check when any H has an interfragment "
           "distance margin below 0.15 Å or nearest distance above 1.35 Å, or an O has an inconsistent sulfur assignment. "
           "fixed_same_confident uses exactly those same frames without reassigning H. Source-order predictions in summary.csv "
           "use all frames, including shared and transferred protons. These are geometric diagnostics, not assigned electronic states.",
           "", "validation.json compares all 18 E/F aggregates to the original independent test scores. The first geometry "
           "of every model was also compared directly to the original single-structure ACVR in both charge modes. "
           "Individual CSVs retain forces and torques on both fragments, radial projections and charge sums; NPZs retain full reference and model atomic forces."]
    (ROOT/'REPORT.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
