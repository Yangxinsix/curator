"""Pair scored fragments by source index, compare their radial forces and binding profiles."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def records(p):
    with p.open() as f:return list(csv.DictReader(f))
def rmse(r):return float(np.sqrt(np.mean(np.square(r))))
def report(root,system):
    runs=[root/system/a for a in ('linear','stf_frame')]
    metas=[json.loads((p/'config.json').read_text())['identity'] for p in runs]
    for key in ('source_checkpoint_sha256','source_resume_sha256','split_sha256','budget','seed'):
        assert metas[0][key]==metas[1][key]
    for p in runs:
        assert json.loads((p/'complete.json').read_text())['checkpoint_sha256']==json.loads((p/'fragments/summary.json').read_text())['checkpoint_sha256']
    rr=[records(p/'fragments/heldout_radial.csv') for p in runs]
    rr=[sorted(r,key=lambda row:int(row['source_index'])) for r in rr]
    assert [r['source_index'] for r in rr[0]]==[r['source_index'] for r in rr[1]]
    ref=np.array([float(r['F_radial_ref']) for r in rr[0]])
    np.testing.assert_allclose(ref,[float(r['F_radial_ref']) for r in rr[1]],atol=1e-9,rtol=0)
    err=np.array([[float(r['radial_error']) for r in g] for g in rr])
    score={'frames':len(ref),'paired_radial_RMSE_meV_A':dict(zip(('linear','stf'),[1000*rmse(x) for x in err])),
           'mean_paired_change_abs_radial_error_meV_A':float(1000*(abs(err[1])-abs(err[0])).mean()),
           'validation':{},'test':{},'fragment_diagnostics':{}}
    for name,p in zip(('linear','stf'),runs):
        score['fragment_diagnostics'][name]=json.loads((p/'fragments/summary.json').read_text())
        for fold in ('validation','test'):score[fold][name]=json.loads((p/(fold+'_final.json')).read_text())['summary']['proposal']
    curves=[records(p/'fragments/binding_curves.csv') for p in runs]
    dim=system=='dimers';key='scan' if dim else 'source_index'; axis='dij_A' if dim else 'R_A'
    figs,ax=plt.subplots(2,2,figsize=(8,5.4),layout='constrained')
    for j,id_ in enumerate(sorted({r[key] for r in curves[0]})):
        for nm,color,g in zip(('Original','STF'),('#D55E00','#0072B2'),curves):
            sub=[r for r in g if r[key]==id_ and (not dim or r['which']=='grid')]
            x=[float(r[axis]) for r in sub]
            ax[0,j].plot(x,[float(r['relative_E_pred']) for r in sub],label=nm,color=color)
            ax[1,j].plot(x,[float(r['F_path' if dim else 'F_radial']) for r in sub],color=color)
        anchors=[r for r in curves[0] if r[key]==id_ and (r.get('which')=='reference' if dim else float(r['delta_A'])==0)]
        if dim:ax[0,j].scatter([float(r[axis]) for r in anchors],[float(r['relative_E_ref']) for r in anchors],s=12,color='black',label='Reference')
        else:ax[0,j].text(.03,.07,'Model-only pull, zero at +8 Å',transform=ax[0,j].transAxes,fontsize=8)
        ax[1,j].scatter([float(r[axis]) for r in anchors],[float(r['F_path_ref' if dim else 'F_radial_ref']) for r in anchors],s=12,color='black')
        ax[0,j].set_title(('Dimer ' if dim else 'Test frame ')+id_);ax[1,j].set_xlabel(('Pair distance' if dim else 'COM distance')+' (Å)')
    ax[0,0].legend(frameon=False);ax[0,0].set_ylabel('Relative energy (eV)');ax[1,0].set_ylabel(('Path force' if dim else 'Radial force')+' (eV/Å)')
    dest=root/'comparison';dest.mkdir(exist_ok=True);figs.savefig(dest/(system+'-curves.png'),dpi=180);plt.close(figs)
    (dest/(system+'.json')).write_text(json.dumps(score,indent=2)+'\n')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
    for system in ('sasa','sabs','sadma','dimers'):report(args.root,system)
