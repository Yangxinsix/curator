"""Read-only evaluation on released EFA DES370K scan coordinates, true anchors and test fold."""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, covalent_radii
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'dpa4c_tensor'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups(z, x):
    r = covalent_radii[z]
    dist = np.linalg.norm(x[:, None] - x, axis=2)
    splits = [connected_components(dist < k*(r[:, None]+r), directed=False) for k in (1.10, 1.18, 1.25)]
    assert all(n == 2 and np.array_equal(l, splits[0][1]) for n, l in splits), 'unvalidated connectivity'
    return splits[0][1] == 0


def projections(x, z, mask, axis, f):
    centers = [np.average(x[m], axis=0, weights=atomic_masses[z[m]]) for m in (mask, ~mask)]
    delta = centers[0]-centers[1]
    ef = {}
    net = [np.sum(f[m], axis=0) for m in (mask, ~mask)]
    torque = [np.cross(x[m]-o, f[m]).sum(0) for m, o in zip((mask, ~mask), centers)]
    for k, v in zip(('F_A','F_B','T_A','T_B'), net+torque):
        ef.update({k+'_'+j: float(t) for j, t in zip('xyz',v)})
    ef['radial_COM_eV_A'] = float((net[0]-net[1])@delta/(2*np.linalg.norm(delta)))
    if axis is not None:
        ef['path_force_eV_A'] = float(net[0]@axis)
    return ef


def frames(src, split_path):
    base=src/'efa_DES370K_subsampled_shifted.npz'; split=json.loads(split_path.read_text())
    with np.load(base) as b:
        X,Z,V,E,F=[b[k] for k in ('positions','atomic_numbers','node_mask','energy','forces')]
    fold = {int(i): k for k,ids in split.items() for i in ids}
    assert len(fold)==len(X) and len(split['test'])==112
    rows=[]; manifest={'base_sha256':digest(base),'split_sha256':digest(split_path),'scans':{}}
    for num in (9,76):
        raw=src/f'efa_{num}_fine_grid.npz'; fig=src/f'efa_figure_dimer-{num}-fine.npz'
        with np.load(raw) as g:
            x,z=g['positions'],g['atomic_numbers'][0];assert np.all(g['energy']==0) and np.all(g['forces']==0)
        with np.load(fig) as g:
            dij, dt, et = (g[k].ravel() for k in ('dij','dij_true','energy_true'))
        m=np.linalg.norm(x[-1]-x[0],axis=1)>0.1
        axis=np.mean(x[-1,m]-x[0,m],axis=0);axis/=np.linalg.norm(axis)
        assert np.array_equal(m,groups(z,x[-1]))
        a,b=np.where(m)[0],np.where(~m)[0]
        distances=np.linalg.norm(x[:,a,None]-x[:,None,b],axis=-1)
        err=np.max(np.abs(distances-dij[:,None,None]),axis=0)
        ia, ib=np.unravel_index(err.argmin(),err.shape);i,j=a[ia],b[ib]
        assert err[ia,ib]<2e-6 and (err<2e-6).sum()==1
        candidates=np.where((V.sum(1)==len(z))&(Z[:,:len(z)]==z).all(1))[0]
        refs=[]; phase=np.mean((x[:,m]-x[0,m])@axis,axis=1)
        for kind, target in [('grid',dij),('reference',dt)]:
            for l,v in enumerate(target):
                idx=None
                if kind=='grid':
                    y=x[l]; forces=None; e=None
                else:
                    ds=np.linalg.norm(X[candidates,i]-X[candidates,j],axis=1)
                    match=candidates[abs(ds-v)<3e-6]; assert len(match)==1
                    idx=int(match[0]); refs.append(idx);y=X[idx][V[idx]].astype(float)
                    forces=F[idx][V[idx]].astype(float); e=float(E[idx,0])
                u=np.mean((y[m]-x[0,m])@axis)
                assert max(np.max(abs(y[m]-x[0,m]-u*axis)),np.max(abs(y[~m]-x[0,~m])))<2e-5
                rows.append((dict(which=kind,scan=num,point=l,source_index=idx,fold=fold.get(idx),dij_A=float(v),
                    path_A=float(u)),y,z,m,axis,e,forces))
        assert len(set(refs))==len(refs) and np.ptp(E[refs,0]-et)<1e-8
        assert abs(max(dt)-max(dij))<2e-5
        manifest['scans'][num]=dict(fragment_A=a.tolist(),fragment_B=b.tolist(),pair=[int(i),int(j)],
            formulas=[Atoms(numbers=z[m_]).get_chemical_formula() for m_ in (m,~m)],
            translation_unit_vector=axis.tolist(),counts={k:sum(fold[i]==k for i in refs) for k in split},
            records=len(refs),grid=len(x),source_sha256=digest(raw),figure_sha256=digest(fig))
    for i in split['test']:
        x=X[i][V[i]].astype(float); z=Z[i][V[i]]
        rows.append((dict(which='test',scan='test',point=int(i),source_index=int(i),fold='test'),x,z,groups(z,x),None,float(E[i,0]),F[i][V[i]]))
    return rows,manifest


def write_csv(path, rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader();w.writerows(rows)


def score(args):
    import torch
    from train_runner import load_tensor_checkpoint, build, _read_only, _forward
    from extension_molecular import physical_ef, p
    from scan import boundaries
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    records,manifest=frames(args.source,args.split)
    args.out.mkdir(exist_ok=True,parents=True)
    refrows=[dict(**meta,energy_eV=e,**projections(x,z,m,axis,f)) for meta,x,z,m,axis,e,f in records if f is not None]
    write_csv(args.out/'reference.csv',refrows)
    path=args.ckpts/(args.arm+'.pt'); st=torch.load(path,weights_only=False,map_location='cpu')
    expected={'pure20':'pure','acvr20':'acvr','les20':'les','old40':'old','orient40':'orient2'}[args.arm]
    ident=st['identity']; assert (ident['system'],ident['arm'],ident['seed'],ident['split_sha256'])==('dimers',expected,3407,manifest['split_sha256'])
    assert st['joint_step']==(40000 if args.arm.endswith('40') else 20000)
    manifest.update(arm=args.arm,checkpoint_sha256=digest(path),joint_step=st['joint_step'])
    output=args.out/(args.arm+'.csv')
    receipt=args.out/(args.arm+'_complete.json')
    if receipt.is_file() and output.is_file():
        last=json.loads(receipt.read_text())
        if (last['checkpoint']==manifest['checkpoint_sha256'] and last['csv']==digest(output)
                and json.loads((args.out/(args.arm+'.json')).read_text())==json.loads(json.dumps(manifest))):
            return
    (args.out/(args.arm+'.json')).write_text(json.dumps(manifest,indent=2))
    if args.arm.endswith('40'):
        model=load_tensor_checkpoint(path)
    else:
        model=build('dimers',['H','C','N','O','S'],expected,3407,'B2').double().cuda().eval()
        model.load_state_dict(st['model'],strict=True)
    is_acvr=expected not in ('pure','les');pred=[]; mono={}
    with (_read_only(model,'cuda') if is_acvr else nullcontext(None)) as head:
        def calc(x,z,mode='proposal'):
            bc=boundaries();bc['condition_state_ptr']=torch.tensor([0,len(x)])
            batch={p.positions:torch.as_tensor(x.copy()).double().requires_grad_(True),p.atomic_numbers:torch.tensor(z,dtype=torch.long),
                p.image_idx:torch.zeros(len(x),dtype=torch.long),p.n_atoms:torch.tensor([len(x)]),p.cell:torch.zeros(3,3),
                p.pbc:torch.zeros(1,3,dtype=torch.bool),p.total_charge:torch.zeros(1),'variational':bc}
            if is_acvr:
                ans=_forward(model,head,batch,'cuda',mode)
                return ans['energy'],ans['forces'].numpy()
            from curator.train.train import tree_to_device
            e,f,_=physical_ef(model,tree_to_device(batch,'cuda'),method=expected,create_graph=False)
            return e.detach().item(),f.detach().cpu().numpy()
        modes=('proposal','uniform') if args.arm in ('old40','orient40') else ('proposal',)
        from fixed_q_batch import panel_energy_charge
        for mode in modes:
            values={}
            for count in sorted({len(z) for _,_,z,*rest in records}):
                ids=[i for i,rec in enumerate(records) if len(rec[2])==count]
                for start in range(0,len(ids),12):
                    idx=ids[start:start+12];b=len(idx)
                    xx=torch.as_tensor(np.concatenate([records[i][1] for i in idx]),device='cuda').double().requires_grad_(True)
                    zz=torch.as_tensor(np.concatenate([records[i][2] for i in idx]),device='cuda').long()
                    data={p.positions:xx,p.atomic_numbers:zz,p.n_atoms:torch.full((b,),count,device='cuda',dtype=torch.long),
                          p.image_idx:torch.arange(b,device='cuda').repeat_interleave(count),
                          p.cell:xx.new_zeros(b*3,3),p.pbc:torch.zeros(b,3,device='cuda',dtype=torch.bool),
                          p.total_charge:xx.new_zeros(b),'variational':{'condition_geometry_index':torch.arange(b,device='cuda'),
                          'boundary_spec':boundaries()['boundary_spec']*b}}
                    if is_acvr:
                        for layer in model.input_modules:data=layer(data)
                        data=model.representation(data)
                        def zero_charge(drive, *args):return torch.zeros_like(drive).reshape(b,count,1)
                        with (patch('fixed_q_batch._readout_charge',zero_charge) if mode=='uniform' else nullcontext()):
                            energy,_=panel_energy_charge(head,data)
                        force,=torch.autograd.grad(-energy.sum(),xx)
                    else:
                        energy,force,_=physical_ef(model,data,method=expected,create_graph=False)
                    e=energy.detach().cpu().numpy().reshape(-1); ff=force.detach().cpu().numpy().reshape(b,count,3)
                    if start==0:
                        # Check atom indexing, model mode and the force derivative against the production path.
                        one=calc(records[idx[0]][1],records[idx[0]][2],mode)
                        np.testing.assert_allclose(e[0],one[0],rtol=0,atol=2e-7)
                        np.testing.assert_allclose(ff[0],one[1],rtol=0,atol=2e-6)
                    values.update({i:(float(ee),f) for i,ee,f in zip(idx,e,ff,strict=True)})
            for k,(meta,x,z,m,axis,_,__) in enumerate(records):
                e,f=values[k];v=dict(**meta,mode=mode,energy_eV=e,**projections(x,z,m,axis,f))
                if axis is not None:
                    key=(meta['scan'],mode)
                    if key not in mono:mono[key]=sum(calc(x[which],z[which],mode)[0] for which in (m,~m))
                    v['interaction_neutral_monomers_eV']=e-mono[key]
                pred.append(v)
            print(args.arm,mode,len(values),flush=True)
    pending=output.with_suffix('.tmp');write_csv(pending,pred);pending.replace(output)
    receipt.write_text(json.dumps({'checkpoint':manifest['checkpoint_sha256'],'csv':digest(output),'rows':len(pred)}))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('source','split','ckpts','out'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--arm',required=True,choices=['pure20','acvr20','les20','old40','orient40'])
    score(ap.parse_args())
