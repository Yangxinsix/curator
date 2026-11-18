"""Read-only paired SA fragment mechanics on the original full held-out folds."""
import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase.data import atomic_masses
from ase.io.trajectory import Trajectory

HERE = Path(__file__).resolve().parent
ALLOWED = ((1, 1, 8, 8, 8, 8, 16, 1, 1, 8, 8, 8, 8, 16),
           (1, 1, 8, 8, 8, 8, 16, 16, 8, 8, 8, 8, 1),
           (1, 1, 8, 8, 8, 8, 16, 7, 1, 6, 1, 1, 1, 6, 1, 1, 1))


def geometry(z, xyz):
    """Source-order fixed fragments, centers of mass, 1->2 axis, distance and H flag."""
    z = np.asarray(z)
    if tuple(z) not in ALLOWED:
        raise ValueError(f"Unexpected SA block identities {z}")
    first = np.arange(len(z)) < 7
    masks = (first, ~first)
    centers = np.stack([np.average(xyz[m], axis=0, weights=atomic_masses[z[m]]) for m in masks])
    r = centers[1] - centers[0]
    length = np.linalg.norm(r)
    if length < 1.e-6:
        raise ValueError("Degenerate separation")
    hyd = np.flatnonzero(z == 1)
    ds = np.column_stack([np.linalg.norm(xyz[hyd,None]-xyz[m & np.isin(z, [6,7,8])],axis=-1).min(axis=1) for m in masks])
    owner = (~first[hyd]).astype(int)
    own, other = ds[np.arange(len(hyd)),owner], ds[np.arange(len(hyd)),1-owner]
    # A geometry heuristic, not a DFT protonation label. Outside bonding distance
    # do not force a proton assignment even when the nearest center is clear.
    ambiguous = (np.abs(own-other) < .15) | (np.minimum(own,other) > 1.35)
    crossed = (other + .15 < own) & (other < 1.35)
    flag = ("shared_or_unassigned" if ambiguous.any() else
            "crossed" if crossed.any() else "stable")
    return masks, centers, r/length, float(length), flag


def mechanics(f, xyz, geom):
    masks, centers, axis, _, _ = geom
    net = np.stack([f[m].sum(0) for m in masks])
    tau = np.stack([np.cross(xyz[m]-c, f[m]).sum(0) for m,c in zip(masks,centers)])
    # Force conjugate to +R under translation by -R/2 and +R/2.
    fr = .5 * np.dot(net[1]-net[0],axis)
    return net,tau,float(fr)


def force_stats(rows):
    ret = dict(n=len(rows))
    for k in ("net_mse","torque_mse","radial_sq","atom_mse"):
        ret[k.replace("mse","rmse").replace("sq","rmse")] = np.sqrt(np.mean([r[k] for r in rows]))
    for k in ("R_A","radial_ref","radial_pred","dE_per_atom_meV"):
        ret[k+"_mean"] = float(np.mean([r[k] for r in rows]))
    return ret


def run(args):
    import torch
    sys.path.insert(0,str(HERE.parent/'dpa4c_tensor'))
    from train_runner import load_tensor_checkpoint, configure, primary_paths, inputs, sha
    from equilibrium import _read_only, _forward, model_inputs
    from curator.data import properties as p
    from curator.data.variational_data import canonical_boundary_hash
    from curator.variational.variables.charge_boundary import fixed_total_charge_recipe
    import fixed_q_batch as fb

    configure()
    src,sp,er=(primary_paths if args.system=="sasa" else inputs)(args.system)
    ref=json.loads(er.read_text()); fold=json.loads(sp.read_text())
    if sha(src)!=ref["source_sha256"] or sha(sp)!=ref["split_sha256"]:
        raise ValueError("Source digest differs")
    ids=sorted(map(int,fold["test"]))
    if not len(ids)==len(set(ids))==2000: raise ValueError("Unexpected test fold")
    if args.limit:
        ids=[ids[i] for i in np.linspace(0,len(ids)-1,args.limit,dtype=int)]
    with Trajectory(str(src)) as tr:
        frames=[tr[i] for i in ids]
    geom=[geometry(a.numbers,a.positions) for a in frames]
    z=frames[0].numbers;n=len(z); total=frames[0].info["total_charge"]
    if total != {"sasa":0.,"sabs":-1.,"sadma":0.}[args.system] or any(
            not np.array_equal(a.numbers,z) or a.info["total_charge"]!=total or a.pbc.any() for a in frames):
        raise ValueError("Mixed identities, PBC or charges")
    off=sum(ref["per_element_eV"][str(v)] for v in z)
    spec=asdict(fixed_total_charge_recipe(total));bh=canonical_boundary_hash(spec)

    def make_batch(atoms):
        b=len(atoms); ix=torch.arange(b)
        return {p.positions:torch.from_numpy(np.concatenate([a.positions for a in atoms])),p.pbc:torch.zeros((b,3),dtype=torch.bool),
                p.cell:torch.zeros((b,3,3)),p.n_atoms:torch.full((b,),n),p.atomic_numbers:torch.from_numpy(np.tile(z,b)),
                p.image_idx:ix.repeat_interleave(n),p.total_charge:torch.full((b,),float(total)),
                "variational":dict(condition_geometry_index=ix,boundary_spec=[spec]*b,boundary_hash=[bh]*b,
                                   condition_state_ptr=torch.arange(b+1)*n,condition_ptr=torch.arange(b+1),
                                   condition_weight=torch.ones(b))}

    def keep_uniform(drive,factors,total_charge,fixed_charge,eps):
        return torch.ones_like(drive)*(total_charge[:,None,None]/n)

    args.output.mkdir(parents=True,exist_ok=True)
    source_sha=ref["source_sha256"]
    for arm in args.arms:
        ck=args.weights/f"{args.system}-{arm}.pt"
        ident=torch.load(ck,map_location="cpu",weights_only=False)
        if (ident["joint_step"],ident["identity"]["system"],ident["identity"]["arm"],ident["identity"]["split_sha256"])!=(40000,args.system,arm,ref["split_sha256"]):
            raise ValueError("Wrong checkpoint")
        del ident
        m=load_tensor_checkpoint(ck); records=[]; parity={}; raw={'proposal':[],'uniform':[]}
        with _read_only(m,"cuda") as head:
            for i in range(0,len(frames),args.batch):
                atoms=frames[i:i+args.batch]; batch=make_batch(atoms)
                d=model_inputs(batch,"cuda");d[p.positions].requires_grad_(True)
                for layer in m.input_modules: d=layer(d)
                d=m.representation(d)
                for mode in ('proposal','uniform'):
                    if mode=='proposal':
                        energy,charge=fb.panel_energy_charge(head,d)
                    else:
                        with patch.object(fb,"_readout_charge",keep_uniform):
                            energy,charge=fb.panel_energy_charge(head,d)
                    force,=torch.autograd.grad(-energy.sum(),d[p.positions],retain_graph=(mode=='proposal'))
                    predF=force.detach().cpu().numpy().reshape(len(atoms),n,3)
                    predE=energy.detach().cpu().numpy()
                    predq=charge.detach().cpu().numpy().reshape(len(atoms),n)
                    raw[mode].append(predF.copy())
                    if not i:
                        native=_forward(m,head,make_batch([atoms[0]]),"cuda",mode)
                        parity[mode]=max(abs(predF[0]-native["forces"].numpy()).max(),abs(predE[0]-native["energy"]))
                        if parity[mode]>1e-6:raise ValueError(f"Batch / production discrepancy {parity}")
                    for k,a in enumerate(atoms):
                        g=geom[i+k]; xyz=a.positions; target=a.get_forces(apply_constraint=False)
                        n0,t0,r0=mechanics(target,xyz,g); n1,t1,r1=mechanics(predF[k],xyz,g)
                        row=dict(source_index=ids[i+k],mode=mode,H_flag=g[-1],R_A=g[3],
                                 net_mse=np.mean((n1-n0)**2),torque_mse=np.mean((t1-t0)**2),radial_sq=(r1-r0)**2,
                                 atom_mse=np.mean((predF[k]-target)**2),radial_ref=r0,radial_pred=r1,
                                 dE_per_atom_meV=1000*(predE[k]+off-a.get_potential_energy())/n,
                                 qA_e=predq[k,:7].sum(),qB_e=predq[k,7:].sum())
                        for key, vec in (('net_reference',n0),('net_model',n1),('torque_reference',t0),('torque_model',t1)):
                            for frag, vv in zip(('A','B'),vec):
                                for ax,v in zip('xyz',vv):row[f'{key}_{frag}{ax}']=v
                        records.append(row)
        dst=args.output/f"{args.system}-{arm}"
        with dst.with_suffix('.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
        np.savez_compressed(dst.with_suffix('.npz'),source_index=ids,z=z,
                            positions=np.stack([a.positions for a in frames]),forces_ref=np.stack([a.get_forces() for a in frames]),
                            F_proposal=np.concatenate(raw['proposal']),F_uniform=np.concatenate(raw['uniform']))
        stats={}
        for mode in ('proposal','uniform'):
            rows=[r for r in records if r['mode']==mode]
            stats[mode]={'all':force_stats(rows),'H':{},'R':{}}
            for flag in ('stable','crossed','shared_or_unassigned'):
                rs=[r for r in rows if r['H_flag']==flag]
                if rs:stats[mode]['H'][flag]=force_stats(rs)
            for lo,hi in ((0,4),(4,6),(6,8),(8,12),(12,20),(20,np.inf)):
                rs=[r for r in rows if lo<=r['R_A']<hi]
                if rs:stats[mode]['R'][f'{lo}-{hi}']=force_stats(rs)
        out=dict(source_sha256=source_sha,split_sha256=ref["split_sha256"],checkpoint_sha256=sha(ck),
                 seed=3407,joint_step=40000,frames=len(ids),fold='test',full_fold=not args.limit,
                 units={'net_rmse':'eV/A per Cartesian component per fragment','torque_rmse':'eV per Cartesian component per fragment',
                        'radial_rmse':'eV/A signed conjugate to COM separation, positive = repulsive','atom_rmse':'eV/A',
                        'dE_per_atom_meV_mean':'signed prediction error, not interaction energy'},
                 single_frame_parity_max=parity,stats=stats)
        dst.with_suffix('.json').write_text(json.dumps(out,allow_nan=False,indent=2)+'\n')
        print(args.system,arm,len(ids),stats['proposal']['all'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--system',choices=['sasa','sabs','sadma'],required=True)
    ap.add_argument('--weights',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--arms',nargs='+',choices=['old','qk2','orient2'],default=['old','orient2','qk2'])
    ap.add_argument('--limit',type=int);ap.add_argument('--batch',type=int,default=32)
    run(ap.parse_args())
