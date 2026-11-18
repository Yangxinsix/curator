"""Fork the trained Cumulene orient2 model and optimizer at joint step 40,000."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dpa4c_tensor"))
import numpy as np
import torch
import train_runner as old
from readout import attach
from scan import score_scan


START = 40000
KINDS = ("linear", "quadratic", "mlp", "chebyshev_z", "stf_frame")


def source(args, ds, split_sha):
    ck = args.parent / "ckpt-020000.pt"
    cp = torch.load(ck, weights_only=False, map_location="cpu")
    resume = args.parent / "resume.pt"
    st = torch.load(resume, weights_only=False, map_location="cpu")
    sid = cp["identity"]
    fingerprint = old.sha(ck)
    if (sid != st['identity'] or st['step'] != 20000 or cp['joint_step'] != START or
        (sid["system"], sid["arm"], sid["seed"], sid["split_sha256"]) !=
        ("cumulene", "orient2", old.SEED, split_sha) or
        old.sha(ck) != json.loads((args.parent / "test_final.json").read_text())["checkpoint_sha256"] or
        list(ds.sources[0].species) != sid["species"]):
        raise ValueError("Parent identity does not match the scored Cumulene orient2 40k endpoint")
    if cp['model'].keys() != st['model'].keys() or any(
        not torch.equal(v, st['model'][k]) for k,v in cp['model'].items() if torch.is_tensor(v)):
        raise ValueError("Parent model differs from its optimizer resume")
    ident = dict(system="cumulene", arm=args.arm, seed=old.SEED, budget=args.steps,
                 source_joint_step=START, source_checkpoint_sha256=fingerprint,
                 source_resume_sha256=old.sha(resume), split_sha256=split_sha, species=sid["species"])
    model = old.make_old("cumulene", sid['species']).cuda().float()
    trainer = old.ScratchTrainer(model, backbone_lr=1e-4, energy_lr=1e-5, proposal_lr=1e-4,
                                 energy_weight=200., force_weight=200.)
    adam, soap = trainer.physics_optimizer, trainer.solver_optimizer
    def add(params):
        adam.param_groups[1]['params'].extend(params)
        trainer.energy.extend(params)
        trainer.physical = trainer.energy + trainer.backbone
    # Restore the *exact* historical order, including the orient2 slots appended at 20k.
    old.install_variant(model, "orient2", seed=old.SEED)
    add(list(model.output_modules[0].internal_energy.tensor_branch.parameters()))
    model.load_state_dict(cp["model"], strict=True)
    adam.load_state_dict(st['adam'])
    soap.load_state_dict(st['soap'])
    fresh = attach(model, args.arm)
    add(fresh)
    if sorted(map(id,trainer.physical)) != sorted(id(p) for g in adam.param_groups for p in g["params"]):
        raise ValueError("Optimizer and physical training groups differ")
    for p,v in adam.state.items():
        if 'exp_avg' in v and v['exp_avg'].shape != p.shape:
            raise ValueError("Parent Adam moment shape mismatch")
    return model, trainer, ident, fresh


def fit(args, ds, split_sha, out):
    model, trainer, ident, fresh = source(args, ds, split_sha)
    adam, soap = trainer.physics_optimizer, trainer.solver_optimizer
    step, elapsed = 0, 0.
    if (out/"resume.pt").is_file():
        state = torch.load(out/"resume.pt", map_location='cpu', weights_only=False)
        if state['identity'] != ident:
            raise ValueError("Readout resume identity mismatch")
        model.load_state_dict(state['model'])
        adam.load_state_dict(state['adam'])
        soap.load_state_dict(state['soap'])
        step, elapsed = state['step'], state['seconds']
        del state
    def snap(name, n, secs):
        old.atomic_save(dict(identity=ident, model=model.state_dict(), joint_step=START+n,
                             added_steps=n, training_seconds=secs), out/name)
    if not step:
        snap('ckpt-000000.pt',0,0.)
    if step == args.steps and (out/'done.json').exists():
        return
    old.write_json(out/'config.json', dict(identity=ident, batch=old.BATCH, new_params=sum(p.numel() for p in fresh),
        starting_optimizer_steps=[int(adam.state[p]['step']) for p in adam.param_groups[0]['params'] if p in adam.state][:1],
        gpu=torch.cuda.get_device_name(), energy_force_weights=[200,200], stationarity_weight=30, optimizer_lrs=[g['lr'] for g in adam.param_groups], soap_lr=soap.param_groups[0]['lr']))
    model.train()
    graph = old.JointGraph(trainer)
    seq = old.Stream(ds, old.SEED+300000)
    wall = [s for s in (300,600,900) if not (out/f'wall-{s:06d}.pt').exists()]
    start = time.monotonic()
    secs = elapsed
    for n in range(step+1, args.steps+1):
        batches = seq.get(START+n-1)
        m = graph.step(batches[0], "joint") if len(batches)==1 and graph.eligible(batches[0]) else old.update(model, trainer, batches, "acvr")
        if n==1 or n%100==0 or n==args.steps:
            torch.cuda.synchronize()
            secs = elapsed+time.monotonic()-start
            row = dict(step=n, joint_step=START+n, training_seconds=secs, **m)
            with (out/'training.jsonl').open('a') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n')
            old.write_json(out/'status.json', row)
            if n==1 or n%1000==0: print(json.dumps(row), flush=True)
        secs = elapsed+time.monotonic()-start
        if wall and secs >= wall[0]:
            snap(f'wall-{wall.pop(0):06d}.pt',n,secs)
        if n in (1000,5000,10000,args.steps):
            snap(f'ckpt-{n:06d}.pt',n,secs)
        if n%1000==0 or n==args.steps:
            old.atomic_save(dict(identity=ident, model=model.state_dict(), adam=adam.state_dict(),
                                 soap=soap.state_dict(), step=n, seconds=secs),out/'resume.pt')
    old.write_json(out/'done.json',dict(identity=ident,steps=args.steps,seconds=secs))


def restore(ck):
    cp = torch.load(ck,map_location='cpu',weights_only=False); ident=cp['identity']
    model = old.make_old('cumulene',ident['species']).cuda().double()
    old.install_variant(model,'orient2',seed=old.SEED)
    attach(model,ident['arm'])
    model.load_state_dict(cp['model'],strict=True)
    return model.eval(),cp


def evaluate(args, ds, split, split_sha, out):
    final = out/f'ckpt-{args.steps:06d}.pt'
    final_sha = old.sha(final)
    for fold in ('validation','test'):
        slot = {int(k):v for v,k in enumerate(split[fold])}
        ids = sorted(slot)
        def batches(indices):
            collate = ds.val_dataloader().collate_fn
            for i in indices:
                sample = ds.datasets[fold][slot[i]]
                if int(sample.meta['source_index']) != i: raise ValueError("held-out index mismatch")
                yield i,collate([sample])
        tasks=[(final, out/(fold+'_final.json'), ids,32)]
        if fold=='validation':
            tasks += [(ck,out/f'validation_{ck.stem}.json',old.panel(ids,32),8)
                      for ck in sorted(out.glob('wall-*.pt'))+sorted(out.glob('ckpt-*.pt'))]
        for ck,dest,frames,count in tasks:
            digest=old.sha(ck)
            if dest.exists() and json.loads(dest.read_text())['checkpoint_sha256']==digest: continue
            model, cp = restore(ck)
            if cp['identity']['split_sha256']!=split_sha or cp['identity']['arm']!=args.arm:
                raise ValueError("Evaluation identity mismatch")
            report = old.evaluate(model, batches(frames), newton_ids=old.panel(frames,count),device='cuda')
            report.update(identity=cp['identity'], joint_step=cp['joint_step'], training_seconds=cp['training_seconds'],
                          checkpoint_sha256=digest,split=fold)
            old.write_json(dest,report)
            del model,cp
    model,_=restore(final)
    scan = score_scan(model,trajectory=args.scan_trajectory,out_dir=out/'torsion',checkpoint_path=final)
    with (out/'torsion/scan.csv').open() as f: rows=list(csv.DictReader(f))
    region=[r for r in rows if 150 <= float(r['angle_deg']) <= 175]
    delta=np.array([float(r['proposal_torque_eV_rad'])-float(r['dft_torque_eV_rad']) for r in region])
    old.write_json(out/'torsion/high_angle.json',dict(angle_deg=[150,175],frames=len(region),
        scan_sha256=scan['scan_sha256'],checkpoint_sha256=final_sha,
        torque_RMSE_eV_rad=float(np.mean(delta**2)**.5),torque_MAE_eV_rad=float(np.abs(delta).mean()),
        torque_mean_signed_error_eV_rad=float(delta.mean())))
    old.write_json(out/'complete.json',dict(arm=args.arm,joint_step=START+args.steps,checkpoint_sha256=final_sha))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm',required=True,choices=KINDS)
    ap.add_argument('--parent',type=Path,default=Path('tensor_runs/cumulene/3407/orient2'))
    ap.add_argument('--out-root',type=Path,default=Path('readout_runs'))
    ap.add_argument('--steps',type=int,default=20000)
    ap.add_argument('--stage',choices=('train','evaluate','all'),default='all')
    ap.add_argument('--scan-trajectory',type=Path,default=old.DEFAULT_SCAN)
    args=ap.parse_args()
    if args.steps<1: ap.error("--steps must be positive")
    old.configure()
    ds, split, digest=old.load_data('cumulene')
    out=args.out_root/args.arm;out.mkdir(parents=True,exist_ok=True)
    if args.stage in ('train','all'):
        fit(args,ds,digest,out);gc.collect();torch.cuda.empty_cache()
    if args.stage in ('evaluate','all'):
        evaluate(args,ds,split,digest,out)


if __name__=='__main__':
    main()
