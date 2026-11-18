"""Matched ACVR continuations; only the named residual differs between arms."""
import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / n) for n in ("dpa4c_tensor", "dpa4c_readout", "dpa4c_transfer")]
import numpy as np
import torch
import train_runner as base
from readout import attach as add_stf
from gating import attach as add_gate
from direction import attach as add_direction

ARMS = ("baseline", "gate", "direction")
DEPLOY = base.ROOT / "experiments/dpa4c_tensor_20261115"


def parent_info(system):
    if system in ("nacl", "aumgo"):
        return DEPLOY / "tensor_runs" / system / "3407/orient2", 40000, "orient2"
    parent = DEPLOY / ("readout_runs" if system == "cumulene" else "transfer_1116/" + system)
    return parent / "stf_frame", 60000, "stf_frame"


def add_arm(model, arm):
    return {"baseline": lambda m: [], "gate": add_gate, "direction": add_direction}[arm](model)


def source(args, ds, split_sha):
    root, offset, role = parent_info(args.system)
    ck, st = root / "ckpt-020000.pt", root / "resume.pt"
    digest = base.sha(ck)
    if digest != json.loads((root / "test_final.json").read_text())["checkpoint_sha256"]:
        raise ValueError("Parent differs from scored endpoint")
    cp = torch.load(ck, map_location="cpu", weights_only=False)
    resume = torch.load(st, map_location="cpu", weights_only=False)
    sid = cp["identity"]
    species = list(ds.sources[0].species)
    if (sid != resume["identity"] or cp["joint_step"] != offset or resume["step"] != 20000 or
        (sid["system"], sid["arm"], sid["seed"], sid["split_sha256"], sid["species"]) !=
        (args.system, role, base.SEED, split_sha, species)):
        raise ValueError("Parent system, endpoint or data mismatch")
    if cp["model"].keys() != resume["model"].keys() or any(
        not torch.equal(v, resume["model"][k]) for k,v in cp["model"].items() if torch.is_tensor(v)):
        raise ValueError("Optimizer resume differs from scored weights")
    model = base.make_old(args.system, species).cuda().float()
    t = base.ScratchTrainer(model, backbone_lr=1e-4, energy_lr=1e-5, proposal_lr=1e-4,
                            energy_weight=200., force_weight=200.)
    adam = t.physics_optimizer
    if args.system == "aumgo":
        gain = model.output_modules[0].internal_energy.attention.group_log_gain
        adam.param_groups[1]["params"] = [p for p in adam.param_groups[1]["params"] if p is not gain]
        adam.add_param_group(dict(params=[gain], lr=.003))
    def add(params):
        adam.param_groups[1]["params"].extend(params)
        t.energy.extend(params)
        t.physical = t.energy + t.backbone
    base.install_variant(model, "orient2", seed=base.SEED)
    add(list(model.output_modules[0].internal_energy.tensor_branch.parameters()))
    if role == "stf_frame":
        add(add_stf(model, "stf_frame"))
    model.load_state_dict(cp["model"], strict=True)
    adam.load_state_dict(resume["adam"])
    t.solver_optimizer.load_state_dict(resume["soap"])
    if role == "orient2":
        add(add_stf(model, "stf_frame"))
    fresh = add_arm(model, args.arm)
    add(fresh)
    if sorted(map(id,t.physical)) != sorted(id(p) for g in adam.param_groups for p in g["params"]):
        raise ValueError("Incomplete/duplicated physical optimizer")
    if any(v["exp_avg"].shape != p.shape for p,v in adam.state.items() if "exp_avg" in v):
        raise ValueError("Restored Adam slot shape mismatch")
    ident = dict(schema="acvr-unified-v1", system=args.system, arm=args.arm, budget=args.steps,
        seed=base.SEED, source_arm=role, source_joint_step=offset, source_checkpoint_sha256=digest,
        source_resume_sha256=base.sha(st), split_sha256=split_sha, species=species)
    return model,t,ident,fresh


def restore(ck):
    cp = torch.load(ck, map_location="cpu", weights_only=False)
    i = cp["identity"]
    m = base.make_old(i["system"], i["species"]).cuda().double()
    base.install_variant(m, "orient2", seed=base.SEED)
    add_stf(m, "stf_frame")
    add_arm(m, i["arm"])
    m.load_state_dict(cp["model"], strict=True)
    return m.eval(), cp


def fit(args,ds,digest,out):
    m,t,i,fresh = source(args,ds,digest)
    offset = i["source_joint_step"]
    adam,soap = t.physics_optimizer,t.solver_optimizer
    n0, elapsed = 0,0.
    if (out/"resume.pt").exists():
        st=torch.load(out/"resume.pt",map_location="cpu",weights_only=False)
        if st["identity"]!=i: raise ValueError("Resume identity mismatch")
        m.load_state_dict(st["model"],strict=True)
        adam.load_state_dict(st["adam"]); soap.load_state_dict(st["soap"])
        n0,elapsed=st["step"],st["seconds"]
        del st
    def snap(name,n,sec):
        base.atomic_save(dict(identity=i, model=m.state_dict(), joint_step=offset+n, added_steps=n,
                              training_seconds=sec),out/name)
    if n0==0: snap("ckpt-000000.pt",0,0.)
    base.write_json(out/"config.json",dict(identity=i,batch=base.BATCH,stream_start=offset,
        new_arm_parameters=sum(p.numel() for p in fresh),gpu=torch.cuda.get_device_name(),
        adam_lrs=[g["lr"] for g in adam.param_groups],soap_lr=soap.param_groups[0]["lr"],
        weights=dict(E=200,F=200,residual=30),proposal_target="original stationarity only"))
    if n0==args.steps: return
    m.train()
    graph=(base.AuMgOGraph if args.system=="aumgo" else base.JointGraph)(t)
    fixed=args.system in ("aumgo","sasa")
    stream=(base.Batches if fixed else base.Stream)(ds,base.SEED+300000)
    wall=[s for s in (300,600,900) if not (out/f"wall-{s:06d}.pt").exists()]
    start=time.monotonic()
    for n in range(n0+1,args.steps+1):
        r=stream.get(offset+n-1); batches=[r] if fixed else r
        if len(batches)==1 and graph.eligible(batches[0]):
            metrics=graph.step(batches[0],"joint")
        elif fixed:
            raise RuntimeError("Fixed-shape source graph unexpectedly unsupported")
        else:
            metrics=base.update(m,t,batches,"acvr")
        seconds=elapsed+time.monotonic()-start
        if n==1 or n%100==0 or n==args.steps:
            torch.cuda.synchronize(); seconds=elapsed+time.monotonic()-start
            row=dict(step=n,joint_step=offset+n,training_seconds=seconds,**metrics)
            base.write_json(out/"status.json",row)
            with (out/"training.jsonl").open("a") as f: f.write(json.dumps(row,allow_nan=False)+"\n")
            if n==1 or n%1000==0: print(row,flush=True)
        if wall and seconds>=wall[0]: snap(f"wall-{wall.pop(0):06d}.pt",n,seconds)
        if n in (1000,5000,10000,args.steps): snap(f"ckpt-{n:06d}.pt",n,seconds)
        if n%1000==0 or n==args.steps:
            base.atomic_save(dict(identity=i,model=m.state_dict(),adam=adam.state_dict(),soap=soap.state_dict(),
                                  step=n,seconds=seconds),out/"resume.pt")
    base.write_json(out/"done.json",dict(identity=i,steps=n,seconds=seconds))


def score(args,ds,split,digest,out):
    ck=out/f"ckpt-{args.steps:06d}.pt"
    m,cp=restore(ck)
    if cp["identity"]["split_sha256"]!=digest: raise ValueError("Scoring split mismatch")
    ch=base.sha(ck)
    def batch(fold,indices):
        slots={int(k):j for j,k in enumerate(split[fold])}
        collate=ds.val_dataloader().collate_fn
        for k in indices:
            v=ds.datasets[fold][slots[k]]
            if int(v.meta["source_index"])!=k: raise ValueError("Held-out ID mismatch")
            yield k,collate([v])
    for fold in ("validation","test"):
        ids=sorted(map(int,split[fold]))
        tasks=[(ck,out/(fold+"_final.json"),ids,32)]
        if fold=="validation":
            tasks.extend((c,out/f"validation_{c.stem}.json",base.panel(ids,32),8)
                         for c in sorted(out.glob("ckpt-*.pt"))+sorted(out.glob("wall-*.pt")))
        for c,dest,frames,count in tasks:
            h=base.sha(c)
            if dest.is_file() and json.loads(dest.read_text())["checkpoint_sha256"]==h: continue
            obj,rec=(m,cp) if c==ck else restore(c)
            report=base.evaluate(obj,batch(fold,frames),newton_ids=base.panel(frames,count),device="cuda")
            report.update(identity=rec["identity"],joint_step=rec["joint_step"],
                training_seconds=rec["training_seconds"],split=fold,checkpoint_sha256=h)
            base.write_json(dest,report); del obj,rec
    frames=list(batch("validation",base.panel(sorted(map(int,split["validation"])),32)))
    with base._read_only(m,"cuda") as head:
        base._forward(m,head,frames[0][1],"cuda","proposal"); torch.cuda.synchronize()
        t0=time.monotonic()
        for _,b in frames: base._forward(m,head,b,"cuda","proposal")
        torch.cuda.synchronize()
        base.write_json(out/"timing.json",dict(checkpoint_sha256=ch,frames=len(frames),
            wall_ms_per_frame=1000*(time.monotonic()-t0)/len(frames),dtype="float64",gpu=torch.cuda.get_device_name(),
            mode="generic one-pass energy,charge,forces including host transfers; single structures"))
    if args.system in ("sadma","sasa","sabs","dimers"):
        from fragment_scores import score_fragments
        score_fragments(args.system,m,ck,out/"fragments")
    if args.system=="cumulene":
        from scan import score_scan
        sc=score_scan(m,trajectory=base.DEFAULT_SCAN,out_dir=out/"torsion",checkpoint_path=ck)
        with (out/"torsion/scan.csv").open() as f:
            rows=[r for r in csv.DictReader(f) if 150<=float(r["angle_deg"])<=175]
        delta=[float(r["proposal_torque_eV_rad"])-float(r["dft_torque_eV_rad"]) for r in rows]
        base.write_json(out/"torsion/high_angle.json",dict(frames=len(rows),checkpoint_sha256=ch,
            scan_sha256=sc["scan_sha256"],angle_deg=[150,175],torque_RMSE_eV_rad=float(np.mean(np.square(delta))**.5)))
    base.write_json(out/"complete.json",dict(checkpoint_sha256=ch,identity=cp["identity"]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--system",required=True,choices=base.SYSTEMS)
    ap.add_argument("--arm",required=True,choices=ARMS)
    ap.add_argument("--steps",type=int,default=20000)
    ap.add_argument("--stage",choices=("train","score","all"),default="all")
    ap.add_argument("--out-root",type=Path,default=Path("runs"))
    args=ap.parse_args()
    if args.steps<1: ap.error("steps must be positive")
    base.configure()
    ds,split,digest=base.load_data(args.system)
    out=args.out_root/args.system/args.arm; out.mkdir(exist_ok=True,parents=True)
    if args.stage in ("train","all"):
        fit(args,ds,digest,out); gc.collect(); torch.cuda.empty_cache()
    if args.stage in ("score","all"): score(args,ds,split,digest,out)


if __name__=="__main__": main()
