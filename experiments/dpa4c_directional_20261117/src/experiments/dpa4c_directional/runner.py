"""Matched single-forward directional-V campaign. No angle or end-group information enters training."""
import argparse, csv, fcntl, hashlib, json, math, os, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from bridge import ARMS, HERE, build, forward, old, p

INPUT = HERE/"inputs"
BUDGET = 10000
MARKS = (2500, BUDGET)

def dump(path, obj):
    old.atomic_write_text(path, json.dumps(obj, indent=2, allow_nan=False)+"\n")

@contextmanager
def output_lock(directory):
    directory.mkdir(exist_ok=True, parents=True)
    with (directory/".lock").open("a") as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

def data_for(args):
    return old.load_cumulene_data(args.npz,args.split,args.eref,args.profile)

def values(model,z,r,dtype):
    U,q,dq,d=forward(model,old.make_batch(z,r,dtype=dtype))
    if dq is None:
        g,dq=torch.autograd.grad(U.sum(),(d[p.positions],q))
    else:
        (g,)=torch.autograd.grad(U.sum(),d[p.positions])
    return tuple(v.detach().cpu().double().numpy() for v in (U,-g.reshape(*r.shape),q,dq))

def score(model,data,fold,out,step):
    model.eval()
    if fold=="profile":
        pr=data["profile"]; z,r,e,f=pr["z"],pr["pos"],pr["e"],pr["f"]; idx=np.arange(len(z))
    else:
        idx=np.asarray(data["split"][fold]);z,r,e,f=[data[k][idx] for k in ("z","pos","e_int","f")]
    U,F,Q,D=[np.concatenate(col,axis=0) for col in zip(*(
        values(model,z[i:i+16],r[i:i+16],torch.float64) for i in range(0,len(z),16)))]
    angle_fn=old.scan.angle_tangent if fold=="profile" else old.geom_angle_tangent
    angle,tangent=zip(*(angle_fn(zz,rr.astype(np.float64)) for zz,rr in zip(z,r)))
    angle=np.asarray(angle);angle_fold=np.minimum(angle,360.-angle);tangent=np.asarray(tangent)
    ref=(tangent*f).sum((1,2)); pred=(tangent*F).sum((1,2)); err=pred-ref
    urel=U-U[0];erel=e-e[0]
    with (out/f"{fold}-{step:06d}.csv.tmp").open("w",newline="") as file:
        w=csv.writer(file);w.writerow(("source_index","angle_deg","fold_angle_deg","torque_DFT_eV_rad","torque_pred_eV_rad","energy_DFT_eV","energy_pred_eV"))
        w.writerows(zip(idx,angle,angle_fold,ref,pred,e,U))
    (out/f"{fold}-{step:06d}.csv.tmp").replace(out/f"{fold}-{step:06d}.csv")
    rms=lambda arr:float(np.mean(arr*arr)**.5)
    def bin_(lo,hi):
        mask=(angle_fold>=lo)&(angle_fold<hi)
        return dict(n=int(mask.sum()),torque_RMSE_eV_rad=rms(err[mask]) if mask.any() else None)
    spans={"lt90":(0,90),"90_150":(90,150),"150_175":(150,175),"170_175":(170,175),"175_180":(175,181),"170_180":(170,181)}
    bins={tag:bin_(*bounds) for tag,bounds in spans.items()}
    row=dict(frames=len(idx),torque_RMSE_eV_rad=rms(err),
             force_RMSE_meV_A=1000*rms(F-f),bins=bins,
             stationarity_RMS_V=rms(D-D.mean(1,keepdims=True)),
             q_transfer_RMS_e=rms(Q-Q.mean(1,keepdims=True)))
    for tag,b in bins.items():row[f"torque_{tag}_RMSE_eV_rad"]=b["torque_RMSE_eV_rad"]
    if fold!="profile":
        row["energy_RMSE_meV_atom"]=1000*rms((U-e)/z.shape[1])
    else:
        row.update(low_angle_90_150=bins["90_150"],high_angle_150_175=bins["150_175"],
                   endpoint_175_180=bins["175_180"],full_torque_RMSE_eV_rad=rms(err),
                   relative_energy_RMSE_eV=rms(urel-erel),barrier_pred_eV=float(urel[-1]),barrier_DFT_eV=float(erel[-1]),
                   endpoint_torque_pred_eV_rad=float(pred[-1]),endpoint_torque_DFT_eV_rad=float(ref[-1]))
    return row

def optimization(model,fresh):
    h=model.output_modules[0]; bb=list(model.representation.parameters()); fids={id(x) for x in fresh}
    energy=[x for x in h.internal_energy.parameters() if id(x) not in fids]; prop=list(h.proposal.parameters())
    physics=bb+energy+fresh
    assert len(set(map(id,physics+prop)))==len(physics)+len(prop)
    groups=[dict(params=bb,lr=1e-4),dict(params=energy,lr=1e-5)]
    if fresh: groups.append(dict(params=fresh,lr=1e-3))
    opt=torch.optim.Adam(groups)
    soap=old.SOAP(prop,lr=1e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:.1+.45*(1+math.cos(math.pi*n/BUDGET)))
    return physics,prop,opt,soap,scheduler

def train_one(model,opts,z,r,e,f):
    physics,prop,opt,soap,scheduler=opts
    model.train()
    opt.zero_grad(set_to_none=True);soap.zero_grad(set_to_none=True)
    U,q,dq,d=forward(model,old.make_batch(z,r,dtype=torch.float32))
    if dq is None:
        g,dq=torch.autograd.grad(U.sum(),(d[p.positions],q),create_graph=True)
    else:
        (g,)=torch.autograd.grad(U.sum(),d[p.positions],create_graph=True)
    F=-g.reshape_as(torch.as_tensor(f,device=g.device))
    E_ref=torch.as_tensor(e,device=g.device,dtype=g.dtype)
    F_ref=torch.as_tensor(f,device=g.device,dtype=g.dtype)
    phys=200*(((U-E_ref)/z.shape[1]).square().mean()+(F-F_ref).square().mean())
    res=30*(dq-dq.mean(1,keepdim=True)).square().mean()
    torch.autograd.backward(phys,inputs=physics,retain_graph=True)
    torch.autograd.backward(res,inputs=prop)
    torch.nn.utils.clip_grad_norm_(physics,10.,error_if_nonfinite=True);torch.nn.utils.clip_grad_norm_(prop,10.,error_if_nonfinite=True)
    opt.step();soap.step();scheduler.step()
    return phys.detach(),(res.detach()/30).sqrt()

def checked_replay_log(path,step):
    if not path.exists():return
    kept=[]
    for line in path.read_text().splitlines():
        try:r=json.loads(line)
        except ValueError:continue
        if r["step"]<=step:kept.append(line)
    old.atomic_write_text(path,"".join(s+"\n" for s in kept))

def train(args,data,out):
    model,cp,fresh,digest=build(args.ckpt,args.arm,args.seed)
    opts=optimization(model,fresh);_,_,opt,soap,scheduler=opts
    ident=dict(schema="acvr-directional-v1",arm=args.arm,seed=args.seed,steps=BUDGET,batch_size=16,
               backbone_lr=1e-4,base_energy_lr=1e-5,new_lr=1e-3 if fresh else None,proposal_lr=1e-4,
               cosine_min_factor=.1,source_checkpoint_sha256=digest,source_joint_step=cp["joint_step"],
               split_sha256=data["split_sha"],npz_sha256=data["npz_sha"],profile_sha256=data["profile_sha"],
               optimizer_reset=True,physical_E_F_loss_weights=[200,200],proposal_loss_weight=30,
               fresh_parameters=sum(x.numel() for x in fresh),training_stream_seed=args.seed+500000)
    config=out/"config.json";res_path=out/"resume.pt";log=out/"training.jsonl"
    if config.exists() and json.loads(config.read_text())!=ident:raise RuntimeError("Run identity mismatch: refusing to reuse directory")
    if (out/"summary.json").exists():return
    rng=torch.Generator().manual_seed(ident["training_stream_seed"]);idx=np.asarray(data["split"]["train"])
    perm=torch.randperm(len(idx),generator=rng).numpy();cursor=0;start=0;prior_s=0.
    if res_path.exists():
        rs=torch.load(res_path,map_location="cpu",weights_only=False)
        if rs["identity"]!=ident:raise RuntimeError("Resume identity mismatch")
        model.load_state_dict(rs["model"]);opt.load_state_dict(rs["adam"]);soap.load_state_dict(rs["soap"]);scheduler.load_state_dict(rs["scheduler"])
        rng.set_state(rs["rng"]);torch.set_rng_state(rs["torch_rng"]);torch.cuda.set_rng_state(rs["cuda_rng"])
        perm,cursor,start,prior_s=rs["perm"],rs["cursor"],rs["step"],rs["elapsed_s"]
        del rs
    else:
        dump(config,ident);old.atomic_torch_save(dict(identity=ident,step=0,model=old.clone_state_cpu(model.state_dict())),out/"ckpt-000000.pt")
    checked_replay_log(log,start)
    t=time.monotonic()
    stop=min(args.stop_after,BUDGET)
    def save(step,elapsed):
        old.atomic_torch_save(dict(identity=ident,model=old.clone_state_cpu(model.state_dict()),
            adam=opt.state_dict(),soap=soap.state_dict(),scheduler=scheduler.state_dict(),
            rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            perm=perm,cursor=cursor,step=step,elapsed_s=elapsed),res_path)
    for step in range(start+1,stop+1):
        if cursor+16>len(idx):perm=torch.randperm(len(idx),generator=rng).numpy();cursor=0
        frames=idx[perm[cursor:cursor+16]];cursor+=16
        loss,rv=train_one(model,opts,*(data[k][frames] for k in ("z","pos","e_int","f")))
        if step==1 or step%50==0 or step==stop:
            elapsed=prior_s+time.monotonic()-t
            row=dict(step=step,joint_step=80000+step,phys_loss=float(loss),stationarity_RMS_V=float(rv),elapsed_s=elapsed)
            with log.open("a") as fp:fp.write(json.dumps(row)+"\n")
            dump(out/"status.json",row)
            if step==1 or step%250==0:print(f"{args.arm} seed={args.seed} step={step} loss={row['phys_loss']:.5f} wall={elapsed:.1f}",flush=True)
        if step in MARKS:
            torch.cuda.synchronize()
            ck=out/f"ckpt-{step:06d}.pt";old.atomic_torch_save(dict(identity=ident,step=step,joint_step=80000+step,model=old.clone_state_cpu(model.state_dict())),ck)
            model.double()
            result=dict(step=step,joint_step=80000+step,checkpoint_sha256=old.sha256(ck),
                        validation=score(model,data,"validation",out,step),profile_diag=score(model,data,"profile",out,step))
            model.float()
            dump(out/f"metrics-{step:06d}.json",result)
        if step%250==0 or step==stop:
            torch.cuda.synchronize();save(step,prior_s+time.monotonic()-t)
    if stop==BUDGET and (out/f"metrics-{BUDGET:06d}.json").exists():
        final=json.loads((out/f"metrics-{BUDGET:06d}.json").read_text())
        dump(out/"summary.json",dict(identity=ident,final=final,elapsed_s=prior_s+time.monotonic()-t,
             evaluation_policy="validation and diagnostic scan only; no test results used for screening"))

def preflight(args,data):
    arms=list(ARMS) if args.arm=="ALL" else [args.arm]
    idx=np.asarray(data["split"]["train"][:2]); z,r,e,f=[data[k][idx] for k in ("z","pos","e_int","f")]
    m,_,_,_=build(args.ckpt,"B0",args.seed,dtype=torch.float64)
    # Independent legacy scorer at zero residual, not the same directional callback twice.
    parent=old.forward_energy_charge(m,old.make_batch(z,r,dtype=torch.float64))
    u0,q0,dq0,d=parent
    (g0,dq0)=torch.autograd.grad(u0.sum(),(d[p.positions],q0))
    ref=[v.detach().cpu().numpy() for v in (u0,-g0.reshape(*r.shape),q0,dq0)]
    del m
    report={}
    for arm in arms:
        model,cp,fresh,digest=build(args.ckpt,arm,args.seed,dtype=torch.float64)
        got=values(model,z,r,torch.float64)
        diffs=[float(np.abs(a-b).max()) for a,b in zip(got,ref)]
        if max(diffs)>2e-10:raise AssertionError(f"Starting E/F/q/dq mismatch {arm}: {diffs}")
        if fresh:
            corr=model.output_modules[0].internal_energy.directional_v
            # Exercise output gates (including the H moment maps) away from zero.
            with torch.no_grad():
                gen=torch.Generator(device="cuda").manual_seed(9102)
                for par in fresh:
                    if par.count_nonzero()==0:par.add_(.002*torch.randn(par.shape,device=par.device,dtype=par.dtype,generator=gen))
        actual=values(model,z,r,torch.float64)
        if max(np.abs(actual[i]-ref[i]).max() for i in (2,3))>1e-10:raise AssertionError("Geometry has affected partial dE/dq or q")
        r=r.astype(np.float64)
        plus=r.copy();minus=r.copy();eps=2e-5
        plus[0,1,2]+=eps;minus[0,1,2]-=eps
        ep=values(model,z[:1],plus[:1],torch.float64)[0][0];em=values(model,z[:1],minus[:1],torch.float64)[0][0]
        err=abs((em-ep)/(2*eps)-actual[1][0,1,2])
        if err>2e-6:raise AssertionError(f"FD mismatch {arm}: {err}")
        rot=np.array([[.36,-.48,.8],[.8,.60,0],[-.48,.64,.60]])
        transl=np.array([1.3,-.4,.2]);perm=np.roll(np.arange(r.shape[1]),3)
        # One proper and one improper rigid transform, both with a permutation.
        rigerr=[]
        for op in (rot,-rot):
            er,fr,_,_=values(model,z[:,perm],(r@op+transl)[:,perm],torch.float64)
            rigerr.append([float(np.max(abs(er-actual[0]))),float(np.max(abs(fr-((actual[1]@op)[:,perm]))))])
        if np.max(rigerr)>4e-5:raise AssertionError(f"Symmetry mismatch {arm}: {rigerr}")
        model,_,fresh,_=build(args.ckpt,arm,args.seed)
        opts=optimization(model,fresh);before=[x.detach().clone() for x in fresh]
        if fresh and getattr(model.output_modules[0].internal_energy.directional_v,"fuse",False):
            h_before=[x.detach().clone() for x in model.output_modules[0].internal_energy.directional_v.fuse_weight]
        else:h_before=[]
        loss,rv=train_one(model,opts,z,r,e,f)
        change=sum(float((x-y).abs().sum()) for x,y in zip(fresh,before))
        if fresh and change<=0:raise AssertionError(f"Dead correction: {arm}")
        if h_before:
            h_change=sum(float((x-y).abs().sum()) for x,y in zip(model.output_modules[0].internal_energy.directional_v.fuse_weight,h_before))
            if h_change<=0:raise AssertionError("H map received zero optimizer update")
        else:h_change=0.
        report[arm]=dict(parent_sha256=digest,parent_step=cp["joint_step"],start_max_abs_E_F_q_partial=diffs,active_fd_error=err,
                active_E_F_errors_rigid=rigerr,active_q_separation=True,one_step_loss=float(loss),one_step_residual_RMS=float(rv),
                new_parameter_update_L1=change,H_update_L1=h_change)
        print(arm,report[arm],flush=True)
        del model,opts
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--action",choices=["train","preflight","score-test"],required=True)
    ap.add_argument("--arm",choices=(*ARMS,"ALL"),required=True)
    ap.add_argument("--seed",type=int,default=3407)
    ap.add_argument("--out-root",required=True,type=Path)
    ap.add_argument("--steps",type=int,choices=[BUDGET],default=BUDGET)
    ap.add_argument("--batch-size",type=int,choices=[16],default=16)
    ap.add_argument("--stop-after",type=int,default=BUDGET,help="Interruptible smoke run; cosine schedule always spans 10,000")
    for name,file in [("npz","efa_cumulene.npz"),("profile","efa_cumulene_profile.npz"),("ckpt","gate_ckpt-020000.pt"),("split","split.json"),("eref","energy_reference.json")]:
        ap.add_argument("--"+name,type=Path,default=INPUT/file)
    args=ap.parse_args()
    if not 1<=args.stop_after<=BUDGET:ap.error("stop-after must be in 1..10000")
    old.configure_deterministic();torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    data=data_for(args)
    if args.action=="preflight":
        args.out_root.mkdir(parents=True,exist_ok=True)
        result=preflight(args,data);dump(args.out_root/("preflight.json" if args.arm=="ALL" else f"preflight-{args.arm}.json"),result)
    else:
        if args.arm=="ALL":ap.error("--arm ALL only for --action preflight")
        out=args.out_root/args.arm
        with output_lock(out):
            if args.action=="train":train(args,data,out)
            else:
                if not (out/"summary.json").exists():raise RuntimeError("No completed training")
                model,_,_,_=build(args.ckpt,args.arm,args.seed,dtype=torch.float64)
                state=torch.load(out/f"ckpt-{BUDGET:06d}.pt",map_location="cpu",weights_only=False)
                if (state["identity"]["seed"],state["identity"]["arm"],state["step"])!=(args.seed,args.arm,BUDGET):raise ValueError("Wrong checkpoint")
                model.load_state_dict(state["model"],strict=True)
                dump(out/"sealed-test.json",dict(step=BUDGET,checkpoint_sha256=old.sha256(out/f"ckpt-{BUDGET:06d}.pt"),test=score(model,data,"test",out,BUDGET)))
if __name__=="__main__":main()
