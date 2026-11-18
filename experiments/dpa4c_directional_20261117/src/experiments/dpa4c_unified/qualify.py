"""Per-system, per-parent numerical qualification before starting all three arms."""
import argparse
from pathlib import Path
import sys

# Import the campaign's run.py explicitly: historical modules also have that name.
from importlib.util import spec_from_file_location, module_from_spec
spec=spec_from_file_location("unified_runner",Path(__file__).with_name("run.py"))
job=module_from_spec(spec); spec.loader.exec_module(job)
import torch
from preflight import native, fast, compare


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--system",required=True,choices=job.base.SYSTEMS)
    args=ap.parse_args()
    job.base.configure(); args.steps=20000
    ds,_,digest=job.base.load_data(args.system)
    batch=ds.val_dataloader().collate_fn([ds.datasets["validation"][0]])
    out=Path("qualification");out.mkdir(exist_ok=True)
    result={}
    for arm in job.ARMS:
        args.arm=arm
        m,t,i,new=job.source(args,ds,digest)
        m.double()
        p=native(m,batch,"proposal")
        if arm=="baseline": ref=p
        z=compare(p,ref,tol=1e-10)
        f=fast(m,batch)
        row=dict(identity=i,initial=z,fast=compare(p,f))
        if new:
            target=batch["variational"]["labels"]["condition_forces"]["values"].to(f[1]).reshape_as(f[1])
            grads=torch.autograd.grad((f[1]-target).square().mean(),new)
            if not all(bool(g.isfinite().all()) for g in grads):raise ValueError("Nonfinite force derivative")
            norm=sum(g.square().sum() for g in grads).sqrt(); assert float(norm)>0
            with torch.no_grad():
                for v,g in zip(new,grads): v.add_(.05*g/norm)
            active=native(m,batch,"proposal")
            row.update(active_fast=compare(active,fast(m,batch)),force_gradient_norm=float(norm),
                       active_force_change=float((active[1]-p[1]).abs().max()),
                       active_q_change=float((active[2]-p[2]).abs().max()))
            assert row["active_force_change"]>1e-12
            if arm=="direction": assert row["active_q_change"]>1e-12
            else: torch.testing.assert_close(active[2],p[2],atol=1e-12,rtol=0)
        result[arm]=row
        del m,t,f,new
    job.base.write_json(out/(args.system+".json"),result)
    print("QUALIFIED",args.system,flush=True)


if __name__=="__main__":main()
