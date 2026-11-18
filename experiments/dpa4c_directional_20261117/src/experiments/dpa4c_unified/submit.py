"""Idempotent Slurm submission: require numerical qualification of the actual parent."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess


def main():
    root=Path.cwd()
    assert root.name=="dpa4c_unified_20261116" and (root/"job.sh").is_file()
    d=root/"dispatch"; d.mkdir(exist_ok=True)
    (root/"logs").mkdir(exist_ok=True)
    with (d/"submit.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        jobs={}
        for system in ("cumulene","nacl","aumgo","sasa","sabs","sadma","dimers"):
            jobs[system]={}
            for arm in ("qualify","baseline","gate","direction"):
                name="u16-"+system+"-"+arm
                receipt=d/(name+".job"); intent=d/(name+".intent")
                dep=[] if arm=="qualify" else ["--kill-on-invalid-dep=yes","--dependency=afterok:"+jobs[system]["qualify"]]
                if receipt.exists(): job=receipt.read_text().strip()
                elif intent.exists():
                    data=subprocess.check_output(["sacct","-nXP","--user",os.environ["USER"],"-S",intent.read_text().strip(),
                        "--name",name,"-o","JobIDRaw,JobName%100"],text=True)
                    ids={r.split("|")[0] for r in data.splitlines() if r.split("|")[1:2]==[name] and r.split("|")[0].isdigit()}
                    if len(ids)!=1: raise ValueError("Resolve prior submit: "+name)
                    job=ids.pop()
                else:
                    intent.write_text(datetime.date.today().isoformat())
                    args=(["../dpa4c_unified/qualify.py","--system",system] if arm=="qualify" else
                          ["../dpa4c_unified/run.py","--system",system,"--arm",arm,"--steps","20000"])
                    try:
                        job=subprocess.check_output(["sbatch","--parsable","--job-name",name,"--chdir",str(root),
                            "--time="+("01:00:00" if arm=="qualify" else "18:00:00"),"--output",str(root/"logs/%x-%j.out"),
                            "--error",str(root/"logs/%x-%j.err"),*dep,str(root/"job.sh"),*args],text=True).strip().split(";")[0]
                    except subprocess.CalledProcessError:
                        intent.unlink()
                        raise
                if not job.isdigit(): raise RuntimeError("Slurm did not return job id")
                receipt.write_text(job+"\n")
                jobs[system][arm]=job
        (d/"jobs.json").write_text(json.dumps(jobs,indent=2)+"\n")
        print(json.dumps(jobs),flush=True)


if __name__=="__main__":main()
