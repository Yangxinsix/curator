"""Submit exactly one job per arm, with durable receipts. Run in the tensor deployment."""
import argparse, datetime, fcntl, json, os, subprocess
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('group',choices=('fragment','efa'));args=ap.parse_args()
    root=Path.cwd();out=root/'transfer_1116';out.mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    specs=[]
    if args.group=='fragment':
        for system in ('sasa','sabs','sadma','dimers'):
            parent=root/'tensor_runs'/system/'3407/orient2'
            for f in ('resume.pt','ckpt-020000.pt','test_final.json','complete.json'): assert (parent/f).is_file()
            for arm in ('linear','stf_frame'):
                specs.append((system,arm,['../dpa4c_transfer/run.py','--system',system,'--arm',arm,'--parent',str(parent),'--steps','20000','--out-root',str(out/system)]))
    else:
        for arm in ('pure','original_efa'):
            specs.append(('cumulene',arm,['../dpa4c_transfer/original_efa.py','--arm',arm,'--out-root',str(out/'cumulene')]))
    with (out/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for s,a,cmd in specs:
            name='x16-'+s+'-'+a; receipt=out/(name+'.job');intent=out/(name+'.intent')
            if receipt.exists():job=receipt.read_text().strip()
            elif intent.exists():
                acct=subprocess.check_output(['sacct','-XnP','--user',os.environ['USER'],'-S',intent.read_text().strip(),'--name',name,'-o','JobIDRaw,JobName%100'],text=True)
                ids={row.split('|')[0] for row in acct.splitlines() if row.split('|')[1:2]==[name] and row.split('|')[0].isdigit()}
                if len(ids)!=1:raise RuntimeError(f'Unresolved prior submission: {name}: {ids}')
                job=ids.pop()
            else:
                intent.write_text(datetime.date.today().isoformat())
                job=subprocess.check_output(['sbatch','--parsable','--chdir',str(root),'--job-name',name,'--time=12:00:00','--output',str(out/'logs/%x-%j.out'),'--error',str(out/'logs/%x-%j.err'),str(root/'job.sh'),*cmd],text=True).strip().split(';')[0]
            assert job.isdigit();receipt.write_text(job+'\n');print(s,a,job,flush=True)

if __name__=='__main__':main()
