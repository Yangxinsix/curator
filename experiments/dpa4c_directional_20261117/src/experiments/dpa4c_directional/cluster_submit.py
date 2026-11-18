"""Once-only, resumable Slurm publication of the 10-arm screen, selection and 3x3 replicate array."""
import datetime
import fcntl
import json
import os
import subprocess
from pathlib import Path

ARMS=['B0','B1','T20','T02','T22','V11','O1','J2','B1H','J2H']

def main():
    root=Path.cwd()
    assert (root/'job.sh').exists() and (root/'sources.json').is_file()
    code=root/'src/experiments/dpa4c_directional'
    assert (code/'runner.py').is_file() and (code/'selection.py').is_file()
    dispatch=root/'dispatch'
    dispatch.mkdir(exist_ok=True)
    logs=root/'logs'
    logs.mkdir(exist_ok=True)
    common=['-A','group_hzxy','--qos=qos_hzxy05','--chdir',str(root),
            '--output',str(logs/'%x-%A-%a.out'),'--error',str(logs/'%x-%A-%a.err')]
    with (dispatch/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ids={}
        for role,deps,gpu,limit,array in [
            ('preflight',[],True,'01:00:00',None),('screen',['preflight'],True,'10:00:00','0-9'),
            ('select',['screen'],False,'00:15:00',None),('confirm',['select'],True,'10:00:00','0-8'),
            ('report',['confirm'],False,'00:15:00',None)]:
            name='dv17-'+role
            receipt=dispatch/(role+'.job')
            intent=dispatch/(role+'.intent')
            if receipt.exists():
                job=receipt.read_text().strip()
            elif intent.exists():
                raw=subprocess.check_output(['sacct','-nXP','--user',os.environ['USER'],'-S',intent.read_text().strip(),'--name',name,'-o','JobID,JobName%50'],universal_newlines=True)
                matches={x.split('|')[0].split('_')[0] for x in raw.splitlines() if x.split('|')[1:2]==[name]}
                if len(matches)!=1: raise RuntimeError('Ambiguous prior submission '+name+': '+str(matches))
                job=matches.pop()
            else:
                intent.write_text(datetime.date.today().isoformat())
                flags=['-p','hgpu2p','--gres=gpu:1','-c','4'] if gpu else ['-p','hcpu48,hcpu48y','-c','1']
                if array:flags+=['--array='+array]
                if deps:flags+=['--kill-on-invalid-dep=yes','--dependency=afterok:'+':'.join(ids[x] for x in deps)]
                try:
                    job=subprocess.check_output(['sbatch','--parsable','--job-name='+name,'--time='+limit,*common,*flags,str(code/'cluster_worker.sh'),role],universal_newlines=True).strip().split(';')[0]
                except subprocess.CalledProcessError:
                    intent.unlink();raise
            if not job.isdigit():raise RuntimeError('Invalid job id '+repr(job))
            receipt.write_text(job+'\n');ids[role]=job
        data=dict(schema='directional-v-cumulene/1',cwd=str(root),jobs=ids,screen_by_task=ARMS,
                  seeds=[3408,3409,3410],confirm_layout='index = 3 * seed_index + index in [B0, chosen0, chosen1]',
                  runs=str(root/'runs'),hashes=json.loads((root/'sources.json').read_text()))
        tmp=dispatch/'receipt.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(dispatch/'receipt.json')
        print('RECEIPT '+json.dumps(data))
if __name__=='__main__': main()
