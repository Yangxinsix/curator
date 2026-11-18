"""Preregistered, validation-only selection and paired-seed reporting. No GPU needed."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ARMS = ('B0','B1','T20','T02','T22','V11','O1','J2','B1H','J2H')
PRIMARY = 3407
SEEDS = (3408,3409,3410)
HASH = {'source_checkpoint_sha256':'fd3ec04f4f2cb8d7c15e64ce3413d77f5897b9de911fc662ab3761e8ae4e7de6',
        'split_sha256':'8979beb238a8ef35bf18cc5f5ef941ad71d567ebadc44a208b9312e2e5f758be',
        'npz_sha256':'5ee897677cdcba26dfe6eee6794633506383be513d92aff4a8584b8950c5a111',
        'profile_sha256':'08e4b224428b2178e788ac5741d3c8bab0f2a25da8af22648eac20b083215f26'}
RULE = ('10k only. Require both ordinary validation 90<=angle<150 torque RMSE and rigid scan 90<=angle<150 RMSE '
        '<= B0 at the same seed. Eligible arms ordered by ordinary all-frame validation torque RMSE then arm id. '
        'If fewer than two eligible, complete two confirmation slots with flagged diagnostics ranked by smallest worst '
        'relative low-angle excess, then ordinary all-frame validation RMSE. No test or 2500-step score used.')
def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def finite(x):
    y=float(x)
    if not math.isfinite(y) or y<0: raise ValueError(f'invalid metric: {x}')
    return y
def load(root,seed,arm,step):
    folder=root/str(seed)/arm
    if step==10000:
        raw=json.loads((folder/'summary.json').read_text()); identity=raw['identity']; f=raw['final']
    else:
        metric=json.loads((folder/f'metrics-{step:06d}.json').read_text()); f=metric
        identity=metric['identity'] if 'identity' in metric else json.loads((folder/'config.json').read_text())
    if (identity['arm'],identity['seed'],f['step']) != (arm,seed,step) or any(identity[k]!=v for k,v in HASH.items()):
        raise ValueError(f'Parent/data/step mismatch: {folder}')
    v=f['validation']; p=f['profile_diag']
    if v['frames']!=500 or p['frames']!=1000: raise ValueError(f'Wrong evaluation coverage: {folder}')
    return dict(arm=arm,seed=seed,step=step,
                val_torque=finite(v['torque_RMSE_eV_rad']),val_low=finite(v['torque_90_150_RMSE_eV_rad']),
                scan_low=finite(p['low_angle_90_150']['torque_RMSE_eV_rad']),
                val_high=finite(v['torque_175_180_RMSE_eV_rad']),
                scan_high=finite(p['endpoint_175_180']['torque_RMSE_eV_rad']),
                val_E_meV_atom=finite(v['energy_RMSE_meV_atom']),val_F_meV_A=finite(v['force_RMSE_meV_A']),
                scan_rel_E_eV=finite(p['relative_energy_RMSE_eV']),val_stationarity_uV=1e6*finite(v['stationarity_RMS_V']))
def guard(row,base):
    excess=max(0,row['val_low']/base['val_low']-1,row['scan_low']/base['scan_low']-1)
    ok=row['val_low']<=base['val_low'] and row['scan_low']<=base['scan_low']
    return ok,excess
def rank(rows):
    baseline=next(row for row in rows if row['arm']=='B0')
    out=[]
    for row in rows:
        if row['arm']=='B0': continue
        ok,excess=guard(row,baseline)
        out.append(dict(**row,low_guard_pass=ok,worst_low_fraction_over_B0=excess,
                        improves_primary=row['val_torque']<baseline['val_torque']))
    out.sort(key=lambda r:(not r['low_guard_pass'],r['worst_low_fraction_over_B0'],r['val_torque'],r['arm']))
    return baseline,out
def select(root):
    rows=[load(root,PRIMARY,a,10000) for a in ARMS]
    baseline,table=rank(rows)
    results=dict(schema='acvr.directional.selection.1',rule=RULE,primary_seed=PRIMARY,confirmation_seeds=list(SEEDS),
                 baseline='B0',chosen=[x['arm'] for x in table[:2]],baseline_metrics=baseline,
                 diagnostics_only=[x['arm'] for x in table[:2] if not x['low_guard_pass'] or not x['improves_primary']],
                 ranked=table,input_hashes=HASH,selector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(root/'selection.json',results); print(json.dumps({k:results[k] for k in ['chosen','baseline','diagnostics_only']}))
def report(root):
    selected=json.loads((root/'selection.json').read_text())
    arms=selected['chosen']
    if len(set(arms))!=2 or any(a not in ARMS[1:] for a in arms) or selected['confirmation_seeds']!=list(SEEDS): raise ValueError('bad selection')
    rows=[]; by_arm={}
    for arm in arms:
        paired=[]
        for s in SEEDS:
            b=load(root,s,'B0',10000); a=load(root,s,arm,10000)
            delta={k:a[k]-b[k] for k in a if k not in ('seed','arm','step')}
            ok,_=guard(a,b)
            paired.append(dict(seed=s,arm_metrics=a,B0=b,delta=delta,low_guard_pass=ok));rows.append(dict(arm=arm,seed=s,low_guard_pass=ok,**delta))
        by_arm[arm]=dict(paired=paired,mean_paired_delta={k:statistics.mean(x['delta'][k] for x in paired) for k in delta},
                         sample_sd_paired_delta={k:statistics.stdev(x['delta'][k] for x in paired) for k in delta},
                         seeds_beating_B0_primary=sum(x['delta']['val_torque']<0 for x in paired),
                         seeds_passing_both_low_guards=sum(x['low_guard_pass'] for x in paired))
    write(root/'paired_confirmation.json',dict(rule=RULE,unit_of_replication='three fresh training seeds; fixed 80k parent, fixed split',
         deltas='candidate minus B0; negative improves for errors',seeds=SEEDS,arms=by_arm))
    with (root/'paired_confirmation.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(root/'paired_confirmation.json')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--select',action='store_true');group.add_argument('--report',action='store_true');group.add_argument('--inspect-2500',action='store_true')
    args=p.parse_args()
    if args.select:select(args.root)
    elif args.report:report(args.root)
    else: print(json.dumps([load(args.root,PRIMARY,a,2500) for a in ARMS],indent=2))
if __name__=='__main__':main()
