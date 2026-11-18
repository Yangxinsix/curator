#!/bin/bash
set -euo pipefail
export SLURM_SUBMIT_DIR="$PWD"
src="$PWD/src/experiments/dpa4c_directional"
runs="$PWD/runs"
hostpy=/data1/home/hzxy05/miniconda3/bin/python
role=${1:?role}
case "$role" in
    select) exec "$hostpy" "$src/selection.py" --root "$runs" --select ;;
    report) exec "$hostpy" "$src/selection.py" --root "$runs" --report ;;
    preflight) phase=preflight; arm=ALL; seed=3407; out="$PWD/qualification" ;;
    screen)
        arms=(B0 B1 T20 T02 T22 V11 O1 J2 B1H J2H)
        arm=${arms[$SLURM_ARRAY_TASK_ID]}; phase=train; seed=3407; out="$runs/$seed" ;;
    confirm)
        read -r arm seed < <("$hostpy" -c '
import json,sys
a=json.load(open(sys.argv[1])); n=int(sys.argv[2]); chosen=a["chosen"]
assert len(chosen)==2 and len(set(chosen+["B0"]))==3
assert a["confirmation_seeds"]==[3408,3409,3410]
print((["B0"]+chosen)[n%3], a["confirmation_seeds"][n//3])' "$runs/selection.json" "$SLURM_ARRAY_TASK_ID")
        phase=train; out="$runs/$seed" ;;
    *) exit 2 ;;
esac
printf 'role=%s arm=%s seed=%s out=%s\n' "$role" "$arm" "$seed" "$out"
inputs="$PWD/inputs"
args=(--action "$phase" --arm "$arm" --seed "$seed" --out-root "$out" --steps 10000 --batch-size 16 --ckpt "$inputs/gate_ckpt-020000.pt" --split "$inputs/split.json" --eref "$inputs/energy_reference.json" --npz "$inputs/efa_cumulene.npz" --profile "$inputs/efa_cumulene_profile.npz")
# Reuse the verified HHU CUDA 12.8 image, per-architecture extension cache and dataset bind.
exec bash "$PWD/job.sh" ../dpa4c_directional/runner.py "${args[@]}"
