#!/bin/bash
# From the dpa4c_tensor_20261115 deployment root.
set -euo pipefail
root=$PWD
out=$root/readout_runs
parent=$root/tensor_runs/cumulene/3407/orient2
for f in "$root/job.sh" "$root/src/experiments/dpa4c_readout/run.py" "$parent/resume.pt" "$parent/ckpt-020000.pt" "$parent/test_final.json"; do
    test -s "$f" || { echo "Missing $f" >&2; exit 1; }
done
mkdir -p "$out/logs" "$out/receipts"
exec 9>"$out/receipts/submit.lock"; flock -x 9
for arm in linear quadratic mlp chebyshev_z stf_frame; do
    name="dr1116-$arm"
    idfile="$out/receipts/$arm"
    if [[ -s "$idfile" ]]; then read -r job < "$idfile"
    elif [[ -s "$idfile.intent" ]]; then
        read -r day < "$idfile.intent"
        job=$(sacct -n -X -P -S "$day" --user="$USER" --name="$name" -o JobIDRaw,JobName%100 | awk -F'|' -v n="$name" '$2==n && $1~/^[0-9]+$/{print $1}' | sort -u)
        [[ "$job" =~ ^[0-9]+$ ]] || { echo "Resolve ambiguous prior submission for $name" >&2; exit 1; }
    else
        date +%F > "$idfile.intent"
        raw=$(sbatch --parsable --chdir="$root" --job-name="$name" --time=08:00:00 \
            --output="$out/logs/%x-%j.out" --error="$out/logs/%x-%j.err" "$root/job.sh" \
            ../dpa4c_readout/run.py --arm "$arm" --parent "$parent" --steps 20000 --out-root "$out")
        job=${raw%%;*}
    fi
    [[ "$job" =~ ^[0-9]+$ ]]
    printf '%s\n' "$job" > "$idfile.tmp"; mv "$idfile.tmp" "$idfile"
    printf '%s\t%s\n' "$arm" "$job"
done
