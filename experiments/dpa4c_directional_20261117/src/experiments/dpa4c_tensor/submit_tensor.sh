#!/bin/bash
# Run from a deployed workspace with the original job.sh/unified and src/experiments/{dpa4c_*,periodic_qefa}.
# bash src/experiments/dpa4c_tensor/submit_tensor.sh [afterok_job_id] [outputs_dir] [original_runs]
set -euo pipefail
work=$PWD
gate=${1:-}
out=${2:-$work/tensor_runs}
origin=${3:-/data1/home/hzxy05/work/yangxin/acvr/experiments/dpa4c_gemini_20261113/runs}
[[ -z "$gate" || "$gate" =~ ^[0-9]+(:[0-9]+)*$ ]] || { echo 'Invalid Slurm dependency' >&2; exit 2; }
[[ -f "$work/job.sh" && -f "$work/src/experiments/dpa4c_tensor/train_runner.py" && -d "$origin" ]] || { echo 'Missing deployment or original runs' >&2; exit 1; }
[[ "$out" = /* && "$origin" = /* ]] || { echo 'Use absolute paths for results and parent runs' >&2; exit 2; }
for s in cumulene nacl aumgo sasa sabs sadma dimers; do
    parent=acvr
    [[ "$s" != aumgo ]] || parent=A3
    [[ "$s" != sasa ]] || parent=B2
    for f in resume.pt ckpt-020000.pt config.json test_final.json; do
        [[ -s "$origin/$s/3407/$parent/$f" ]] || { echo "Missing $s/$parent/$f" >&2; exit 1; }
    done
done
mkdir -p "$out"/{logs,dispatch/ids,dispatch/intent}
exec 9>"$out/dispatch/submit.lock"
flock -x 9
ledger="$out/jobs.tsv"
[[ -e "$ledger" ]] || printf 'system\tarm\tjob_id\tdependency\n' > "$ledger"
hash=$(printf '%s' "$out" | sha256sum | cut -c1-7)
for sys in cumulene nacl aumgo sasa sabs sadma dimers; do
    arms=(old qk2 orient2)
    if [[ "$sys" == cumulene ]]; then arms+=(orient12); fi
    for arm in "${arms[@]}"; do
        key="$sys-$arm"
        name="dt-$hash-$key"
        receipt="$out/dispatch/ids/$key"
        intent="$out/dispatch/intent/$key"
        if [[ -s "$receipt" ]]; then
            read -r job < "$receipt"
        elif [[ -e "$intent" ]]; then
            read -r day < "$intent"
            job=$(squeue -h -u "$USER" -n "$name" -o '%A' | sort -u)
            if [[ -z "$job" ]]; then
                job=$(sacct -X -n -P -u "$USER" -S "$day" --name="$name" -o JobIDRaw,JobName%100 | awk -F '|' -v n="$name" '$2==n && $1~/^[0-9]+$/{print $1}' | sort -u)
            fi
            [[ "$job" =~ ^[0-9]+$ ]] || { echo "Ambiguous submission $key, check Slurm and $intent" >&2; exit 1; }
        else
            date +%F > "$intent"
            opts=(--parsable --chdir="$work" --job-name="$name" --time=08:00:00 --output="$out/logs/%x-%j.out" --error="$out/logs/%x-%j.err")
            [[ -z "$gate" ]] || opts+=(--dependency="afterok:$gate")
            raw=$(sbatch "${opts[@]}" "$work/job.sh" ../dpa4c_tensor/train_runner.py --system "$sys" --arm "$arm" --steps 20000 --source-runs "$origin" --out-root "$out" --stage all)
            job=${raw%%;*}
        fi
        [[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid job id for $key: $job" >&2; exit 1; }
        printf '%s\n' "$job" > "$receipt.tmp"; mv -f "$receipt.tmp" "$receipt"
        if ! awk -F '\t' -v s="$sys" -v a="$arm" -v j="$job" '$1==s && $2==a && $3==j{v=1} END{exit !v}' "$ledger"; then
            printf '%s\t%s\t%s\t%s\n' "$sys" "$arm" "$job" "${gate:--}" >> "$ledger"
        fi
        printf '%s\t%s\t%s\n' "$sys" "$arm" "$job"
    done
done
