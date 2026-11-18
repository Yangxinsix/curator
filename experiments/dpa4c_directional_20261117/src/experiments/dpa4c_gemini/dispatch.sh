#!/bin/bash
#SBATCH -A group_hzxy
#SBATCH --qos=qos_hzxy05
#SBATCH -p hcpu48,hcpu48y
#SBATCH -c 1
#SBATCH --time=00:20:00
# Submit in the deployed campaign root containing job.sh, src/, unified/ and runs/.
# start [all|aumgo|sasa] [existing-native-build-ID] is safe to repeat.
set -euo pipefail

mode=${1:-start}
system=${2:-all}
external_build=${3:-}
work=$PWD
src="$work/src/experiments/dpa4c_gemini"
driver="$src/dispatch.sh"
if [[ -f "$work/dispatch.sh" ]]; then driver="$work/dispatch.sh"; fi
launcher=${ACVR_CPU_LAUNCHER:-/data1/home/hzxy05/work/yangxin/acvr/runtime/curator-dev-cu010/run-acvr.sh}
state="$work/dispatch"
[[ -f "$work/job.sh" && -f "$driver" && -f "$src/followup.py" ]] || { echo 'Run from the deployed campaign root' >&2; exit 2; }
[[ "$system" =~ ^(aumgo|sasa|all)$ ]] || { echo "Unknown system: $system" >&2; exit 2; }
if [[ "$mode" != start && "$mode" != __submit_helpers && "$system" == all ]]; then echo 'Selector needs one system' >&2; exit 2; fi
[[ -z "$external_build" || "$external_build" =~ ^[0-9]+$ && "$mode" == start ]] || exit 2
mkdir -p "$state"/{intent,ids,manifests} "$work/logs"
exec 9>"$state/submit.lock"
flock -x 9
ledger="$work/jobs.tsv"
[[ -f "$ledger" ]] || printf 'task\tjob_id\tdependency\tcreated_utc\n' > "$ledger"
# Older receipts remain usable even after a Slurm job ages out of accounting.
# If the client exits between sbatch and its receipt, recover by the unique job
# name from Slurm instead of submitting another copy.
tag=$(printf '%s' "$work" | sha256sum | cut -c1-8)
record() {
    local key=$1 id=$2 dep=$3
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "Invalid job ID for $key: $id" >&2; return 1; }
    printf '%s\n' "$id" > "$state/ids/$key.tmp"
    mv -f "$state/ids/$key.tmp" "$state/ids/$key"
    if ! awk -F '\t' -v k="$key" -v j="$id" '$1==k && $2==j {f=1} END{exit !f}' "$ledger"; then
        printf '%s\t%s\t%s\t%s\n' "$key" "$id" "${dep:--}" "$(date -u +%FT%TZ)" >> "$ledger"
    fi
    job_id=$id
}

submit() {
    local key=$1 dep=$2 script=$3
    shift 3
    local name="d4-$tag-$key" id raw date_iso
    local opts=(--parsable --chdir="$work" --job-name="$name" --output="$work/logs/%x-%j.out" --error="$work/logs/%x-%j.err")
    if [[ -e "$state/ids/$key" ]]; then
        read -r id < "$state/ids/$key"
        record "$key" "$id" "$dep"
        return
    fi
    if [[ -e "$state/intent/$key" ]]; then
        read -r date_iso < "$state/intent/$key"
        # Full exact name and one exact allocation row; ignore .batch/.extern rows.
        id=$(squeue -h -u "$USER" -n "$name" -o '%A' | sort -u)
        if [[ -z "$id" ]]; then
            id=$(sacct -X -n -P -u "$USER" -S "$date_iso" --name="$name" -o JobIDRaw,JobName%100 | awk -F '|' -v n="$name" '$2==n && $1~/^[0-9]+$/{print $1}' | sort -u)
        fi
        if [[ "$id" =~ ^[0-9]+$ ]]; then
            record "$key" "$id" "$dep"
            return
        fi
        echo "Unresolved submit receipt for $name. Inspect Slurm and $state/intent/$key before retrying." >&2
        return 1
    fi
    date +%Y-%m-%d > "$state/intent/$key"
    if [[ -n "$dep" ]]; then opts+=(--dependency="afterok:$dep"); fi
    if ! raw=$(sbatch "${opts[@]}" "$script" "$@"); then
        # Slurm rejected the request; retain intent for lookup on retry when the
        # scheduler may have accepted it despite a lost client reply.
        echo "Submission not confirmed for $name" >&2
        return 1
    fi
    id=${raw%%;*}
    record "$key" "$id" "$dep"
    printf '%s\t%s\t%s\n' "$key" "$job_id" "${dep:--}"
}

join_ids() { local IFS=:; echo "$*"; }

train_score() {
    local sys=$1 seed=$2 arm=$3 dep=$4
    submit "$sys-$seed-$arm-fit" "$dep" "$work/job.sh" train.py --system "$sys" --seed "$seed" --arm "$arm"
    submit "$sys-$seed-$arm-val" "$job_id" "$work/job.sh" score.py --system "$sys" --seed "$seed" --arm "$arm" --stage validation
}

case "$mode" in
__submit_helpers) return 0 ;;
start)
    if [[ -n "$external_build" ]]; then
        if [[ -s "$state/ids/native-build" && "$(cat "$state/ids/native-build")" != "$external_build" ]]; then
            echo 'Native build ID conflicts with existing receipt' >&2; exit 1
        fi
        record native-build "$external_build" ""
    else
        submit native-build "" "$work/job.sh" compile_gpu.py
    fi
    build_id=$job_id
    systems=("$system")
    if [[ "$system" == all ]]; then systems=(aumgo sasa); fi
    for sys in "${systems[@]}"; do
        submit "$sys-base" "$build_id" "$work/job.sh" train.py --system "$sys" --seed 3407 --arm base
        base_id=$job_id
        prefix=A; if [[ "$sys" == sasa ]]; then prefix=B; fi
        validations=()
        for k in 0 1 2 3 4; do
            train_score "$sys" 3407 "$prefix$k" "$base_id"
            validations+=("$job_id")
        done
        submit "$sys-screen" "$(join_ids "${validations[@]}")" "$driver" screen "$sys"
    done
    ;;
screen|select-test)
    manifest="$state/manifests/$system-$mode.json"
    export SINGULARITY_BINDPATH="${SINGULARITY_BINDPATH:+$SINGULARITY_BINDPATH,}$work:$work"
    "$launcher" cpu python "$src/followup.py" "$mode" --runs "$work/runs" --system "$system" --output "$manifest"
    "$launcher" cpu python "$src/summarize_clock.py" --runs "$work/runs" --system "$system" --output "$state/manifests/$system-$mode-clock.csv"
    "$launcher" cpu python "$src/report.py" --runs "$work/runs" --out "$work/summary"
    rows=$("$launcher" cpu python -c 'import json,sys; tasks=json.load(open(sys.argv[1]))["tasks"]; assert tasks; print("\n".join("{system} {seed} {arm}".format(**t) for t in tasks))' "$manifest")
    validations=()
    tests=()
    while read -r sys seed arm; do
        [[ "$sys" == "$system" && "$seed" =~ ^340[789]$ && "$arm" =~ ^[AB][0-4]$ ]] || { echo "Invalid selection: $sys $seed $arm" >&2; exit 2; }
        if [[ "$mode" == screen ]]; then
            # screen is after all five validation jobs. The shared 3407 base is
            # already complete, including when a selector is restarted manually.
            train_score "$sys" "$seed" "$arm" ""
            validations+=("$job_id")
        else
            submit "$sys-$seed-$arm-test" "" "$work/job.sh" score.py --system "$sys" --seed "$seed" --arm "$arm" --stage test
            tests+=("$job_id")
        fi
    done <<< "$rows"
    if [[ "$mode" == screen ]]; then
        submit "$system-select-test" "$(join_ids "${validations[@]}")" "$driver" select-test "$system"
    else
        submit "$system-first-tests-complete" "$(join_ids "${tests[@]}")" "$src/extension_dispatch.sh" first-tests-complete "$system"
    fi
    ;;
*) echo 'Usage: bash dispatch.sh start [all|aumgo|sasa] [existing-native-build-ID]' >&2; exit 2 ;;
esac
