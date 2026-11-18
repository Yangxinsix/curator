#!/bin/bash
#SBATCH -A group_hzxy
#SBATCH --qos=qos_hzxy05
#SBATCH -p hcpu48,hcpu48y
#SBATCH -c 1
#SBATCH --time=00:20:00
# Run via an afterok dependency, in the same campaign root as dispatch.sh.
set -euo pipefail
extension_stage=${1:?Expected first-tests-complete or expand}
extension_system=${2:-}
campaign_src="$PWD/src/experiments/dpa4c_gemini"
# Share one campaign lock, atomic receipts and GPU resource policy with round one.
source "$campaign_src/dispatch.sh" __submit_helpers all

extension_seed=3407
selection="dispatch/manifests/sasa-select-test.json"
extension_driver="$campaign_src/extension_dispatch.sh"
submit_extension() {
    local sys=$1
    local -a options=(--system "$sys" --seed "$extension_seed" --selection "$selection")
    submit "ext-$sys-base" "" "$work/job.sh" extension_run.py "${options[@]}" --arm base --stage train
    local base_id=$job_id
    local arm fit_id
    for arm in pure acvr les; do
        submit "ext-$sys-$arm-fit" "$base_id" "$work/job.sh" extension_run.py "${options[@]}" --arm "$arm" --stage train
        fit_id=$job_id
        submit "ext-$sys-$arm-val" "$fit_id" "$work/job.sh" extension_run.py "${options[@]}" --arm "$arm" --stage validation
        extension_validations+=("$job_id")
        # All three methods are predetermined. Only selector data are validation
        # results from SA-SA; none of the held-out errors select a subsequent run.
        submit "ext-$sys-$arm-test" "$job_id" "$work/job.sh" extension_run.py "${options[@]}" --arm "$arm" --stage test
        extension_tests+=("$job_id")
    done
}

extension_tests=()
mkdir -p "$state/completed"
case "$extension_stage" in
first-tests-complete)
    [[ "$extension_system" == sasa || "$extension_system" == aumgo ]] || exit 2
    # These callbacks run after each first-round system's six tests succeed.
    # Durable markers allow the two systems to finish days apart, past Slurm's
    # MinJobAge. An interrupted callback safely resumes from per-task receipts.
    touch "$state/completed/$extension_system-first-tests"
    if [[ ! -f "$state/completed/sasa-first-tests" || ! -f "$state/completed/aumgo-first-tests" ]]; then
        echo "Recorded $extension_system first-round tests; awaiting other system."
        exit 0
    fi
    [[ -s "$selection" ]] || { echo "Missing SA-SA validation selection $selection" >&2; exit 1; }
    extension_validations=()
    for sys in nacl sabs; do submit_extension "$sys"; done
    submit "ext-first-report" "$(join_ids "${extension_tests[@]}")" "$extension_driver" report
    submit "ext-expand" "$(join_ids "${extension_validations[@]}")" "$extension_driver" expand
    ;;
expand)
    [[ -s "$selection" && -f "$state/completed/sasa-first-tests" && -f "$state/completed/aumgo-first-tests" ]] || { echo 'Missing initial confirmations' >&2; exit 1; }
    extension_validations=()
    for sys in cumulene dimers sadma; do submit_extension "$sys"; done
    submit "ext-final-report" "$(join_ids "${extension_tests[@]}")" "$extension_driver" report
    ;;
report)
    export SINGULARITY_BINDPATH="${SINGULARITY_BINDPATH:+$SINGULARITY_BINDPATH,}$work:$work"
    "$launcher" cpu python "$src/report.py" --runs "$work/runs" --out "$work/summary"
    ;;
*) echo 'Unknown extension stage' >&2; exit 2 ;;
esac
