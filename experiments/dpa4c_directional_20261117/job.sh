#!/bin/bash
#SBATCH -A group_hzxy
#SBATCH --qos=qos_hzxy05
#SBATCH -p hgpu2p
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --time=04:00:00
set -euo pipefail
work=$(cd "$(dirname "$(readlink -f "${SLURM_SUBMIT_DIR:-$PWD}/job.sh")")" && pwd)
cd "$work"
project=/data1/home/hzxy05/work/yangxin/acvr
cuda=/data0/software/cuda/12.8.0
export SINGULARITY_BINDPATH="$project:$project,$cuda:/usr/local/cuda"
export SINGULARITYENV_CURATOR_DPA4C_SOURCE="$work/unified" SINGULARITYENV_ACVR_SOURCE="$work/src/experiments/periodic_qefa/curator_source"
export SINGULARITYENV_CUDA_HOME=/usr/local/cuda SINGULARITYENV_TORCH_EXTENSIONS_DIR="$work/extensions" SINGULARITYENV_MAX_JOBS=3
export SINGULARITYENV_PYTHONUNBUFFERED=1 SINGULARITYENV_ACVR_JOB_ID="$SLURM_JOB_ID"
mkdir -p extensions
"$project/runtime/curator-dev-cu010/run-acvr.sh" gpu sh -c '
    export PATH="/usr/local/cuda/bin:$PATH"
    # Triton CUDA helpers in this SIF resolve their driver symbols in the process.
    export LD_PRELOAD="libcuda.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
    # Each compute capability has its own extension cache and compiler flags.
    arch=$(python -c "import torch; a,b=torch.cuda.get_device_capability(); print(str(a)+chr(46)+str(b))")
    export TORCH_CUDA_ARCH_LIST="$arch" TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR/$arch"
    exec python -u "$@"
' sh "src/experiments/dpa4c_gemini/${1:?script}" "${@:2}"
