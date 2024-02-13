#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=xeon40
#SBATCH -N 1      # Minimum of 1 node
#SBATCH -n 40     # 8 MPI processes per node
#SBATCH --time=12:00:00 # 2 days of runtime (can be set to 7 days)

module load GPAW/22.8.0-foss-2022a

mpiexec -n 40 python LiFePO4_test.py