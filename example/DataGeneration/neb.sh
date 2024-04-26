#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=xeon40
#SBATCH -N 3      # Minimum of 1 node
#SBATCH -n 120     # 8 MPI processes per node
#SBATCH --time=50:00:00 # 2 days of runtime (can be set to 7 days)

module load GPAW/22.8.0-foss-2022a

mpiexec -n 120 python /home/energy/mahpe/Curator_new/test_data/LiFePO4/NEB_relax.py