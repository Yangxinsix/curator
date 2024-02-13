#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=xeon40
#SBATCH -N 2     # Minimum of 1 node
#SBATCH -n 80     # 8 MPI processes per node
#SBATCH --time=50:00:00 # 2 days of runtime (can be set to 7 days)

module load GPAW/22.8.0-foss-2022a

mpiexec -n 80 python MD_sim.py --traj_path='LiFePO4_relax_5.traj'