#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclude=node055,node150,node151
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3850
#SBATCH --time=03:00:00
#SBATCH --account=XXXX
module purge
module load GCCcore/10.2.0
module load parallel/20210322

MY_PARALLEL_OPTS="-N 1 --delay .2 -j $SLURM_NTASKS --joblog parallel-${SLURM_JOBID}.log"
# srun itself should launch 1 instance of our program and not oversubscribe resources
MY_SRUN_OPTS="-N 1 -n 1 --exclusive"
# Use parallel to launch srun with these options


parallel $MY_PARALLEL_OPTS srun $MY_SRUN_OPTS ~/miniconda3/envs/fenicsproject/bin/python ./Tools_EKI/ForwardRun1D.py  ::: {1..10} ::: $2 ::: $1


