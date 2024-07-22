#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=init
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=XXXX


module purge


echo "Running on `hostname`"
cd $SLURM_SUBMIT_DIR

~/miniconda3/envs/fenicsproject/bin/python setup_$1.py


