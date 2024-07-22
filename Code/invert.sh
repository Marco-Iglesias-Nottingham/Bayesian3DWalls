#!/bin/bash
#SBATCH --exclude=node055
#SBATCH --time=02:10:00
#SBATCH --job-name=Invert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=XXXX


~/miniconda3/envs/fenicsproject/bin/python ./Tools_EKI/KalmanUpdate.py $1



