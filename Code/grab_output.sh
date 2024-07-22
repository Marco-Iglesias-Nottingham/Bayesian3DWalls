#!/bin/bash
#SBATCH --exclude=node022
#SBATCH --time=01:00:00
#SBATCH --job-name=init
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=XXXX



echo "Running on `hostname`"

~/miniconda3/envs/fenicsproject/bin/python ./Tools_EKI/outputs$1.py 



 
