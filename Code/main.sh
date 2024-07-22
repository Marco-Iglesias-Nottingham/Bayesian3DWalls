#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=7770
#SBATCH --account=XXXXX



flag=0

string=$(sbatch initialise.sh $1)
jid0=$(echo $string | awk '{print $4}')

string=$(sbatch --dependency=afterok:$jid0  loop.sh $1 $flag)

