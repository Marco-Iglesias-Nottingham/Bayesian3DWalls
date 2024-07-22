#!/bin/bash
#SBATCH --exclude=node055,node150,node151
#SBATCH --time=00:10:00
#SBATCH --job-name=Loop
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --account=XXXX


if [ $1 == '3D' ] || [ $1 == '3D_syn' ]; then
    string=$(sbatch run_ensemble3D.sh $1 $2)
else
    string=$(sbatch run_ensemble1D.sh $1 $2)
fi

jid1=$(echo $string | awk '{print $4}')
echo $string
echo "Part 1 run_ensemble.sh done"


string2=$(sbatch --dependency=afterok:$jid1 invert.sh  $1)
string2B=$(sbatch --dependency=afternotok:$jid1 convergence.sh  $1 $2)
echo "Part 3 Second.sh done"
jid2=$(echo $string2 | awk '{print $4}')
string3=$(sbatch --dependency=afterok:$jid2 convergence.sh $1 $2)

