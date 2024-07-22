#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --job-name=Convergence
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=XXXX


FILE=Converged_$1.mat

if [ -f "$FILE" ]; then
    echo "$FILE exists."
    flag=1

    if [ $1 == '3D' ] || [ $1 == '3D_syn' ]; then
        string=$(sbatch run_ensemble3D.sh $1 $flag)
    else
        string=$(sbatch run_ensemble1D.sh $1 $flag)
    fi

    jid0=$(echo $string | awk '{print $4}')
    string3=$(sbatch --dependency=afterok:$jid0  grab_output.sh $1)

else
    string4=$(sbatch loop.sh $1 $2)
    echo "EKI has not converged yet"
fi


