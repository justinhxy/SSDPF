#!/bin/bash
#SBATCH -J wzp_test
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH -A F00120230017
##################################################################

module load cuda11.3/toolkit/11.3.0
function Func1(){
    cal=1
    sleep 5
    while true
    do
    nvidia-smi
    cal=$(($cal+1))
    if [ $cal -gt 10 ]
    then break
    fi
    sleep 2
    done
}

function Func2(){
    sh -u /mntcephfs/lab_data/wangcm/wangzhipeng/pe_main/test.sh
}
Func1&Func2