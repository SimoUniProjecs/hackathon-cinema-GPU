#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=1       # number of HW threads per task (equal to OMP_NUM_THREADS*4)
#SBATCH --gres=gpu:1            # this refers to the number of requested gpus per node, and can vary between 1 and 4
#SBATCH --time 00:10:00         
#SBATCH -p boost_usr_prod
#SBATCH -q boost_qos_dbg

module purge
module load nvhpc/24.3

make clean
make

nsys profile --trace=cuda,nvtx ./binary

