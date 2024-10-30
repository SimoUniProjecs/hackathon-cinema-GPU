#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --ntasks-per-socket=1   # number of tasks per socket
#SBATCH --cpus-per-task=1       # number of HW threads per task (equal to OMP_NUM_THREADS*4)
#SBATCH --gres=gpu:1            # this refers to the number of requested gpus per node, and can vary between 1 and 4
#SBATCH --time 00:10:00         # format: HH:MM:SS
#SBATCH --output=cfd.out
#SBATCH -p boost_usr_prod


module purge
module load nvhpc/24.3

make

./cfd_acc.exe 200 50

nsys profile --trace=openacc,nvtx --output=report_acc_data1 --force-overwrite=true --capture-range=nvtx --nvtx-capture='main_loop' --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0  ./cfd_acc.exe 200 50
