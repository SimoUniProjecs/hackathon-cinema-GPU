#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --ntasks-per-socket=1   # number of tasks per socket
#SBATCH --cpus-per-task=1       # number of HW threads per task (equal to OMP_NUM_THREADS*4)
#SBATCH --gres=gpu:1            # this refers to the number of requested gpus per node, and can vary between 1 and 4
#SBATCH --time 00:20:00         # format: HH:MM:SS
#SBATCH --output=cfd.out
#SBATCH -p boost_usr_prod
#SBATCH --exclusive

module purge
module load nvhpc

make

rm -rf /tmp/nvidia
ln -s $TMPDIR /tmp/nvidia
nsys profile --trace=nvtx --backtrace=dwarf --output=report_nvtx --force-overwrite=true ./cfd_nvtx.exe 200 50
rm -rf /tmp/nvidia

