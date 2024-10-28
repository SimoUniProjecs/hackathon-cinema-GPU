#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --ntasks-per-socket=1   # number of tasks per socket
#SBATCH --cpus-per-task=1       # number of HW threads per task (equal to OMP_NUM_THREADS*4)
#SBATCH --gres=gpu:1            # this refers to the number of requested gpus per node, and can vary between 1 and 4
#SBATCH --time 00:10:00         # format: HH:MM:SS
#SBATCH --gres=gpu:1
#SBATCH --output=cfd.out
#SBATCH --partition=boost_usr_prod
#SBATCH --exclusive

module purge
module load nvhpc/24.3

make

# uninstrumented run, use time as a reference
./cfd_serial.exe 200 50

# instrumented run
rm -rf /tmp/nvidia
ln -s $TMPDIR /tmp/nvidia
nsys profile --trace=none --output=report_serial --force-overwrite=true --backtrace=dwarf ./cfd_serial.exe 200 50
rm -rf /tmp/nvidia
