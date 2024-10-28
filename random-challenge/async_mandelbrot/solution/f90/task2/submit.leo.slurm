#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --ntasks-per-socket=1   # number of tasks per socket
#SBATCH --cpus-per-task=1       # number of HW threads per task (equal to OMP_NUM_THREADS*4)
#SBATCH --gres=gpu:1            # this refers to the number of requested gpus per node, and can vary between 1 and 4
#SBATCH --time 00:20:00         # format: HH:MM:SS
#SBATCH --output=mand.out
#SBATCH -p boost_usr_prod
#SBATCH --exclusive

module purge
module load nvhpc/24.3
make

./mandelbrot.x 2
./mandelbrot.x 4
./mandelbrot.x 16

nsys profile --trace=openacc,cuda --output=report-mandelbrot --force-overwrite=true ./mandelbrot.x

