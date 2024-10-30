#!/bin/bash
#SBATCH --time=00:20:00           # Walltime in hh:mm:ss
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mpi
#SBATCH --output=cfd.out
#SBATCH --exclusive
#SBATCH --partition=boost_usr_prod

module purge
module load nvhpc/24.3
module load openmpi/4.1.6--nvhpc--24.3

make

mpirun --map-by socket:PE=8 --rank-by core -np 4 cfd 200 500

nsys profile --trace=mpi,nvtx mpirun --map-by socket:PE=8 --rank-by core -np 4 cfd 200 500
