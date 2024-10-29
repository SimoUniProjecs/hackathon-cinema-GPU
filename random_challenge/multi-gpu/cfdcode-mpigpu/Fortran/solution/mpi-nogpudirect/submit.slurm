#!/bin/bash
#SBATCH --time=00:10:00           # Walltime in hh:mm:ss
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mpi
#SBATCH --output=cfd.out
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition=boost_usr_prod
#SBATCH -q boost_qos_dbg

module purge
module load nvhpc/24.3
module load openmpi/4.1.6--nvhpc--24.3

make

mpirun --map-by socket:PE=8 --rank-by core -np 4 cfd 200 500 

nsys profile -o report-mpi --force-overwrite=true --trace=mpi,openacc,nvtx mpirun --map-by socket:PE=8 --rank-by core -np 4 cfd 200 500 

nsys stats report-mpi.nsys-rep > summary-all.out

#mpirun --map-by socket:PE=8 --rank-by core -np 4 nsys profile -o report-mpi%q{OMPI_COMM_WORLD_RANK} --force-overwrite=true --trace=openacc,mpi,cuda cfd 200 5000 

