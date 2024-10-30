#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "jacobi.h"

__global__
jacobi (double *grid, double *grid_new, int nx, int ny ) {
    // Mi trovo l'indice Globale:
    int idx = threadIdx.x + (blockIdx.x * blockDim.x)
    // Calcolo gli indici precisi
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx<(nx*ny)&&((i<=nx)&&(i>0))&&((y<ny)&&(y>0))) {
        grid_new[idx]=0.25*(grid[idx%nx-1,idx%ny]+grid[idx%nx+1,idx%ny]+grid[idx%nx, idx%ny-1]+grid[idx%nx, idx%ny+1]);
    }
    double *tmp;
    tmp = grid;
    grid = grid_new;
    grid_new = tmp;
}

__global__
normale(double *grid, nx, ny)   {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x)
    if(idx<(nx*ny)&&((idx%nx<=nx)&&(idx%nx>0))&&((idx%ny<ny)&&(idx%ny>0))) {
        int k;
        k = (nx+2) * (idx%nx)
    }
}

__device__
swap

int main(int argc, char *argv[])
{

    int nx, ny;
    double left = LEFT, right = RIGHT, top = TOP, bottom = BOTTOM;
    int ndump = NDUMP;
    int dump = 1;
    int blockSize = 256;

    if (argc < 3) // Parametri insufficienti
    {
        usage(argv);
        exit(1);
    }
    else
    { // Inizializzazione dei parametri
        get_options(argc, argv, &nx, &ny, &ndump, &left, &right, &top, &bottom);
    }

    // Calcolo il numero di Blocchi
    int numBlocks = ((nx * ny) + blockSize - 1) / blockSize;
    printf("numBlocks=%d\n", numBlocks);

    // Alloco la memoria per la matrice che poi dovrò copiare su DEVICE
    printf("grid size %d X %d \n", nx, ny);
    double *h_grid = (double *)malloc(sizeof(double) * (nx + 2) * (ny + 2));
    double *h_grid_new = (double *)malloc(sizeof(double) * (nx + 2) * (ny + 2));

    // Alloco la memoria per IL DEVICE
    double *d_grid;
    double *d_grid_new;
    checkCuda(cudaMalloc(d_grid, (double *)malloc(sizeof(double) * (nx + 2) * (ny + 2))));
    checkCuda(cudaMalloc(d_grid_new, (double *)malloc(sizeof(double) * (nx + 2) * (ny + 2))));

    // Passo la memoria sul DEVICE    
    cudaMemcpy(d_grid, h_grid, nx*ny, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_new, h_grid_new, nx*ny, cudaMemcpyHostToDevice);

    // Iterazione
    for(int i = 0; i < MAX_ITER; i++){
        start = cpuSecond();
        jacobi<<<(((nx+2)*(ny+2))/blockSize),blockSize>>>(grid, grid_new, nx,ny);
        cudaDeviceSynchronize();
        double gpuTime = cpuSecond() - start;
        printf("GPU execution time: %f, seconds\n", gpuTime);
    }

    return 0;
}
