#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "jacobi.h"
#include <cuda_runtime.h>

__global__ void jacobi_norm_kernel(double *grid, double *grid_new, double *norm_array, int nx, int ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= ny && j <= nx) {
        int k = (nx + 2) * i + j;

        // Calcolo della nuova griglia usando la formula Jacobi
        grid_new[k] = 0.25 * (grid[k - 1] + grid[k + 1] + grid[k - (nx + 2)] + grid[k + (nx + 2)]);

        // Calcolo della norma residua
        double residue = grid[k] * 4.0 - grid[k - 1] - grid[k + 1] - grid[k - (nx + 2)] - grid[k + (nx + 2)];
        norm_array[(i - 1) * nx + (j - 1)] = residue * residue;
    }
}

__global__ void reduce_kernel(double *input, double *output, int n) {
    extern __shared__ double shared_data[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    shared_data[tid] = (i < n ? input[i] : 0.0) + (i + blockDim.x < n ? input[i + blockDim.x] : 0.0);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = shared_data[0];
}

double reduce(double *d_input, int n) {
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block * 2 - 1) / (threads_per_block * 2);
    double *d_intermediate, *d_result;

    cudaMalloc((void **)&d_intermediate, blocks_per_grid * sizeof(double));
    cudaMalloc((void **)&d_result, sizeof(double));

    reduce_kernel<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(double)>>>(d_input, d_intermediate, n);
    cudaDeviceSynchronize();

    if (blocks_per_grid > 1) {
        reduce_kernel<<<1, threads_per_block, threads_per_block * sizeof(double)>>>(d_intermediate, d_result, blocks_per_grid);
    } else {
        cudaMemcpy(d_result, d_intermediate, sizeof(double), cudaMemcpyDeviceToDevice);
    }

    double result;
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_intermediate);
    cudaFree(d_result);

    return result;
}

int main(int argc, char *argv[]) {
    int nx, ny;
    double left = LEFT, right = RIGHT, top = TOP, bottom = BOTTOM;
    int ndump = NDUMP;
    int dump = 1;

    if (argc < 3) {
        usage(argv);
        exit(1);
    }
    get_options(argc, argv, &nx, &ny, &ndump, &left, &right, &top, &bottom);
    if (ndump == 0) {
        dump = 0;
        printf("Dumping frames off\n");
    }

    printf("grid size %d X %d \n", nx, ny);
    size_t grid_size = sizeof(double) * (nx + 2) * (ny + 2);
    double *h_grid = (double *)malloc(grid_size);
    double *h_grid_new = (double *)malloc(grid_size);

    init_grids(h_grid, h_grid_new, nx, ny, left, right, top, bottom);

    double *d_grid, *d_grid_new, *d_norm_array;
    size_t norm_array_size = sizeof(double) * nx * ny;

    cudaMalloc((void **)&d_grid, grid_size);
    cudaMalloc((void **)&d_grid_new, grid_size);
    cudaMalloc((void **)&d_norm_array, norm_array_size);

    cudaMemcpy(d_grid, h_grid, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_new, h_grid_new, grid_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);

    jacobi_norm_kernel<<<gridDim, blockDim>>>(d_grid, d_grid_new, d_norm_array, nx, ny);
    cudaDeviceSynchronize();

    double tmpnorm = reduce(d_norm_array, nx * ny);
    double bnorm = sqrt(tmpnorm);

    struct timeval begin, end;
    gettimeofday(&begin, 0);

    if (dump)
        dump_grid(h_grid, nx, ny);

    int iter;
    double norm;
    for (iter = 0; iter < MAX_ITER; iter++) {
        jacobi_norm_kernel<<<gridDim, blockDim>>>(d_grid, d_grid_new, d_norm_array, nx, ny);
        cudaDeviceSynchronize();

        tmpnorm = reduce(d_norm_array, nx * ny);
        norm = sqrt(tmpnorm) / bnorm;

        if (norm < TOLERANCE)
            break;

        double *tmp = d_grid;
        d_grid = d_grid_new;
        d_grid_new = tmp;

        if (dump && iter % ndump == 0) {
            cudaMemcpy(h_grid, d_grid, grid_size, cudaMemcpyDeviceToHost);
            dump_grid(h_grid, nx, ny);
        }

        if (iter % NPRINT == 0)
            printf("Iteration =%d ,Relative norm=%e\n", iter, norm);
    }

    printf("Terminated on %d iterations, Relative Norm=%e \n", iter, norm);
    printf("Frames stored=%d \n", get_num_frames());

    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds * 1e-6;
    printf("Time measured: %.3f seconds.\n", elapsed);

    cudaMemcpy(h_grid, d_grid, grid_size, cudaMemcpyDeviceToHost);

    free(h_grid);
    free(h_grid_new);

    cudaFree(d_grid);
    cudaFree(d_grid_new);
    cudaFree(d_norm_array);

    return 0;
}
