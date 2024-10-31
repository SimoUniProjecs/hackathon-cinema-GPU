
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "jacobi.h"
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <nvtx3/nvToolsExt.h>

// Kernel to execute the jacobi algorithm
__global__ void jacobi_kernel(float *grid, float *grid_new, int nx, int ny)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= ny && j <= nx)
    {
        int k = (nx + 2) * i + j;
        grid_new[k] = 0.25 * (grid[k - 1] + grid[k + 1] + grid[k - (nx + 2)] + grid[k + (nx + 2)]);
    }
}

// Kernel to compute the residual norm
__global__ void compute_norm_kernel(float *grid, float *norm_array, int nx, int ny)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1; 
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= ny && j <= nx)
    {
        int k = (nx + 2) * i + j;
        float residue = grid[k] * 4.0 - grid[k - 1] - grid[k + 1] - grid[k - (nx + 2)] - grid[k + (nx + 2)];
        norm_array[(i - 1) * nx + (j - 1)] = residue * residue;
    }
}

int main(int argc, char *argv[])
{
    int nx, ny;
    float left = LEFT, right = RIGHT, top = TOP, bottom = BOTTOM;
    int ndump = NDUMP;
    int dump = 1;

    if (argc < 3)
    {
        usage(argv);
        exit(1);
    }
    get_options(argc, argv, &nx, &ny, &ndump, &left, &right, &top, &bottom);
    if (ndump == 0)
    {
        dump = 0;
        printf("Dumping frames off\n");
    }

    printf("grid size %d X %d \n", nx, ny);
    size_t grid_size = sizeof(float) * (nx + 2) * (ny + 2);
    float *h_grid = (float *)malloc(grid_size);
    float *h_grid_new = (float *)malloc(grid_size);

    init_grids(h_grid, h_grid_new, nx, ny, left, right, top, bottom);

    // Allocation of memory
    float *d_grid, *d_grid_new, *d_norm_array;
    size_t norm_array_size = sizeof(float) * nx * ny;

    cudaMalloc((void **)&d_grid, grid_size);
    cudaMalloc((void **)&d_grid_new, grid_size);
    cudaMalloc((void **)&d_norm_array, norm_array_size);

    // Copy data from host to device
    cudaMemcpy(d_grid, h_grid, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_new, h_grid_new, grid_size, cudaMemcpyHostToDevice);

    // Compute initial bnorm
    dim3 blockDim(16, 16);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);

    nvtxRangePushA("compute norm kernel");
    compute_norm_kernel<<<gridDim, blockDim>>>(d_grid, d_norm_array, nx, ny);
    cudaDeviceSynchronize();
    nvtxRangePop();

    // Sum the norm_array
    thrust::device_ptr<float> dev_ptr(d_norm_array);
    float tmpnorm = thrust::reduce(dev_ptr, dev_ptr + nx * ny, 0.0, thrust::plus<float>());
    float bnorm = sqrt(tmpnorm);

    // Start timer
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    // Start grid file
    if (dump)
        dump_grid(h_grid, nx, ny);

    // Main iteration loop
    int iter;
    float norm;
    nvtxRangePushA("main loop");
    for (iter = 0; iter < MAX_ITER; iter++)
    {
        float tmpnorm;
        nvtxRangePushA("Norm");
        compute_norm_kernel<<<gridDim, blockDim>>>(d_grid, d_norm_array, nx, ny);
        cudaDeviceSynchronize();

        // Sum the norm_array
        tmpnorm = thrust::reduce(dev_ptr, dev_ptr + nx * ny, 0.0, thrust::plus<float>());

        norm = sqrt(tmpnorm) / bnorm;

        if (norm < TOLERANCE)
            break;

        nvtxRangePop();

        // Update grid

        nvtxRangePushA("Stencil");
        jacobi_kernel<<<gridDim, blockDim>>>(d_grid, d_grid_new, nx, ny);
        cudaDeviceSynchronize();
        nvtxRangePop();

        // Swap pointers
        float *tmp = d_grid;
        d_grid = d_grid_new;
        d_grid_new = tmp;

        if (dump && iter % ndump == 0)
        {
            // Copy grid back to host
            cudaMemcpy(h_grid, d_grid, grid_size, cudaMemcpyDeviceToHost);
            dump_grid(h_grid, nx, ny);
        }

        if (iter % NPRINT == 0)
            printf("Iteration =%d ,Relative norm=%e\n", iter, norm);
    }
    nvtxRangePop();

    printf("Terminated on %d iterations, Relative Norm=%e \n", iter, norm);
    printf("Frames stored=%d \n", get_num_frames());

    // Stop timing
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    float elapsed = seconds + microseconds * 1e-6;
    printf("Time measured: %.3f seconds.\n", elapsed);

    // Save results to CSV
    /*FILE *file = fopen("results_1step.csv", "a");
    if (file == NULL) {
        perror("Error opening file");
        exit(1);
    }
    fprintf(file, "%d,%.3f\n", nx, elapsed);
    fclose(file);*/

    // Copy final grid back to host
    cudaMemcpy(h_grid, d_grid, grid_size, cudaMemcpyDeviceToHost);

    free(h_grid);
    free(h_grid_new);

    // Free device memory
    cudaFree(d_grid);
    cudaFree(d_grid_new);
    cudaFree(d_norm_array);

    return 0;
}
