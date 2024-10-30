# Jacobi stencil for GPU

## What is the Jacobi Stencil?

For the theory behind the Jacobi stencil please see [here](jacobi-description.pdf).
In practice, the algorithm is very simple because it just involves updating each cell of a grid (for simplicity, assume 2D) by the average of its neighbours:
```code
a[i,j] = 0.25*(a[i-1,j] + a[i+1,j] + a[i,j+1] + a[i,j-1])
```
We assume fixed boundary conditions so these cells are not updated.
Convergence is obtained when the difference between the grid and the updated grid is less than some predefined tolerance.

## CPU version

We provide a version written in C which can be compiled and run as follows (*WARNING*: minimal error checking):
```bash
make
./jacobi -x 100 -y 100
```
This will run the stencil for a grid of 100x100 with pre-defined boundary conditions
and should complete in 8111 iterations. You can also experiment with different grid sizes and boundary conditions.
NB: There is some code for storing the grid configurations for visualisation - for this exercise this can be disabled.


## Hints for CUDA version

The C version is in double precision. Is this precision needed?

First thing to do is to fix the number of thread blocks for the grid nx * ny :
```C
// GPU threads/block

  int blockSize=256;
  int numBlocks = ((nx*ny)+blockSize-1)/blockSize;
  printf("numBlocks=%d\n",numBlocks);
```

Before starting to optimise the main loop, you should allocate CUDA versions of the grids and copy them onto the device. Alternatively, use CUDA managed memory.

In the first case you could do something like:
```code
  REAL *d_grid, *d_grid_new;
  cudaMalloc(&d_grid,(nx+2)*(ny+2)*sizeof(REAL));
  cudaMalloc(&d_grid_new,(nx+2)*(ny+2)*sizeof(REAL));
  cudaMemcpy(d_grid,grid,(nx+2)*(ny+2)*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_grid_new,grid_new,(nx+2)*(ny+2)*sizeof(REAL), cudaMemcpyHostToDevice);
```
where REAL is `float` or `double`.

You should identify three kernels which should be ported for CUDA:
 1. the stencil  sum identified as above
 2. Calculation of the norm parameter (for testing convergence)
 3. Reduction of the norm parameters  over the cuda Blocks
 
 The first is easiest to do and can be done first.
 
 ### stencil sum
The stenci sum kernel (the jacobi average given above) can be called as follows from the main program:
```
stencil_sum<<<numBlocks, blockSize>>>(d_grid,d_grid_new,nx,ny); 
```
where both the current and new grids are on the device.

 



