#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "jacobi.h"


__global__
void stencil_sum(float*grid, float *grid_new, int nx, int ny)
{
  int index=blockIdx.x * blockDim.x +threadIdx.x; // global thread id

  int nrow=index/ny;
  int diff=index-(nrow*ny);
  int k=(nrow+1)*(nx+2)+diff+1;

  if (index<nx*ny) 
      grid_new[k]=0.25 * (grid[k-1]+grid[k+1] + grid[k-(nx+2)] + grid[k+(nx+2)]);
}

__global__
void stencil_norm(float*grid, float*arraynorm, int nx, int ny)
{
  int index=blockIdx.x * blockDim.x +threadIdx.x; // globEl thread id
  
  int nrow=index/ny;
  int diff=index-(nrow*ny);
  int k=(nrow+1)*(nx+2)+diff+1;

  if (index<nx*ny)
     arraynorm[index]=(float)pow(grid[k]*4.0-grid[k-1]-grid[k+1] - grid[k-(nx+2)] - grid[k+(nx+2)], 2);

}

//   
//  Taken from CUDA document. Uses  Reduce v4. 
//  Partial sums performed for each block
//  

__global__
void reduce(float* g_idata, float *g_odata, int nx, int ny) {
extern __shared__ float sdata[];

  int tid=threadIdx.x;
  int i=blockIdx.x*(blockDim.x*2) + threadIdx.x;

  if ( (i+blockDim.x) < (nx*ny) ) 
     sdata[tid]=g_idata[i]+g_idata[i+blockDim.x];
  else
     sdata[tid]=0.0;

  __syncthreads();

  for(int s=blockDim.x/2;s>0;s>>=1) {
     if (tid<s) {
        sdata[tid] += sdata[tid+s];
     }
     __syncthreads();
  }
  if (tid ==0) { 
      g_odata[blockIdx.x]=sdata[0];
  }
}

void getDeviceInfo() {

  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    if  (cudaGetDeviceProperties(&prop, i) != cudaSuccess ) {
	printf("GPUs not found. Exiting\n");
	exit(1);
    }

    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }

}


// MAIN LOOP 
int main(int argc, char*argv[]) {

  int i,j,k;
  int nx, ny;
  float tmpnorm,bnorm,norm;
  float left = LEFT, right = RIGHT, top = TOP, bottom = BOTTOM; // default boundary conditions

  int ndump=NDUMP;
  int dump=1;
  printf("Jacobi 4-point stencil\n");
  printf("----------------------\n\n");
  
  if (argc < 3)
  {
    usage(argv);
    exit(1);
  }

  get_options(argc, argv, &nx, &ny, &ndump, &left, &right, &top, &bottom);
  if (ndump == 0) {
	  printf("Frame dumping off\n");
	  dump=0;
  }

  //int ny2 = ny + 2;

  // GPU info
  getDeviceInfo();

 // One device
  cudaSetDevice(0);

  printf("grid size %d X %d \n",nx,ny);

// GPU threads/block

  int blockSize=256;
  int numBlocks = ((nx*ny)+blockSize-1)/blockSize;
  printf("numBlocks=%d\n",numBlocks);

//
// host allocated memory
//

  float *grid= (float*)malloc(sizeof(float)*(nx+2)*(ny+2));
  float *grid_new= (float*)malloc(sizeof(float)*(nx+2)*(ny+2));
  float *arraynorm= (float*)malloc(sizeof(float)*nx*ny);
  float*blocknorm=(float*)malloc(sizeof(float)*numBlocks);

  //
  // Device allocated memory
  //

  float *d_grid, *d_grid_new, *d_arraynorm, *d_blocknorm;
  cudaMalloc(&d_grid,(nx+2)*(ny+2)*sizeof(float));
  cudaMalloc(&d_grid_new,(nx+2)*(ny+2)*sizeof(float));
  cudaMalloc(&d_arraynorm,nx*ny*sizeof(float));
  cudaMalloc(&d_blocknorm,numBlocks*sizeof(float)); 

// shared memory size on GPU 
  int smemsize=blockSize*sizeof(float);

  init_grids(grid, grid_new, nx, ny, left, right, top, bottom);
  
  // initial norm factor
  tmpnorm=0.0;
  for (i=1;i<=ny;i++) {
    for (j=1;j<=nx;j++) {
      k=(nx+2)*i+j;            
      tmpnorm=tmpnorm+(float)pow(grid[k]*4.0-grid[k-1]-grid[k+1] - grid[k-(nx+2)] - grid[k+(nx+2)], 2); 
    }
  }
  bnorm=sqrt(tmpnorm);


// copy arrays to device

  cudaMemcpy(d_grid,grid,(nx+2)*(ny+2)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_grid_new,grid_new,(nx+2)*(ny+2)*sizeof(float), cudaMemcpyHostToDevice);


// start grid file
  if (dump) dump_grid(grid, nx, ny);

//  start  timing **
    struct timeval begin, end;
    gettimeofday(&begin, 0); 

//    MAIN LOOP 
  int iter;
  for (iter=0; iter<MAX_ITER; iter++) {

    // calculate norm array
    stencil_norm<<<numBlocks,blockSize>>>(d_grid,d_arraynorm,nx,ny); 
    
    // perform reduction
    reduce<<<numBlocks,blockSize,smemsize>>>(d_arraynorm,d_blocknorm,nx,ny);
    cudaMemcpy(blocknorm,d_blocknorm,numBlocks*sizeof(float),cudaMemcpyDeviceToHost);
 
    // sum up temporary block sums
    tmpnorm=0.0;
    for (i=0;i<numBlocks;i++) {
      tmpnorm=tmpnorm+blocknorm[i];
    }
   
    norm=(float)sqrt(tmpnorm)/bnorm;

    if (norm < TOLERANCE) break;

    stencil_sum<<<numBlocks,blockSize>>>(d_grid,d_grid_new,nx,ny);

  // Wait for GPU to finish
  cudaDeviceSynchronize();

    float *temp=d_grid_new;
    d_grid_new=d_grid;
    d_grid=temp;

    // This is expensive - do as little as possible
    if (dump && iter % ndump == 0) {
      cudaMemcpy(grid,d_grid,(nx+2)*(ny+2)*sizeof(float), cudaMemcpyDeviceToHost);
      dump_grid(grid, nx, ny);
    }
      
    if (iter % NPRINT ==0) printf("Iteration =%d ,Relative norm=%e\n",iter,norm);
  }

  printf("Terminated on %d iterations, Relative Norm=%e \n", iter,norm);
  printf("Frames dumped %d \n", get_num_frames() );
  
// stop timing **
  gettimeofday(&end, 0);
  long seconds = end.tv_sec - begin.tv_sec;
  long microseconds = end.tv_usec - begin.tv_usec;
  double elapsed = seconds + microseconds*1e-6;
  printf("Time measured: %.3f seconds.\n", elapsed);

  cudaFree(d_grid);
  cudaFree(d_grid_new);
  cudaFree(d_arraynorm);

  free(grid);
  free(grid_new);
  free(arraynorm);

  return 0;
    

  }
