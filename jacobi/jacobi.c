#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "jacobi.h"

int main(int argc, char *argv[])
{

  int i, j, k;
  double tmpnorm, bnorm, norm;

  int nx, ny;
  double left = LEFT, right = RIGHT, top = TOP, bottom = BOTTOM;
  int ndump=NDUMP;
  int dump=1;

  if (argc < 3)
  {
    usage(argv);
    exit(1);
  }
  get_options(argc, argv, &nx, &ny, &ndump, &left, &right, &top, &bottom);
  if (ndump==0) {
	  dump=0;
	  printf("Dumping frames off\n");
  }

  int ny2 = ny + 2;

  printf("grid size %d X %d \n", nx, ny);
  double *grid = (double *)malloc(sizeof(double) * (nx + 2) * (ny + 2));
  double *grid_new = (double *)malloc(sizeof(double) * (nx + 2) * (ny + 2));
  double *temp;

  init_grids(grid, grid_new, nx, ny, left, right, top, bottom);

  // Initial norm factor
  tmpnorm = 0.0;
  for (i = 1; i <= ny; i++)
  {
    for (j = 1; j <= nx; j++)
    {
      k = (nx + 2) * i + j;
      tmpnorm = tmpnorm + (double)pow(grid[k] * 4.0 - grid[k - 1] - grid[k + 1] - grid[k - (nx + 2)] - grid[k + (nx + 2)], 2);
    }
  }
  bnorm = sqrt(tmpnorm);

  // Start measuring time
  struct timeval begin, end;
  gettimeofday(&begin, 0);

  // start grid file
  if (dump) dump_grid(grid, nx, ny);

  //    MAIN LOOP
  int iter;
  for (iter = 0; iter < MAX_ITER; iter++)
  {

    tmpnorm = 0.0;

    // Calculate norm factor
    for (i = 1; i <= ny; i++)
    {
      for (j = 1; j <= nx; j++)
      {
        k = (nx + 2) * i + j;
        tmpnorm = tmpnorm + (double)pow(grid[k] * 4.0 - grid[k - 1] - grid[k + 1] - grid[k - (nx + 2)] - grid[k + (nx + 2)], 2);
      }
    }

    norm = (double)sqrt(tmpnorm) / bnorm;

    if (norm < TOLERANCE)
      break;

      // Update grid
    for (i = 1; i <= ny; i++)
    {
      for (j = 1; j <= nx; j++)
      {
        k = (nx + 2) * i + j;
        grid_new[k] = 0.25 * (grid[k - 1] + grid[k + 1] + grid[k - (nx + 2)] + grid[k + (nx + 2)]);
      }
    }

    temp = grid;
    grid = grid_new;
    grid_new = temp;

    if (dump && iter % ndump == 0)
      dump_grid(grid, nx, ny);

    if (iter % NPRINT == 0)
      printf("Iteration =%d ,Relative norm=%e\n", iter, norm);
  }

  printf("Terminated on %d iterations, Relative Norm=%e \n", iter, norm);
  printf("Frames stored=%d \n",get_num_frames());

  // stop timing **
  gettimeofday(&end, 0);
  long seconds = end.tv_sec - begin.tv_sec;
  long microseconds = end.tv_usec - begin.tv_usec;
  double elapsed = seconds + microseconds * 1e-6;
  printf("Time measured: %.3f seconds.\n", elapsed);

  // Save results to CSV
  FILE *file = fopen("results_1step.csv", "a");
  if (file == NULL) {
      perror("Error opening file");
      exit(1);
  }
  fprintf(file, "%d,%.3f\n", nx, elapsed);
  fclose(file);

  free(grid);
  free(grid_new);

  return 0;
}
