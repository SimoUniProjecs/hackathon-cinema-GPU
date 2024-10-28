#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include "jacobi.h"

static int frame=0;
static char filename[10];

void set_filename(void)
{
   sprintf(filename,"frame.%d",frame);
}

int get_num_frames(void)
{
  return frame;
}

int get_frame(void)
{
  return frame;
}

void usage(char **argv)
{
   printf("Usage: %s -x GRIDX -y GRIDY -l -r -t -b for wall values (optional)\n",argv[0]);
}

void get_options(const int argc, char **argv, int *nx, int *ny, int *ndump, double *left, double *right, double *top, double *bottom)
{
  int c;
  while ((c = getopt(argc, argv, "x:y:l:r:t:b:n:")) != -1)
    switch (c)
    {
    // x dimensions of box
    case 'x':
      *nx = atoi(optarg);
      break;
    // y dimensions of box
    case 'y':
      *ny = atoi(optarg);
      break;
    
    // value for left boundary
    case 'l':
      *left = atof(optarg);
      break;
    
    // value for right boundary
    case 'r':
      *right = atof(optarg);
      break;

    // value for top boundary  
    case 't':
      *top = atof(optarg);
      break;

    // value for bottom boundary  
    case 'b':
      *bottom = atof(optarg);
      break;

    // dump frame interval. If =0, no frames dumped  
    case 'n':
      *ndump=atoi(optarg);
      break;  
    default:     
      abort();
    }

}

void set_borders(double *grid, int nx, int ny, double left, double right, double top, double bottom)
{
  int i, j;
  for (i = 0; i < nx + 2; i++)
  {
    grid[i] = top;
    j = (ny + 1) * (nx + 2) + i;
    grid[j] = bottom;
  }
  for (i = 1; i < ny + 1; i++)
  {
    j = (nx + 2) * i;
    grid[j] = left;
    grid[j + nx + 1] = right;
  }
}

void init_grids(double *grid, double *grid_new, int nx, int ny, double left, double right, double top, double bottom)
{

  int i, j, k;
  set_borders(grid, nx, ny, left, right, top, bottom);
  set_borders(grid_new, nx, ny, left, right, top, bottom);

  // Initialise rest of grids
  for (i = 1; i <= ny; i++)
  {
    for (j = 1; j <= nx; j++)
    {
      k = (nx + 2) * i + j;
      grid_new[k] = grid[k] = 0.0;
    }
  }
}

void dump_grid(double *grid, int nx, int ny)
{
  int i, j, k;
  set_filename();
  FILE *fp = fopen(filename, "w");
  fprintf(fp, "%d %d \n", nx, ny);
  for (i = 0; i < ny + 2; ++i)
  {  
    for (j = 0; j < nx + 2; ++j)
    {
      k = (nx + 2) * i + j;
      fprintf(fp, "%12.6f", grid[k]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  frame++;
}

void read_grid_size(int *nx, int *ny)
{
  set_filename();
  FILE *fp = fopen(filename, "r");
  int err = fscanf(fp, "%d %d\n", nx, ny);
  fclose(fp);
}

void read_grid(double *grid, int nx, int ny)
{
  int i, j, k, err;
  set_filename();
  FILE *fp = fopen(filename, "r");
  err = fscanf(fp, "%d %d", &nx, &ny);
  for (i = 0; i < nx + 2; i++)
  {
    err = fscanf(fp, "\n");
    for (j = 0; j < ny + 2; j++)
    {
      k = (ny + 2) * i + j;
      err = fscanf(fp, "%lf", &grid[k]);
    }
  }
  fclose(fp);
}
