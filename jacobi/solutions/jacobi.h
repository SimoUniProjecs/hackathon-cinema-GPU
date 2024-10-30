#ifdef __cplusplus
extern "C" {
#endif
// Grid boundary conditions
#define RIGHT 1.0
#define LEFT 1.0
#define TOP 1.0
#define BOTTOM 10.0

// precision

#define TOLERANCE 0.0001

// Algorithm settings
#define NPRINT 1000
#define MAX_ITER 100000
#define NDUMP 0

void init_grids(float*, float*,int , int, float, float, float, float);
void dump_grid(float*,int,int);
void read_grid(float *grid,int nx, int ny);
void read_grid_size(int *nx, int *ny);
void set_borders(float *,int,int, float, float, float, float);
void get_options(int, char **, int*,int*,int*,float*,float*,float*,float*);
void usage(char**);
int get_num_frames();


#ifdef __cplusplus
}
#endif


