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

void init_grids(double*, double*,int , int, double, double, double, double);
void dump_grid(double*,int,int);
void read_grid(double *grid,int nx, int ny);
void read_grid_size(int *nx, int *ny);
void set_borders(double *,int,int, double, double, double, double);
void get_options(int, char **, int*,int*,int*,double*,double*,double*,double*);
void usage(char**);
int get_num_frames();


#ifdef __cplusplus
}
#endif


