#
# makefile for jacobi solver
#
#

# 
# precision
#
SINGLE=1

#
# ARCH specific flags
#
POWER8=0

ifdef SINGLE
   OPTS=-DSINGLE 
endif


CC=nvcc
CFLAGS=-O3 -g $(OMP) -DSINGLE -arch=sm_80 
OCMPFLAGS=-D__cplusplus__

LDFLAGS= -lm -lrt $(OMP)



#
## SRC directory
DEPS=jacobi.h
SRC=jacobi_optimized utils.c
OBJ=utils.o jacobi.o

.c.o:
	$(CC) -c $(CFLAGS) $<

build:jacobi_cuda

jacobi_cuda: $(OBJ) 
	$(CC) -o $@ $(OBJ) $(LDFLAGS) 


jacobi.o:jacobi_optimized.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

utils.o:utils.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm *.o jacobi_cuda
