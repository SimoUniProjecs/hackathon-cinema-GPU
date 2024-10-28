#
# makefile for jacobi solver
#
#

# 
# precision
#
SINGLE=1

ifdef SINGLE
   #OPTS=-DSINGLE -Wdouble-promotion -fsingle-precision-constant
   OPTS=-DSINGLE -fsingle-precision-constant
endif

CC=gcc
CFLAGS=-O2 -g -fopenmp 
LDFLAGS= -lm -lrt -fopenmp


#
## SRC directory
DEPS=jacobi.h 
SRC=jacobi.c utils.c
OBJ=jacobi.o utils.o

ITER_SRC=c-iterator.c
ITER_OBJ=c-iterator.o utils.o
INIT_OBJ=grid-init.o utils.o

# INCLUDEs
INCS=$(COMMON_DIR)

build:jacobi


jacobi: $(OBJ) 
	$(CC) -o $@ $(OBJ) $(LDFLAGS)	 

c-iterator: $(ITER_OBJ)
	$(CC) -o $@ $(ITER_OBJ) $(LDFLAGS) 

grid-init: $(INIT_OBJ)
	$(CC) -o $@ $(INIT_OBJ) $(LDFLAGS) 


%.o:%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)


clean:
	-rm *.o jacobi 
