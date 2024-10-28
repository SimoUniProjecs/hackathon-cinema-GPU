# System dependent definitions

FC=	nvfortran
FFLAGS=	
LFLAGS= -L$(CUDA_HOME)/lib64 -lcudart  -lnvhpcwrapnvtx

# System independent definitions

MF=	Makefile

EXE=	binary

SRC= \
        mod_hostdata.f90 \
	mod_functions.f90 \
        gemm.f90

#
# No need to edit below this line
#

.SUFFIXES:
.SUFFIXES: .f90 .o

OBJ=	$(SRC:.f90=.o)

.f90.o:
	$(FC) $(FFLAGS) -c $<

all:	$(EXE)

$(EXE):	$(OBJ)
	$(FC) $(FFLAGS) -o $@ $(OBJ) $(LFLAGS)

$(OBJ):	$(MF)


tar:
	tar cvf cfd.tar $(MF) $(INC) $(SRC)

clean:
	rm -f $(OBJ) $(EXE) *.mod core
