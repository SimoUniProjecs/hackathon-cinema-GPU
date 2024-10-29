PGCC  = pgcc -fast -acc -gpu=cc80,cuda12.3 -Minfo=accel
PGF90 = pgfortran -fast -acc -gpu=cc80,cuda12.3 -Minfo=accel

BIN = binary

help:
	@echo "Please specify the make target"
	@echo "make pgi_c or make pgi_f90"

pgi_c: *.c
	$(PGCC) -o $(BIN) $<

pgi_f90: *.f90
	$(PGF90) -o $(BIN) $<

clean:
	$(RM) $(BIN) 
