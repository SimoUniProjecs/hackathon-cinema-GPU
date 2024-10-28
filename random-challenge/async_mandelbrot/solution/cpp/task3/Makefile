CXX=nvc++
CXXFLAGS=-fast -acc=gpu -gpu=pinned -Minfo=all

mandelbrot.x: mandelbrot.o main_solution.o
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f *.o core *.lst *.ptx *.pgprof *.cubin *.s *.x *.mod *.nvprof
