Optimise matmul
=============

Steps
----

1. Add data and `parallel loop/seq` directives, without optimization clauses. 
2. Open the Makefile and add instructions to target Leonardo's accelerators; use `-acc=noautopar` to inhibit automatic loop optimizations done by the compiler and `-Minfo=accel` to get information on how the code is compiled for GPUs.
3. Modify the jobscript is order to compile and run the code on compute nodes.
4. Modify the parallel directives by adding clauses for loop optimizations; rerun the code.
5. Try also with `kernels` directive and `-acc=autopar`.

Questions
--------

- How does the compiler offloads the loop in the different cases?
- Compare the time to solution GPU code in the three cases. Do you observe performance improvement?

