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

Solution
-------

| Version    | Fortran | 
| -------- | ------- | 
| Single loop  |  0.7526160 s    |
| Optimized | 0.3015850 s |

With optimization clauses, the gridsize along X (and equivalently the number of gangs) has increased significantly, while the blocksize is remained unchanged. The `collapse` clause increses the amount of parallelism exposed for `gang` parallelization, but the inner loop contains in both cases a reduction that has to be executed sequentially within a gang.
