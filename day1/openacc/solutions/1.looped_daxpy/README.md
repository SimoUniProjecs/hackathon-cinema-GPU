DAXPY offload
=============

Steps
----

1. Offload the do/for loops with parallel directives.
2. Open the Makefile and add instructions to targer Leonardo's accelerators.
3. Modify the jobscript is order to compile and run the C or Fortran version of the code. 
4. Modify the Makefile in order to target multicore; run the code.

Questions
--------

- Compare the time to solution GPU code vs serial code. Do you observe a speed up?
- Compare the time to solution multicore code vs serial code. Do you observe a speedup? 
- Can you guess why we achieve performance improvement only in one of the two cases?

Solution
-------

| Version    | Fortran | C  |
| -------- | ------- | ------ |
| Serial  | 14.15728 s    | 4.463225 s|
| GPU | 29.018  s   | 7.407395 s |
| Multicore    |  3.579 s   | 0.849510 s |

The multicore code is faster because memory is shared; in the GPU case, data need to be copied before and after the offloaded loops. The amount of time spend in data movements vs kernel computation can be checked with the NSight Systems. 

Modify the jobscript in order to trace OpenACC runtime.Once the report is generated, postprocess the \*.nsys-rep file with the following commands: 

`module load nvhpc/24.3; nsys stats -r cuda_gpu_sum report1.nsys-rep`

|Time (%)|Total Time (s)|Instances|Category|Operation|
|---|---|---|---|---|
|88.8|27.11527|4|MEMORY\_OPER|[CUDA memcpy Device-to-Host]|
|11.0|3.35921|2|MEMORY\_OPER|[CUDA memcpy Host-to-Device]|
|0.1|0.03583|1|CUDA\_KERNEL|daxpy\_29\_gpu|
|0.1|0.03461|1|CUDA\_KERNEL|daxpy\_20\_gpu|

