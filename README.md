# Simo & David
## DAY 1: 29 Oct - Jacobe optimization Serial
David and I have started working on the analysis of the parallelization of the Jacobi algorithm. 
Our goal is to understand how to optimize this iterative method to solve systems of linear equations, exploiting the potential of parallel computing.

1) We have analyzed the problem.
2) Initialized the memory on the Host and Device
3) Created our first Kernel.

We are a bit confused having as never used CUDA and having as a background only worked with C with pthread libraries in the Operating Systems course

## DAY 2: 30 Oct - Jacobe optimization Parallel
### What we Achived:
- Finished the Jacobe Parallelization Problem
- Benchmark test with Leonardo
- An x11 Speedup with the GPU
![Situazione globale con passo 10](https://hackmd.io/_uploads/H1u3i31bJg.png)
![Situazione più precisa con passo 1](https://hackmd.io/_uploads/ryEeC3yb1x.png)

## Our Conclusion:
We have noticed and understood that it's possible that the CPU can be faster than the GPU (only for small cases)

### Profilation of the Jacobe Parallel Code
Even though Jacobe it's an easy problem we decided to try to do it with NVIDIA Nsight System
![NVIDIA TOOL](https://hackmd.io/_uploads/HJzCXCkZ1g.png)

### Little Ottimization
Using 2 streams instead of 1 we noticed an improvement, here the details:
![IMAGE 2024-10-31 08:32:51](https://hackmd.io/_uploads/B1rnG2eb1l.jpg)
![Risultato](https://hackmd.io/_uploads/rJzt1heW1l.png)
![IMAGE 2024-10-31 08:32:58](https://hackmd.io/_uploads/SkG2MneZ1g.jpg)

## Here the differences between the two codes:
![Figure_1](https://hackmd.io/_uploads/SJXQ9hg-yl.png)

## DAY 3: 31 Oct
Goal to Achieve:
- Plot the difference between **BEFORE & AFTER** the optimization ( did this morning in the car )
- Understanding the PROFILING process
- finishing the PROFILING process of the Jacobe
- Trying another Exercise