Single-node multi-GPU mandelbrot
================================

This code is taken from the repository of [OpenACC best practice guide](https://github.com/OpenACC/openacc-best-practices-guide/tree/main/examples/mandelbrot) from NVIDIA. 

In this code you will start from the block version of the mandelbrot exercise, and use openmp threads to send each block to a different GPU in the node. To this, you need to bind to each thread one of the available GPUs in a round-robin fashion.

Thread-GPU binding
------------------

As a first step, we need to use the OpenMP and OpenACC/CUDA APIs to query the number of openmp threads available and bind threads to gpus.
Consider that the number of threads is equivalent to the number of gpus on the node, unkown a priori. Use the following APIs:

- `acc_set_device`
- `acc_get_num_devices`
- `omp_get_thread_num`

from the OpenACC and OpenMP libraries. Do not forget to include these libraries in the header of the program.

To check if everything works, print 

Multi-gpu offload
-----------------

Distribute the first do loop among openmp threads and parallelize the inner loops. Use also asynchronous and wait clause to send each block processing and value update to queue 1 of its GPU; check the behaviour on the timeline view. Is the `async` directive needed?


