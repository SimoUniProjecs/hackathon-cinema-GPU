Gemm with CUDA libraries from OpenACC
=====================================

This toy code is composed by 3 files

- `mod_hostdata.f90` contains the definition of the global variables
- `mod_functions.f90` contains the initialisation routines
- `gemm.f90` is the main of the program, contains the gemm operation on the CPU and on the GPU

Steps
-----

1. Manage the data movements with `enter data` and `exit data` directives in initialisation/finalisation routines.
2. After computing `ZGEMM` on the CPU, add a call to `cublasZGEMM` on the GPU. Be careful to provide the device buffer (with OpenACC) as an input of the cuBLAS API.
3. Add *nvtx* ranges to wrap the `ZGEMM` operation on the CPU and on GPU
4. Open the `Makefile` and add `cublas` and `nvlamath` to `--cudalib` flag. 
5. Modify the jobscript in order to compile, run and profile the application; submit. 

NSight Systems instruments also cuBLAS API. To this use `--trace=cublas`

Questions
--------

- Open the report and look at the timeline, in corrispondence with the ZGEMM offloaded to the GPU: does the *nvtx* range wrapping the CUBLAS correctly measure the time taken by the operation on the GPU? Why?
- How much speedup did we achieve by offloading the BLAS to GPU, with respect to the CPU version?

Solution
-------

cuBLAS APIs are asynchronous, and the nvtx wrapper is a call on the host. Thus, the duration of the range corresponds to the time taken to enter the API, implement eventually initializations, launch the operation on the GPU and exit the API. To measure the time on the GPU, the **nvtx projection** on the GPU hardware can be inferred from the timeline or the summary. 

:exclamation: In the timeline, we observe that the nvtx range associated to the GPU BLAS takes much more time than the actual operation on the GPU, and contains initialization plus launch operations. Once the operation is launched to the GPU, the cpu exits the range. The projection of the range on the GPU can be retrieven from the timeline in the CUDA hardware panel.

In order to verify that the large amount of time spent in cuBLAS ZGEMM on the host side is due to initializations, you can do a second call and repeat the measurement.

In this second experiment, the time for a gpublas on the host side is hundreds of microseconds instead of two seconds.
