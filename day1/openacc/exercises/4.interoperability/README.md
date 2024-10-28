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
