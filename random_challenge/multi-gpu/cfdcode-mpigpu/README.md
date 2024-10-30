CFD multi-gpu with MPI
=============

The non-gpu version of this code is taken from the repository of [David Henty](https://github.com/davidhenty/cfd/tree/master).

This is the MPI-distributed version of the CFD previously offloded to a single GPU. In this case, each MPI rank will be binded to a GPU for the offload. The aim is to implement efficient GPU to GPU communications.

The code is composed by four files 

- `cfd.f90` contains the main loop of the program
- `jacobi.f90` contains the jacobi step the reduction for error calculation
- `boundary.f90` contains initialization routine and the haloswap
- `cfdio.f90` contains routines for IO.

Non-GPU run
-----------

Instrument the `haloswap` and the main loop with nvtx ranges. Run the non GPU case e trace the application with `nsys`; how much time is spent in communications (haloswap) within the loop?

MPI GPU-binding
---------------

Implement MPI-GPU binding in a round robin fashion, by using the following routines

- `acc_get_device_type`
- `acc_get_num_devices`
- `acc_set_device_num`
- `acc_get_device_num`

Check MPI-GPU binding the following message

`write(*,*) "MPI ", rank, " is using GPU: ", acc_get_device_num(acc_get_device_type())`

Identify which routine contains local compuation to offload, and which routine require communications among them. 

MPI non aware implementation
----------------------------

As a first step, add data clauses to update buffers between CPU and GPU before and after the communications; move only the portion of the data needed (you can do array slicing). Modify the jobscript in order to trace also openacc and cuda, submit. Open the report with `nsys-ui`.

- How much time does communication take?
- Which is the ratio among MPI communication and computation inside the main loop?
- Can you identify a potential performance improvement?

MPI aware implementation
------------------------

Use the `host_data` directive to pass the buffer to the MPI APIs where needed. Repeat the run and check for the different time spent in MPI communications, the amount of data moved and the kind of memory operations.

