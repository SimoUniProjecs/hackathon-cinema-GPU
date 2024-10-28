!  Copyright 2014 NVIDIA Corporation
!  
!  Licensed under the Apache License, Version 2.0 (the "License");
!  you may not use this file except in compliance with the License.
!  You may obtain a copy of the License at
!  
!      http://www.apache.org/licenses/LICENSE-2.0
!  
!  Unless required by applicable law or agreed to in writing, software
!  distributed under the License is distributed on an "AS IS" BASIS,
!  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!  See the License for the specific language governing permissions and
!  limitations under the License.

program mandelbrot_main
use mandelbrot_mod
use openacc
use omp_lib
implicit none
integer      :: num_blocks
integer(1), allocatable   :: image(:,:)
integer      :: iy, ix
integer      :: block, block_size, block_start
integer      :: starty, endy
integer      :: my_gpu, num_gpus, queue
real         :: startt, stopt
character(8) :: arg

num_gpus = acc_get_num_devices(acc_device_nvidia)


!$omp parallel num_threads(num_gpus)
call acc_init(acc_device_nvidia)
call acc_set_device(omp_get_thread_num(),acc_device_nvidia)
!$omp end parallel

num_blocks = 32
block_size = (HEIGHT*WIDTH)/num_blocks
allocate(image(1:HEIGHT, 1:WIDTH))

image = 0
queue = 1

!$omp parallel num_threads(num_gpus) private(my_gpu) firstprivate(queue)
my_gpu = omp_get_thread_num()
call acc_set_device_num(my_gpu,acc_device_nvidia)
print *, "Thread:",my_gpu,"is using GPU",acc_get_device_num(acc_device_nvidia)
startt = omp_get_wtime()
!$acc data create(image(1:HEIGHT, 1:WIDTH))
!$omp do schedule(static,1)
do block=0,(num_blocks-1)
  starty = block  * (WIDTH/num_blocks) + 1
  endy   = min(starty + (WIDTH/num_blocks), WIDTH)
  !$acc parallel loop async(queue)
  do iy=starty,endy
    do ix=1,HEIGHT
      image(ix,iy) = min(max(int(mandelbrot(ix-1,iy-1)),0),MAXCOLORS)
    enddo
  enddo
  !$acc update self(image(:,starty:endy)) async(queue)
enddo
!$omp end do
!$acc wait
!$acc end data
stopt =  omp_get_wtime()
print *,"Time:",(stopt-startt)
!$omp end parallel

call write_pgm(image,'image.pgm')

deallocate(image)
end
