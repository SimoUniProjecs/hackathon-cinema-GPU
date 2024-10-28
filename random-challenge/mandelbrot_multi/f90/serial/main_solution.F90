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
real         :: startt, stopt
character(8) :: arg

call acc_init(acc_device_nvidia)

num_blocks = 64
block_size = (HEIGHT*WIDTH)/num_blocks
allocate(image(1:HEIGHT, 1:WIDTH))

image = 0

startt = omp_get_wtime()
do block=0,(num_blocks-1)
  starty = block  * (WIDTH/num_blocks) + 1
  endy   = min(starty + (WIDTH/num_blocks), WIDTH)
  do iy=starty,endy
    do ix=1,HEIGHT
      image(ix,iy) = min(max(int(mandelbrot(ix-1,iy-1)),0),MAXCOLORS)
    enddo
  enddo
enddo
stopt =  omp_get_wtime()

print *,"Time:",(stopt-startt)

call write_pgm(image,'image.pgm')

deallocate(image)
end
