### Mandelbrot exercise 

Starting from the serial version, accomplish the following tasks

1. Offload the kernel ( careful with mandelbrot routine! ) 

.. code-block:: console
   
   do iy=starty,endy
     do ix=1,HEIGHT
       image(ix,iy) = min(max(int(mandelbrot(ix-1,iy-1)),0),MAXCOLORS)
     enddo
   enddo

  Mange explicitly data movements (update at the end of the double loop)

2. Implement asynchronous operations (use async clause in parallel loop and update)

   ! Set pinned memory to observe performance improvement ( -gpu=pinned )
