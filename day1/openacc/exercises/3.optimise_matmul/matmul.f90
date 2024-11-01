program matrix_multiply
   use openacc

   implicit none
   integer :: i, j, k, myid, m, n, compiled_for, option, nm
   integer :: t1, t2, dt, count_rate, count_max
   real, allocatable, dimension(:,:) :: a, b, c
   real :: tmp, secs


   call system_clock(count_max=count_max, count_rate=count_rate)
   m = 4
   n = 1000*2**(m-1)    ! 1000, 2000, 4000, 8000
   nm = 1000
   allocate( a(n,nm), b(nm,n), c(n,n) )
   call system_clock(t1)
   ! Initialize matrices

   c(:,:)=0.0

   do j=1,nm
       do i=1,n
          a(i,j) = real(i + j)
          b(i,j) = real(i - j)
       enddo
    enddo

    do j=1,n
       do i=1,n
          tmp = 0.0d0
          do k=1,nm
            tmp = tmp + a(i,k) * b(k,j)
          end do
          c(i,j) = tmp
       enddo
    enddo
    call system_clock(t2)
    dt = t2-t1
    secs = real(dt)/real(count_rate)
    print*, 'wall clock time ', secs

    deallocate(a, b, c)

end program matrix_multiply
