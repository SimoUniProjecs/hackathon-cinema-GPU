program daxpy
  
  implicit none
  real(kind=8) :: start, finish

  integer :: N=1e12
  real(kind=8), allocatable :: D(:), X(:), Y(:)
  integer :: i
  integer :: A = 16
  
  ! Allocate memory
  allocate( D(N), X(N), Y(N))

  ! Initilise data
  
  ! Start timer
  call cpu_time(start)

  !$acc data create(X,Y) copyout(D)
  !$acc parallel loop
  do i = 1, N
    D(i) = 0
    X(i) = 1
    Y(i) = 2
  end do

  !$acc parallel loop 
  do i = 1, N 
     D(i) = A* X(i) + Y(i)
  end do
  !$acc end data

  call cpu_time(finish)

  ! Print result
  do i = 1, 10
     Print *, D(i)
  end do
  
  write(*,"(A)")         "------------------------------------"
  write(*,"(A,F10.3)")   "runtime:  ", finish-start
  write(*,"(A)")         "------------------------------------"

  deallocate(D,X,Y)
end program daxpy


