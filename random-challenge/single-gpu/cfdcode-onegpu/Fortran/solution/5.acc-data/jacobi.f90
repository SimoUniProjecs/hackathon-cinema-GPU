module jacobi
  USE nvtx
  implicit none

contains

subroutine jacobistep_acc(psinew, psi, m, n)
  integer :: m, n
  integer :: i, j

  double precision, dimension(0:m+1, 0:n+1) :: psinew, psi
 
   call nvtxStartRange("jacobi_acc",1) 
   !$acc parallel loop collapse(2) present(psi(0:m+1,0:n+1),psinew(0:m+1,0:n+1))
   do j = 1, n
      do i = 1, m
          psinew(i, j) = 0.25d0*(psi(i+1, j) + psi(i-1, j) + &
                             psi(i, j+1) + psi(i, j-1)     )
      end do
  end do
  !$acc end parallel loop
  call nvtxEndRange()

end subroutine jacobistep_acc

double precision function deltasq(new, old, m, n)

  integer :: m, n, i, j
  double precision, dimension(0:m+1, 0:n+1) :: new, old
  double precision :: deltasq

  integer :: ierr
   
   deltasq = 0.d0
    !$acc parallel loop collapse(2) present(new,old) reduction(+:deltasq)
     do j = 1, n
      do i = 1, m
      deltasq = deltasq + (new(i,j)-old(i,j))**2
     end do
    end do
    !$acc end parallel loop

end function deltasq

end module jacobi
                                    


