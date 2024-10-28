module mod_functions
  implicit none

  save
  contains

  subroutine initialise()
    use mod_hostdata
    implicit none

    integer :: i

    m=6000
    n=6000
    k=6000

    lda=m
    ldb=k
    ldc=m

    allocate(arr(lda,k),brr(ldb,n),crr(lda,n))
    allocate(tmpr(lda,max(k,n)),tmpi(lda,max(k,n)))

    call random_number(tmpi(1:lda,1:k))
    call random_number(tmpr(1:lda,1:k))
    arr(:,:)=tmpr(1:lda,1:k)+i*tmpi(1:lda,1:k)

    call random_number(tmpi(1:ldb,1:n))
    call random_number(tmpr(1:ldb,1:n))
    brr(:,:)=tmpr(1:ldb,1:n)+i*tmpi(1:ldb,1:n)

    call random_number(tmpi(1:lda,1:n))
    call random_number(tmpr(1:lda,1:n))
    crr(:,:)=tmpr(1:lda,1:n)+i*tmpi(1:lda,1:n)

  end subroutine initialise

  subroutine finalise()
    use mod_hostdata
    implicit none

    deallocate(arr,brr,crr)

  end subroutine

END MODULE mod_functions

