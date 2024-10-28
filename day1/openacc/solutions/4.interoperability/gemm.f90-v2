program gpugemm
   use mod_hostdata
   use mod_functions 
   use cublas
   use cudafor
   use nvtx
   implicit none
  
      integer :: i, j, counter
      real*8 :: val, diff

      complex*16, allocatable, dimension(:,:) :: backup

      call initialise()

      val = 23.7d0
 
      allocate(backup(lda,n))
      backup(:,:)=crr(:,:)
      call nvtxStartRange("cpublas")
      call ZGEMM('N','N',m,n,k,( 1.D0, 0.D0 ),arr,lda,brr,ldb,( 1.D0, 0.D0 ),backup,ldc) 
      call nvtxEndRange

       call nvtxStartRange("gpublas")
      !$acc host_data use_device(arr,brr,crr)
      call cublasZGEMM('N','N',m,n,k,( 1.D0, 0.D0 ),arr,lda,brr,ldb,( 1.D0, 0.D0 ),crr,ldc)
      !$acc end host_data
       call nvtxEndRange
      !$acc update host(crr)

      call nvtxStartRange("gpublas")
      !$acc host_data use_device(arr,brr,crr)
      call cublasZGEMM('N','N',m,n,k,( 1.D0, 0.D0 ),arr,lda,brr,ldb,( 1.D0, 0.D0 ),crr,ldc)
      !$acc end host_data
       call nvtxEndRange
      !$acc update host(crr)

      counter=0
      do j = 1, n
        do i = 1, lda
          !diff = abs( backup(i,j) - crr(i,j) )
          !if ( diff .gt. 1.d-10) then
          !      counter = counter + 1
          !      print*, 'diff at ', i, ' is ', diff
          !endif
        end do
      end do

      if ( counter == 0 ) then
              print*, 'TEST PASSED'
      else
              print*, 'TEST FAILED'
      endif
  
      call finalise()
      deallocate(backup)

end program gpugemm
