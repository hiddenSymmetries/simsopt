      SUBROUTINE reinit(ier_flag)
      USE vmec_main
      USE vmec_input, ONLY: bloat, ncurr
      USE vmec_params
      USE vacmod
      USE parallel_include_module
      USE parallel_vmec_module, ONLY: FinalizeRunVmec
      USE parallel_vmec_module, ONLY: FinalizeSurfaceComm
      IMPLICIT NONE
C-----------------------------------------------
C   D u m m y   A r g u m e n t s
C-----------------------------------------------
      INTEGER, INTENT(inout) :: ier_flag
C-----------------------------------------------
C   L o c a l   V a r i a b l e s
C-----------------------------------------------
C      INTEGER :: ireadseq, iosnml
C-----------------------------------------------
      
C      print *, raxis_cc      

      CALL FinalizeSurfaceComm(NS_COMM)
      CALL FinalizeRunVmec(RUNVMEC_COMM_WORLD)

      IF (lrecon .and. itse.le.0 .and. imse.le.0) lrecon = .false.
      IF (lfreeb .and. mgrid_file.eq.'NONE') lfreeb = .false.

      IF (bloat .eq. zero) bloat = one
      IF ((bloat.ne.one) .and. (ncurr.ne.1)) THEN
         ier_flag = 3
         RETURN
      ENDIF
!
!     COMPUTE NTHETA, NZETA VALUES
!
      mpol = ABS(mpol)
      ntor = ABS(ntor)
      IF (mpol .gt. mpold) STOP 'mpol>mpold: lower mpol'
      IF (ntor .gt. ntord) STOP 'ntor>ntord: lower ntor'
      mpol1 = mpol - 1
      ntor1 = ntor + 1
      IF (ntheta .le. 0) ntheta = 2*mpol + 6    !number of theta grid points (>=2*mpol+6)
      ntheta1 = 2*(ntheta/2)
      ntheta2 = 1 + ntheta1/2                   !u = pi
      IF (ntor .eq. 0) lthreed = .false.
      IF (ntor .gt. 0) lthreed = .true.

      IF (ntor.eq.0 .and. nzeta.eq.0) nzeta = 1
      IF (nzeta .le. 0) nzeta = 2*ntor + 4      !number of zeta grid points (=1 IF ntor=0)
      mnmax = ntor1 + mpol1*(1 + 2*ntor)        !SIZE of rmnc,  rmns,  ...
      mnsize = mpol*ntor1                       !SIZE of rmncc, rmnss, ...

      mf = mpol+1
      nf = ntor
      nu = ntheta1
      nv = nzeta
      mf1 = 1+mf
      nf1 = 2*nf+1
      mnpd = mf1*nf1
      nfper = nfp

!
!     INDEXING FOR PACKED-ARRAY STRUCTURE OF XC, GC 
!
      rcc = 1;  zsc = 1
      rss = 0;  rsc = 0;  rcs = 0
      zcc = 0;  zss = 0;  zcs = 0     
      IF (.not.lasym) THEN
         ntheta3 = ntheta2
         mnpd2 = mnpd
         IF (lthreed) THEN
            ntmax = 2
            rss = 2;  zcs = 2
         ELSE
            ntmax = 1
         END IF
      ELSE
         ntheta3 = ntheta1
         mnpd2 = 2*mnpd
         IF (lthreed) THEN
             ntmax = 4
             rss = 2;  rsc = 3;  rcs = 4
             zcs = 2;  zcc = 3;  zss = 4
         ELSE
             ntmax = 2
             rsc = 2;  zcc = 2
         END IF
      END IF

      nuv = nu*nv
      nu2 = nu/2 + 1
      nu3 = ntheta3
      nznt = nzeta*ntheta3
      nuv3 = nznt
!     IF (nuv3 < mnpd) THEN
!        PRINT *, ' nuv3 < mnpd: not enough integration points'
!        STOP 11
!     ENDIF

      IF (ncurr.eq.1 .and. ALL(ac.eq.cbig)) ac = ai            !!Old FORMAT: may not be reading in ac
      WHERE (ac .eq. cbig) ac = zero

      END SUBROUTINE reinit
