!-----------------------------------------------------------------------
!     Subroutine:    reinit
!     Authors:       S. Lazerson, Caoxiang Zhu
!     Date:          05/26/2012; 06/14/2020
!     Description:   This subroutine handles reseting the VMEC internal
!                    variables so that the input file does not need to
!                    be read again. Mimics readin.f (VMEC/PARVMEC)
!-----------------------------------------------------------------------
      SUBROUTINE reinit
!-----------------------------------------------------------------------
!     Libraries
!-----------------------------------------------------------------------
      USE vmec_main
      USE vmec_params
      USE vacmod
!      USE vsvd
      USE vspline
      USE timer_sub
      USE mgrid_mod, ONLY: nextcur, curlabel, nfper0, read_mgrid, free_mgrid,&
                           mgrid_path_old
      USE init_geometry
      USE mpi_params, ONLY: MPI_COMM_MYWORLD     
      USE parallel_vmec_module, ONLY: FinalizeRunVmec
      USE parallel_vmec_module, ONLY: FinalizeSurfaceComm

      IMPLICIT NONE
!-----------------------------------------------------------------------
!     Local Variables
!        ier         Error flag
!        iunit       File unit number
!----------------------------------------------------------------------
      INTEGER :: iexit, ipoint, n, iunit, ier_flag_init,i, ni, m, nsmin, igrid, mj, isgn, ioff, joff, iflag
      REAL(rprec), DIMENSION(:,:), POINTER :: rbcc, rbss, rbcs, rbsc, zbcs, zbsc, zbcc, zbss
      REAL(rprec) :: rtest, ztest, tzc, trc, delta
      REAL(rprec), ALLOCATABLE :: temp(:)
      CHARACTER(LEN=100) :: line, line2
      CHARACTER(LEN=1)   :: ch1, ch2
      
!----------------------------------------------------------------------
!     BEGIN SUBROUTINE
!----------------------------------------------------------------------
      ier_flag_init = 0
!     Adjust vaccum grid file
      IF (lfreeb) THEN
         CALL free_mgrid(iflag)
         mgrid_path_old = " "
         CALL read_mgrid(mgrid_file,extcur,nzeta,nfp,.false.,iflag,MPI_COMM_MYWORLD)
      END IF
!     PARSE NS_ARRAY
      nsin = MAX (3, nsin)
      multi_ns_grid = 1
      IF (ns_array(1) .eq. 0) THEN                    !!Old input style
          ns_array(1) = MIN(nsin,nsd)
          multi_ns_grid = 2
          ns_array(multi_ns_grid) = ns_default        !!Run on 31-point mesh
      ELSE
          nsmin = 1
          DO WHILE (ns_array(multi_ns_grid) .gt. nsmin .and. multi_ns_grid .lt. 100)
             nsmin = MAX(nsmin, ns_array(multi_ns_grid))
             IF (nsmin .le. nsd) THEN
                multi_ns_grid = multi_ns_grid + 1
             ELSE                                      !!Optimizer, Boozer code overflows otherwise
                ns_array(multi_ns_grid) = nsd
                nsmin = nsd
             END IF
          END DO
          multi_ns_grid = multi_ns_grid - 1
      END IF
      IF (ftol_array(1) .eq. zero) THEN
         ftol_array(1) = 1.e-8_dp
         IF (multi_ns_grid .eq. 1) ftol_array(1) = ftol
         DO igrid = 2, multi_ns_grid
            ftol_array(igrid) = 1.e-8_dp * (1.e8_dp * ftol)**( REAL(igrid-1,rprec)/(multi_ns_grid-1) )
         END DO
      END IF
      ns_maxval = nsmin
!     Handle nvacskip
      IF (nvacskip.le.0) nvacskip = nfp
!     Handle bloat
      IF (bloat .ne. 1.0) phiedge = phiedge * bloat
!     CONVERT TO REPRESENTATION WITH RBS(m=1) = ZBC(m=1)
      IF (lasym) THEN
         delta = ATAN( (rbs(0,1) - zbc(0,1))/(ABS(rbc(0,1)) + ABS(zbs(0,1))) )
         IF (delta .ne. 0.0) THEN
           DO m = 0,mpol
             DO n = -ntor,ntor
               trc = rbc(n,m)*COS(m*delta) + rbs(n,m)*SIN(m*delta)
               rbs(n,m) = rbs(n,m)*COS(m*delta) - rbc(n,m)*SIN(m*delta)
               rbc(n,m) = trc
               tzc = zbc(n,m)*COS(m*delta) + zbs(n,m)*SIN(m*delta)
               zbs(n,m) = zbs(n,m)*COS(m*delta) - zbc(n,m)*SIN(m*delta)
               zbc(n,m) = tzc
             ENDDO
           ENDDO
         END IF
      END IF
!     CONVERT TO INTERNAL REPRESENTATION OF MODES
!
!     R = RBCC*COS(M*U)*COS(N*V) + RBSS*SIN(M*U)*SIN(N*V)
!         + RBCS*COS(M*U)*SIN(N*V) + RBSC*SIN(M*U)*COS(N*V)
!     Z = ZBCS*COS(M*U)*SIN(N*V) + ZBSC*SIN(M*U)*COS(N*V)
!         + ZBCC*COS(M*U)*COS(N*V) + ZBSS*SIN(M*U)*SIN(N*V)
!
!
!     POINTER ASSIGNMENTS (NOTE: INDICES START AT 1, NOT 0, FOR POINTERS, EVEN THOUGH
!                          THEY START AT ZERO FOR RMN_BDY)
!     ARRAY STACKING ORDER DETERMINED HERE
      rbcc => rmn_bdy(:,:,rcc)
      zbsc => zmn_bdy(:,:,zsc)
      IF (lthreed) THEN
         rbss => rmn_bdy(:,:,rss)
         zbcs => zmn_bdy(:,:,zcs)
      END IF

      IF (lasym) THEN
         rbsc => rmn_bdy(:,:,rsc)
         zbcc => zmn_bdy(:,:,zcc)
         IF (lthreed) THEN
            rbcs => rmn_bdy(:,:,rcs)
            zbss => zmn_bdy(:,:,zss)
         END IF
      END IF

      rmn_bdy = 0;  zmn_bdy = 0

      ioff = LBOUND(rbcc,1)
      joff = LBOUND(rbcc,2)

      DO m=0,mpol1
         mj = m+joff
         IF (lfreeb .and.(mfilter_fbdy.gt.1 .and. m.gt.mfilter_fbdy)) CYCLE
         DO n=-ntor,ntor
            IF (lfreeb .and.(nfilter_fbdy.gt.0 .and. ABS(n).gt.nfilter_fbdy)) CYCLE
            ni = ABS(n) + ioff
            IF (n .eq. 0) THEN
               isgn = 0
            ELSE IF (n .gt. 0) THEN
               isgn = 1
            ELSE
               isgn = -1
            END IF
            rbcc(ni,mj) = rbcc(ni,mj) + rbc(n,m)
            IF (m .gt. 0) zbsc(ni,mj) = zbsc(ni,mj) + zbs(n,m)

            IF (lthreed) THEN
               IF (m .gt. 0) rbss(ni,mj) = rbss(ni,mj) + isgn*rbc(n,m)
               zbcs(ni,mj) = zbcs(ni,mj) - isgn*zbs(n,m)
            END IF

            IF (lasym) THEN
               IF (m .gt. 0) rbsc(ni,mj) = rbsc(ni,mj) + rbs(n,m)
               zbcc(ni,mj) = zbcc(ni,mj) + zbc(n,m)
               IF (lthreed) THEN
                  rbcs(ni,mj) = rbcs(ni,mj) - isgn*rbs(n,m)
                  IF (m .gt. 0) zbss(ni,mj) = zbss(ni,mj) + isgn*zbc(n,m)
               END IF
            END IF

            IF (ier_flag_init .ne. norm_term_flag) CYCLE
            trc = ABS(rbc(n,m)) + ABS(rbs(n,m)) + ABS(zbc(n,m)) + ABS(zbs(n,m))
            IF (m .eq. 0) THEN
               IF (n .lt. 0) CYCLE
               IF (trc.eq.zero .and. ABS(raxis_cc(n)).eq.zero .and. ABS(zaxis_cs(n)).eq.zero) CYCLE
            ELSE
               IF (trc .eq. zero) CYCLE
            END IF
         END DO
      END DO

!
!     CHECK SIGN OF JACOBIAN (SHOULD BE SAME AS SIGNGS)
!
      m = 1
      mj = m+joff
      rtest = SUM(rbcc(1:ntor1,mj))
      ztest = SUM(zbsc(1:ntor1,mj))
      lflip=(rtest*ztest .lt. zero)
      signgs = -1
      IF (lflip) CALL flip_theta(rmn_bdy, zmn_bdy)


!
!     CONVERT TO INTERNAL FORM FOR (CONSTRAINED) m=1 MODES
!     INTERNALLY, FOR m=1: XC(rss) = .5(RSS+ZCS), XC(zcs) = .5(RSS-ZCS)
!     WITH XC(zcs) -> 0 FOR POLAR CONSTRAINT 
!     (see convert_sym, convert_asym in totzsp_mod file)
!

      IF (lconm1 .and. (lthreed .or. lasym)) THEN
         ALLOCATE (temp(SIZE(rbss,1)))
         IF (lthreed) THEN
            mj = 1+joff
            temp = rbss(:,mj)
            rbss(:,mj) = p5*(temp(:) + zbcs(:,mj))
            zbcs(:,mj) = p5*(temp(:) - zbcs(:,mj))
         END IF
         IF (lasym) THEN
            mj = 1+joff
            temp = rbsc(:,mj)
            rbsc(:,mj) = p5*(temp(:) + zbcc(:,mj))
            zbcc(:,mj) = p5*(temp(:) - zbcc(:,mj))
         END IF
         IF (ALLOCATED(temp)) DEALLOCATE (temp)
      END IF
!     Handle curtor
      currv = mu0*curtor              !Convert to Internal units

      ! finalize comm
      CALL FinalizeSurfaceComm(NS_COMM)
      CALL FinalizeRunVmec(RUNVMEC_COMM_WORLD)

      RETURN
!----------------------------------------------------------------------
!     END SUBROUTINE
!----------------------------------------------------------------------
      END SUBROUTINE reinit
