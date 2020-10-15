      MODULE read_wout_mod
!
!     USE READ_WOUT_MOD to include variables dynamically allocated
!     in this module
!     Call DEALLOCATE_READ_WOUT to free this memory when it is no longer needed
! 
!     Reads in output from VMEC equilibrium code(s), contained in wout file
!
!     Contained subroutines:
!
!     read_wout_file      wrapper alias called to read/open wout file      
!     read_wout_text      called by read_wout_file to read text file wout
!     read_wout_nc        called by read_wout_file to read netcdf file wout
!
!     Post-processing routines
!
!     mse_pitch           user-callable function to compute mse pitch angle
!                         for the computed equilibrium
!

      USE v3_utilities
      USE vmec_input, ONLY: lrfp, lmove_axis, nbfld
      USE mgrid_mod

      IMPLICIT NONE
!#if defined(NETCDF)
!------------------------------------------------
!   L O C A L   P A R A M E T E R S
!------------------------------------------------
! Variable names (vn_...) : put eventually into library, used by read_wout too...
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------
      INTEGER :: nfp, ns, mpol, ntor, mnmax, mnmax_nyq, itfsq, niter
      INTEGER :: iasym, ireconstruct, ierr_vmec, imse, itse, nstore_seq
      INTEGER :: isnodes, ipnodes, imatch_phiedge, isigng, mnyq, nnyq, ntmax
      INTEGER :: mnmaxpot, vmec_type
      REAL(rprec) :: wb, wp, gamma, pfac, rmax_surf, rmin_surf
      REAL(rprec) :: zmax_surf, aspect, betatot, betapol, betator, betaxis, b0
      REAL(rprec) :: tswgt, msewgt, flmwgt, bcwgt, phidiam, version_
      REAL(rprec) :: delphid, IonLarmor, VolAvgB
      REAL(rprec) :: fsql, fsqr, fsqz, ftolv
      REAL(rprec) :: Aminor, Rmajor, Volume, RBtor, RBtor0, Itor, machsq
      REAL(rprec), ALLOCATABLE :: rzl_local(:,:,:,:)
      REAL(rprec), DIMENSION(:,:), ALLOCATABLE :: rmnc, zmns, lmns, rmns, zmnc, lmnc, bmnc, gmnc, bsubumnc
      REAL(rprec), DIMENSION(:,:), ALLOCATABLE :: bsubvmnc, bsubsmns, bsupumnc, bsupvmnc, currvmnc, currumnc, bbc, raxis, zaxis 
      REAL(rprec), DIMENSION(:,:), ALLOCATABLE :: bmns, gmns, bsubumns, bsubvmns, bsubsmnc, bsupumns, bsupvmns, currumns, currvmns
      REAL(rprec), DIMENSION(:,:), ALLOCATABLE :: pparmnc, ppermnc, hotdmnc, pbprmnc, ppprmnc, sigmnc, taumnc
      REAL(rprec), DIMENSION(:,:), ALLOCATABLE :: pparmns, ppermns, hotdmns, pbprmns, ppprmns, sigmns, taumns
      REAL(rprec), DIMENSION(:,:), ALLOCATABLE :: protmnc, protrsqmnc, prprmnc, protmns, protrsqmns, prprmns 
      REAL(rprec), DIMENSION(:), ALLOCATABLE :: bsubumnc_sur, bsubumns_sur, bsubvmnc_sur, bsubvmns_sur, bsupumnc_sur, bsupumns_sur, bsupvmnc_sur, bsupvmns_sur
      REAL(rprec), DIMENSION(:), ALLOCATABLE :: iotas, iotaf, presf, phipf, mass, pres, beta_vol, xm, xn, potsin, potcos, xmpot, xnpot, qfact, chipf, phi, chi
      REAL(rprec), DIMENSION(:), ALLOCATABLE :: xm_nyq, xn_nyq, phip, buco, bvco, vp, overr, jcuru, jcurv, specw, jdotb, bdotb, bdotgradv, fsqt, wdot, am, ac, ai
      REAL(rprec), DIMENSION(:), ALLOCATABLE :: am_aux_s, am_aux_f, ac_aux_s, ac_aux_f, ai_aux_s, ai_aux_f, Dmerc, Dshear, Dwell, Dcurr, Dgeod, equif, extcur
      REAL(rprec), DIMENSION(:), ALLOCATABLE :: sknots, ystark, y2stark, pknots, ythom, y2thom, anglemse, rmid, qmid, shear, presmid, alfa, curmid, rstark
      REAL(rprec), DIMENSION(:), ALLOCATABLE :: qmeas, datastark, rthom, datathom, dsiobt      
      REAL(rprec), DIMENSION(:), ALLOCATABLE :: pmap, omega, tpotb        ! SAL -FLOW
      LOGICAL :: lasym, lthreed, lwout_opened=.false.
      CHARACTER(LEN=200) :: mgrid_file
      CHARACTER(LEN=100) :: input_extension
      CHARACTER(LEN=20)  :: pmass_type, pcurr_type, piota_type

!     INTEGER, PARAMETER :: norm_term_flag=0, bad_jacobian_flag=1, more_iter_flag=2, jac75_flag=4

!     OVERLOAD SUBROUTINE READ_WOUT_FILE TO ACCEPT BOTH UNIT NO. (OPENED EXTERNALLY)
!     OR FILENAME (HANDLE OPEN/CLOSE HERE)
      INTERFACE read_wout_file
          MODULE PROCEDURE readw_and_open, readw_only
      END INTERFACE

      PRIVATE :: read_wout_text, read_wout_nc, write_wout_text, write_wout_nc
!     PRIVATE :: norm_term_flag, bad_jacobian_flag, more_iter_flag, jac75_flag

      CONTAINS

      SUBROUTINE readw_and_open(file_or_extension, ierr, iopen)
      USE safe_open_mod
      IMPLICIT NONE
!------------------------------------------------
!   D u m m y   A r g u m e n t s
!------------------------------------------------
      INTEGER, INTENT(out) :: ierr
      INTEGER, OPTIONAL :: iopen
      CHARACTER(LEN=*), INTENT(in) :: file_or_extension
!------------------------------------------------
!   L o c a l   V a r i a b l e s
!------------------------------------------------
      INTEGER, PARAMETER :: iunit_init = 10
      INTEGER :: iunit
      LOGICAL :: isnc
      CHARACTER(len=LEN_TRIM(file_or_extension)+10) :: filename
!------------------------------------------------
!
!     THIS SUBROUTINE READS THE WOUT FILE CREATED BY THE VMEC CODE
!     AND STORES THE DATA IN THE READ_WOUT MODULE
!
!     FIRST, CHECK IF THIS IS A FULLY-QUALIFIED PATH NAME
!     MAKE SURE wout IS NOT EMBEDDED IN THE NAME (PERVERSE USER...)
!
      filename = 'wout'
      CALL parse_extension(filename, file_or_extension, isnc)
      CALL flush(6)
!SPH  IF (.not.isnc) STOP 'ISNC ERR IN READ_WOUT_MOD'
      IF (isnc) THEN
!#if defined(NETCDF)
         CALL read_wout_nc(filename, ierr)
!#else
         PRINT *, "NETCDF wout file can not be opened on this platform"
         ierr = -100
!#endif
      ELSE
         iunit = iunit_init
         CALL safe_open (iunit, ierr, filename, 'old', 'formatted')
         IF (ierr .eq. 0) CALL read_wout_text(iunit, ierr)
         CLOSE(unit=iunit)
      END IF
      
      IF (PRESENT(iopen)) iopen = ierr
      lwout_opened = (ierr .eq. 0)
      ! WHEN READING A NETCDF FILE, A BAD RUN MAY PREVENT XN FROM BEING
      ! READ, SUBSEQUENTLY WE MUST CHECK TO SEE IF XN HAS BEEN ALLOCATED
      ! BEFORE DOING ANYTHING WITH IT OTHERWISE WE DEFAULT LTHREED TO
      ! FALSE.  - SAL 09/07/11
      IF (ALLOCATED(XN)) THEN
         lthreed = ANY(NINT(xn) .ne. 0)
      ELSE
         lthreed = .FALSE.
      END IF

      END SUBROUTINE readw_and_open

      SUBROUTINE write_wout_file(file_or_extension, ierr)
      USE safe_open_mod
      IMPLICIT NONE
!------------------------------------------------
!   D u m m y   A r g u m e n t s
!------------------------------------------------
      INTEGER, INTENT(out) :: ierr
      CHARACTER(LEN=*), INTENT(in) :: file_or_extension
!------------------------------------------------
!   L o c a l   V a r i a b l e s
!------------------------------------------------
      INTEGER, PARAMETER :: iunit_init = 10
      INTEGER :: iunit
      LOGICAL :: isnc
      CHARACTER(len=LEN_TRIM(file_or_extension)+10) :: filename
!------------------------------------------------
!
!     THIS SUBROUTINE WRITES THE WOUT FILE FROM 
!     THE DATA IN THE READ_WOUT MODULE
!
!     FIRST, CHECK IF THIS IS A FULLY-QUALIFIED PATH NAME
!     MAKE SURE wout IS NOT EMBEDDED IN THE NAME (PERVERSE USER...)
!
      filename = 'wout'
      CALL parse_extension(filename, file_or_extension, isnc)
      IF (isnc) THEN
!#if defined(NETCDF)
         CALL write_wout_nc(filename, ierr)
!#else
         PRINT *, "NETCDF wout file can not be opened on this platform"
         ierr = -100
!#endif
      ELSE
         IF (ierr .eq. 0) CALL write_wout_text(filename, ierr)
      END IF

      END SUBROUTINE write_wout_file


      SUBROUTINE readw_only(iunit, ierr, iopen)
      IMPLICIT NONE
!------------------------------------------------
!   D u m m y   A r g u m e n t s
!------------------------------------------------
      INTEGER, INTENT(in) :: iunit
      INTEGER, INTENT(out):: ierr
      INTEGER, OPTIONAL :: iopen
!------------------------------------------------
!   L o c a l   V a r i a b l e s
!------------------------------------------------
      INTEGER :: istat
      CHARACTER(LEN=256) :: vmec_version
      LOGICAL :: exfile
!------------------------------------------------
!
!     User opened the file externally and has a unit number, iunit
!
      ierr = 0

      INQUIRE(unit=iunit, exist=exfile, name=vmec_version,iostat=istat)
      IF (istat.ne.0 .or. .not.exfile) THEN
        PRINT *,' In READ_WOUT_FILE, Unit = ',iunit,                    &
                ' File = ',TRIM(vmec_version),' DOES NOT EXIST'
        IF (PRESENT(iopen)) iopen = -1
        ierr = -1
        RETURN
      ELSE
        IF (PRESENT(iopen)) iopen = 0
      END IF

      CALL read_wout_text(iunit, ierr)
      lwout_opened = (ierr .eq. 0)
      lthreed = ANY(NINT(xn) .ne. 0)

      END SUBROUTINE readw_only

      SUBROUTINE Compute_Currents(bsubsmnc_, bsubsmns_,                          &
     &                            bsubumnc_, bsubumns_,                          &
     &                            bsubvmnc_, bsubvmns_,                          &
     &                            xm_nyq_, xn_nyq_, mnmax_nyq_,                  &
     &                            lasym_, ns_,     &
     &                            currumnc_, currvmnc_,                          &
     &                            currumns_, currvmns_)
      USE stel_constants, ONLY: mu0
      IMPLICIT NONE

      REAL(rprec), DIMENSION(:,:), INTENT(in)  :: bsubsmnc_
      REAL(rprec), DIMENSION(:,:), INTENT(in)  :: bsubsmns_
      REAL(rprec), DIMENSION(:,:), INTENT(in)  :: bsubumnc_
      REAL(rprec), DIMENSION(:,:), INTENT(in)  :: bsubumns_
      REAL(rprec), DIMENSION(:,:), INTENT(in)  :: bsubvmnc_
      REAL(rprec), DIMENSION(:,:), INTENT(in)  :: bsubvmns_

      REAL(rprec), DIMENSION(:), INTENT(in)    :: xm_nyq_
      REAL(rprec), DIMENSION(:), INTENT(in)    :: xn_nyq_

      INTEGER, INTENT(in)                      :: mnmax_nyq_
      LOGICAL, INTENT(in)                      :: lasym_
      INTEGER, INTENT(in)                      :: ns_

      REAL(rprec), DIMENSION(:,:), INTENT(out) :: currumnc_
      REAL(rprec), DIMENSION(:,:), INTENT(out) :: currvmnc_
      REAL(rprec), DIMENSION(:,:), INTENT(out) :: currumns_
      REAL(rprec), DIMENSION(:,:), INTENT(out) :: currvmns_

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------
      INTEGER :: js
      REAL(rprec) :: ohs, hs, shalf(ns_), sfull(ns_)
      REAL(rprec), DIMENSION(mnmax_nyq_) :: bu1, bu0, bv1, bv0, t1, t2, t3
!-----------------------------------------------
!
!     Computes current harmonics for currXmn == sqrt(g)*JsupX, X = u,v
!     [Corrected above "JsubX" to "JsupX", JDH 2010-08-16]

!     NOTE: bsub(s,u,v)mn are on HALF radial grid
!          (in earlier versions, bsubsmn was on FULL radial grid)

!     NOTE: near the axis, b_s is dominated by the m=1 component of gsv ~ cos(u)/sqrt(s)
!           we average it with a weight-factor of sqrt(s)
!

      ohs = (ns_-1)
      hs  = 1._dp/ohs

      DO js = 2, ns_
         shalf(js) = SQRT(hs*(js-1.5_dp))
         sfull(js) = SQRT(hs*(js-1))
      END DO

      DO js = 2, ns_-1
         WHERE (MOD(INT(xm_nyq_),2) .EQ. 1)
            t1 = 0.5_dp*(shalf(js+1)*bsubsmns_(:,js+1)                         &
               +         shalf(js)  *bsubsmns_(:,js)) /sfull(js)
            bu0 = bsubumnc_(:,js  )/shalf(js)
            bu1 = bsubumnc_(:,js+1)/shalf(js+1)
            t2 = ohs*(bu1-bu0)*sfull(js)+0.25_dp*(bu0+bu1)/sfull(js)
            bv0 = bsubvmnc_(:,js  )/shalf(js)
            bv1 = bsubvmnc_(:,js+1)/shalf(js+1)
            t3 = ohs*(bv1-bv0)*sfull(js)+0.25_dp*(bv0+bv1)/sfull(js)
         ELSEWHERE
            t1 = 0.5_dp*(bsubsmns_(:,js+1)+bsubsmns_(:,js))
            t2 = ohs*(bsubumnc_(:,js+1)-bsubumnc_(:,js))
            t3 = ohs*(bsubvmnc_(:,js+1)-bsubvmnc_(:,js))
         ENDWHERE
         currumnc_(:,js) = -xn_nyq_(:)*t1 - t3
         currvmnc_(:,js) = -xm_nyq_(:)*t1 + t2
      END DO         
   
      WHERE (xm_nyq_ .LE. 1)
         currvmnc_(:,1) =  2*currvmnc_(:,2) - currvmnc_(:,3)
         currumnc_(:,1) =  2*currumnc_(:,2) - currumnc_(:,3)
      ELSEWHERE
         currvmnc_(:,1) = 0
         currumnc_(:,1) = 0
      ENDWHERE

      currumnc_(:,ns_) = 2*currumnc_(:,ns_-1) - currumnc_(:,ns_-2)
      currvmnc_(:,ns_) = 2*currvmnc_(:,ns_-1) - currvmnc_(:,ns_-2)
      currumnc_ = currumnc_ /mu0;   currvmnc_ = currvmnc_/mu0

      IF (.NOT.lasym_) RETURN

      DO js = 2, ns_-1
         WHERE (MOD(INT(xm_nyq_),2) .EQ. 1)
            t1 = 0.5_dp*(shalf(js+1)*bsubsmnc_(:,js+1)                         &
               +         shalf(js)  *bsubsmnc_(:,js)) / sfull(js)
            bu0 = bsubumns_(:,js  )/shalf(js+1)
            bu1 = bsubumns_(:,js+1)/shalf(js+1)
            t2 = ohs*(bu1-bu0)*sfull(js) + 0.25_dp*(bu0+bu1)/sfull(js)
            bv0 = bsubvmns_(:,js  )/shalf(js)
            bv1 = bsubvmns_(:,js+1)/shalf(js+1)
            t3 = ohs*(bv1-bv0)*sfull(js)+0.25_dp*(bv0+bv1)/sfull(js)
         ELSEWHERE
            t1 = 0.5_dp*(bsubsmnc_(:,js+1) + bsubsmnc_(:,js))
            t2 = ohs*(bsubumns_(:,js+1)-bsubumns_(:,js))
            t3 = ohs*(bsubvmns_(:,js+1)-bsubvmns_(:,js))
         END WHERE
         currumns_(:,js) =  xn_nyq_(:)*t1 - t3
         currvmns_(:,js) =  xm_nyq_(:)*t1 + t2
      END DO         

      WHERE (xm_nyq_ .LE. 1)
         currvmns_(:,1) =  2*currvmns_(:,2) - currvmns_(:,3)
         currumns_(:,1) =  2*currumns_(:,2) - currumns_(:,3)
      ELSEWHERE
         currvmns_(:,1) = 0
         currumns_(:,1) = 0
      END WHERE
      currumns_(:,ns_) = 2*currumns_(:,ns_-1) - currumns_(:,ns_-2)
      currvmns_(:,ns_) = 2*currvmns_(:,ns_-1) - currvmns_(:,ns_-2)
      currumns_ = currumns_/mu0;   currvmns_ = currvmns_/mu0

      END SUBROUTINE Compute_Currents

      SUBROUTINE read_wout_deallocate
      IMPLICIT NONE
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------
      INTEGER :: istat(10)
!-----------------------------------------------
      istat=0
      lwout_opened=.false.

      IF (ALLOCATED(extcur)) DEALLOCATE (extcur, curlabel,              &
              stat = istat(1))
      IF (ALLOCATED(overr)) DEALLOCATE (overr, stat = istat(2))

      IF (ALLOCATED(xm)) DEALLOCATE (xm, xn, xm_nyq, xn_nyq,            &
        rmnc, zmns, lmns, bmnc, gmnc, bsubumnc, iotaf, presf, phipf,    &
        chipf,                                                          &
        bsubvmnc, bsubsmns, bsupumnc, bsupvmnc, currvmnc, iotas, mass,  &
        pres, beta_vol, phip, buco, bvco, phi, vp, jcuru, am, ac, ai,   &
        jcurv, specw, Dmerc, Dshear, Dwell, Dcurr, Dgeod, equif, jdotb, &
        bdotb, bdotgradv, raxis, zaxis, fsqt, wdot, stat=istat(3))
 
      IF (ALLOCATED(potsin)) DEALLOCATE (potsin)
      IF (ALLOCATED(potcos)) DEALLOCATE (potcos)

      IF (ALLOCATED(chipf)) DEALLOCATE (chipf, chi)

      IF (ALLOCATED(am_aux_s)) DEALLOCATE (am_aux_s, am_aux_f,          &
          ac_aux_s, ac_aux_f, ai_aux_s, ai_aux_f, stat=istat(6)) 

      IF (ireconstruct.gt.0 .and. ALLOCATED(sknots)) DEALLOCATE (       &
          ystark, y2stark, pknots, anglemse, rmid, qmid, shear,         &
          presmid, alfa, curmid, rstark, datastark, rthom, datathom,    &
          ythom, y2thom, plflux, dsiobt, bcoil, plbfld, bbc, sknots,    &
          pfcspec, limitr, rlim, zlim, nsetsn, stat = istat(4))

      IF (ALLOCATED(rmns)) DEALLOCATE (rmns, zmnc, lmnc,                &
          bmns, gmns, bsubumns, bsubvmns, bsubsmnc,                     &
          bsupumns, bsupvmns, stat=istat(5))

      IF (ALLOCATED(bsubumnc_sur)) THEN
         DEALLOCATE(bsubumnc_sur, bsubvmnc_sur,                         &
                    bsupumnc_sur, bsupvmnc_sur)
      END IF

      IF (ALLOCATED(bsubumns_sur)) THEN
         DEALLOCATE(bsubumns_sur, bsubvmns_sur,                         &
                    bsupumns_sur, bsupvmns_sur)
      END IF

!  Note currvmnc deallocated above.
      IF (ALLOCATED(currumnc)) DEALLOCATE (currumnc)
      IF (ALLOCATED(currumns)) DEALLOCATE (currumns, currvmns)
      IF (ALLOCATED(rzl_local)) DEALLOCATE (rzl_local)

      IF (ANY(istat .ne. 0))                                            &
            STOP 'Deallocation error in read_wout_deallocate'

      END SUBROUTINE read_wout_deallocate

      SUBROUTINE tosuvspace (s_in, u_in, v_in, gsqrt,                   &
                             bsupu, bsupv, jsupu, jsupv, lam)
      USE stel_constants, ONLY: zero, one
      IMPLICIT NONE
!------------------------------------------------
!   D u m m y   A r g u m e n t s
!------------------------------------------------
      REAL(rprec), INTENT(in) :: s_in, u_in, v_in
      REAL(rprec), INTENT(out), OPTIONAL :: gsqrt, bsupu, bsupv,        &
                                            jsupu, jsupv, lam
!------------------------------------------------
!   L o c a l   V a r i a b l e s
!------------------------------------------------
      REAL(rprec), PARAMETER :: c1p5 = 1.5_dp
      INTEGER :: m, n, n1, mn, ipresent, jslo, jshi
      REAL(rprec) :: hs1, wlo, whi, wlo_odd, whi_odd
      REAL(rprec), DIMENSION(mnmax_nyq) :: gmnc1, gmns1, bsupumnc1,     &
         bsupumns1, bsupvmnc1, bsupvmns1, jsupumnc1, jsupumns1,         &
         jsupvmnc1, jsupvmns1, wmins, wplus,  lammns1, lammnc1
      REAL(rprec) :: cosu, sinu, cosv, sinv, tcosmn, tsinmn, sgn
      REAL(rprec) :: cosmu(0:mnyq), sinmu(0:mnyq),                      &
                     cosnv(0:nnyq), sinnv(0:nnyq)
      LOGICAL :: lgsqrt, lbsupu, lbsupv, ljsupu, ljsupv, llam
!------------------------------------------------
!
!     COMPUTE VARIOUS HALF/FULL-RADIAL GRID QUANTITIES AT THE INPUT POINT
!     (S, U, V) , WHERE 
!        S = normalized toroidal flux (0 - 1),
!        U = poloidal angle 
!        V = N*phi = toroidal angle * no. field periods
!
!     HALF-RADIAL GRID QUANTITIES
!     gsqrt, bsupu, bsupv
!   
!     FULL-RADIAL GRID QUANTITIES
!     dbsubuds, dbsubvds, dbsubsdu, dbsubsdv
!
!------------------------------------------------
      IF (s_in.lt.zero .or. s_in.gt.one) THEN
         WRITE(6, *) ' In tosuvspace, s(flux) must be between 0 and 1'
         RETURN
      END IF

      IF (.not.lwout_opened) THEN
         WRITE(6, *)' tosuvspace can only be called AFTER opening wout file!'
         RETURN
      END IF

!
!     SETUP TRIG ARRAYS
!
      cosu = COS(u_in);   sinu = SIN(u_in)
      cosv = COS(v_in);   sinv = SIN(v_in)

      cosmu(0) = 1;    sinmu(0) = 0
      cosnv(0) = 1;    sinnv(0) = 0
      DO m = 1, mnyq
         cosmu(m) = cosmu(m-1)*cosu - sinmu(m-1)*sinu
         sinmu(m) = sinmu(m-1)*cosu + cosmu(m-1)*sinu
      END DO

      DO n = 1, nnyq
         cosnv(n) = cosnv(n-1)*cosv - sinnv(n-1)*sinv
         sinnv(n) = sinnv(n-1)*cosv + cosnv(n-1)*sinv
      END DO


!
!     FIND INTERPOLATED s VALUE AND COMPUTE INTERPOLATION WEIGHTS wlo, whi
!     RECALL THAT THESE QUANTITIES ARE ON THE HALF-RADIAL GRID...
!     s-half(j) = (j-1.5)*hs, for j = 2,...ns
!
      hs1 = one/(ns-1)
      jslo = INT(c1p5 + s_in/hs1)
      jshi = jslo+1
      wlo = (hs1*(jshi-c1p5) - s_in)/hs1
      whi = 1 - wlo
      IF (jslo .eq. ns) THEN
!        USE Xhalf(ns+1) = 2*Xhalf(ns) - Xhalf(ns-1) FOR "GHOST" POINT VALUE 1/2hs OUTSIDE EDGE
!        THEN, X = wlo*Xhalf(ns) + whi*Xhalf(ns+1) == Xhalf(ns) + whi*(Xhalf(ns) - Xhalf(ns-1)) 
         jshi = jslo-1
         wlo = 1+whi; whi = -whi
      ELSE IF (jslo .eq. 1) THEN
         jslo = 2
      END IF

!
!     FOR ODD-m MODES X ~ SQRT(s), SO INTERPOLATE Xmn/SQRT(s)
! 
      whi_odd = whi*SQRT(s_in/(hs1*(jshi-c1p5)))
      IF (jslo .ne. 1) THEN
         wlo_odd = wlo*SQRT(s_in/(hs1*(jslo-c1p5)))
      ELSE
         wlo_odd = 0
         whi_odd = SQRT(s_in/(hs1*(jshi-c1p5)))
      END IF

      WHERE (MOD(NINT(xm_nyq(:)),2) .eq. 0)
         wmins = wlo
         wplus = whi
      ELSEWHERE
         wmins = wlo_odd
         wplus = whi_odd
      END WHERE

      ipresent = 0
      lgsqrt = PRESENT(gsqrt)
      IF (lgsqrt) THEN
         gsqrt = 0 ;  ipresent = ipresent+1
         gmnc1 = wmins*gmnc(:,jslo) + wplus*gmnc(:,jshi)
         IF (lasym) gmns1 = wmins*gmns(:,jslo) + wplus*gmns(:,jshi)
      END IF
      lbsupu = PRESENT(bsupu)
      IF (lbsupu) THEN
         bsupu = 0 ;  ipresent = ipresent+1
         bsupumnc1 = wmins*bsupumnc(:,jslo) + wplus*bsupumnc(:,jshi)
         IF (lasym) bsupumns1 = wmins*bsupumns(:,jslo) + wplus*bsupumns(:,jshi)
      END IF
      lbsupv = PRESENT(bsupv)
      IF (lbsupv) THEN
         bsupv = 0 ;  ipresent = ipresent+1
         bsupvmnc1 = wmins*bsupvmnc(:,jslo) + wplus*bsupvmnc(:,jshi)
         IF (lasym) bsupvmns1 = wmins*bsupvmns(:,jslo) + wplus*bsupvmns(:,jshi)
      END IF
      llam = PRESENT(lam)
      IF (llam) THEN
         lam = zero ;  ipresent = ipresent+1
         lammns1 = wmins*lmns(:,jslo) + wplus*lmns(:,jshi)
         IF (lasym) lammnc1 = wmins*lmnc(:,jslo) + wplus*lmnc(:,jshi)
      END IF

      IF (ipresent .eq. 0) GOTO 1000

!
!     COMPUTE GSQRT, ... IN REAL SPACE
!     tcosmn = cos(mu - nv);  tsinmn = sin(mu - nv)
!
      DO mn = 1, mnmax_nyq
         m = NINT(xm_nyq(mn));  n = NINT(xn_nyq(mn))/nfp
         n1 = ABS(n);   sgn = SIGN(1,n)
         tcosmn = cosmu(m)*cosnv(n1) + sgn*sinmu(m)*sinnv(n1)   
         tsinmn = sinmu(m)*cosnv(n1) - sgn*cosmu(m)*sinnv(n1)
         IF (lgsqrt) gsqrt = gsqrt + gmnc1(mn)*tcosmn
         IF (lbsupu) bsupu = bsupu + bsupumnc1(mn)*tcosmn
         IF (lbsupv) bsupv = bsupv + bsupvmnc1(mn)*tcosmn
         IF (llam)   lam   = lam   + lammns1(mn)*tsinmn
      END DO

      IF (.not.lasym) GOTO 1000

      DO mn = 1, mnmax_nyq
         m = NINT(xm_nyq(mn));  n = NINT(xn_nyq(mn))/nfp
         n1 = ABS(n);   sgn = SIGN(1,n)
         tcosmn = cosmu(m)*cosnv(n1) + sgn*sinmu(m)*sinnv(n1)   
         tsinmn = sinmu(m)*cosnv(n1) - sgn*cosmu(m)*sinnv(n1)
         IF (lgsqrt) gsqrt = gsqrt + gmns1(mn)*tsinmn
         IF (lbsupu) bsupu = bsupu + bsupumns1(mn)*tsinmn
         IF (lbsupv) bsupv = bsupv + bsupvmns1(mn)*tsinmn
         IF (llam)   lam   = lam   + lammnc1(mn)*tcosmn
      END DO

 1000 CONTINUE

!     FULL-MESH QUANTITIES
!
!     FIND INTERPOLATED s VALUE AND COMPUTE INTERPOLATION WEIGHTS wlo, whi
!     RECALL THAT THESE QUANTITIES ARE ON THE FULL-RADIAL GRID...
!     s-full(j) = (j-1)*hs, for j = 1,...ns
!
      hs1 = one/(ns-1)
      jslo = 1+INT(s_in/hs1)
      jshi = jslo+1
      IF (jslo .eq. ns) jshi = ns
      wlo = (hs1*(jshi-1) - s_in)/hs1
      whi = 1 - wlo
!
!     FOR ODD-m MODES X ~ SQRT(s), SO INTERPOLATE Xmn/SQRT(s)
! 
      whi_odd = whi*SQRT(s_in/(hs1*(jshi-1)))
      IF (jslo .ne. 1) THEN
         wlo_odd = wlo*SQRT(s_in/(hs1*(jslo-1)))
      ELSE
         wlo_odd = 0
         whi_odd = SQRT(s_in/(hs1*(jshi-1)))
      END IF

      WHERE (MOD(NINT(xm_nyq(:)),2) .eq. 0)
         wmins = wlo
         wplus = whi
      ELSEWHERE
         wmins = wlo_odd
         wplus = whi_odd
      END WHERE

      ipresent = 0
      ljsupu = PRESENT(jsupu)
      IF (ljsupu) THEN
         IF (.not.lgsqrt) STOP 'MUST compute gsqrt for jsupu'
         jsupu = 0 ;  ipresent = ipresent+1
         jsupumnc1 = wmins*currumnc(:,jslo) + wplus*currumnc(:,jshi)
         IF (lasym) jsupumns1 = wmins*currumns(:,jslo) + wplus*currumns(:,jshi)
      END IF

      ljsupv = PRESENT(jsupv)
      IF (ljsupv) THEN
         IF (.not.lgsqrt) STOP 'MUST compute gsqrt for jsupv'
         jsupv = 0 ;  ipresent = ipresent+1
         jsupvmnc1 = wmins*currvmnc(:,jslo) + wplus*currvmnc(:,jshi)
         IF (lasym) jsupvmns1 = wmins*currvmns(:,jslo) + wplus*currvmns(:,jshi)
      END IF

      IF (ipresent .eq. 0) RETURN

      DO mn = 1, mnmax_nyq
         m = NINT(xm_nyq(mn));  n = NINT(xn_nyq(mn))/nfp
         n1 = ABS(n);   sgn = SIGN(1,n)
         tcosmn = cosmu(m)*cosnv(n1) + sgn*sinmu(m)*sinnv(n1)   
         IF (ljsupu) jsupu = jsupu + jsupumnc1(mn)*tcosmn
         IF (ljsupv) jsupv = jsupv + jsupvmnc1(mn)*tcosmn
      END DO

      IF (.not.lasym) GOTO 2000

      DO mn = 1, mnmax_nyq
         m = NINT(xm_nyq(mn));  n = NINT(xn_nyq(mn))/nfp
         n1 = ABS(n);   sgn = SIGN(1,n)
         tsinmn = sinmu(m)*cosnv(n1) - sgn*cosmu(m)*sinnv(n1)
         IF (ljsupu) jsupu = jsupu + jsupumns1(mn)*tsinmn
         IF (ljsupv) jsupv = jsupv + jsupvmns1(mn)*tsinmn
      END DO

 2000 CONTINUE

      IF (ljsupu) jsupu = jsupu/gsqrt
      IF (ljsupv) jsupv = jsupv/gsqrt

      END SUBROUTINE tosuvspace

      SUBROUTINE LoadRZL
      IMPLICIT NONE
!------------------------------------------------
!   L o c a l   V a r i a b l e s
!------------------------------------------------
      INTEGER     :: rcc, rss, zsc, zcs, rsc, rcs, zcc, zss
      INTEGER     :: mpol1, mn, m, n, n1
      REAL(rprec) :: sgn
!------------------------------------------------
!
!     Arrays must be stacked (and ns,ntor,mpol ordering imposed)
!     as coefficients of cos(mu)*cos(nv), etc
!     Only need R, Z components(not lambda, for now anyhow)
!      
      IF (ALLOCATED(rzl_local)) RETURN

      mpol1 = mpol-1
      rcc = 1;  zsc = 1
      IF (.not.lasym) THEN
         IF (lthreed) THEN
            ntmax = 2
            rss = 2;  zcs = 2
         ELSE
            ntmax = 1
         END IF
      ELSE
         IF (lthreed) THEN
            ntmax = 4
            rss = 2;  rsc = 3;  rcs = 4
            zcs = 2;  zcc = 3;  zss = 4
         ELSE
            ntmax = 2
            rsc = 2;  zcc = 2
         END IF
      END IF

!     really only need to ALLOCATE 2*ntmax (don't need lambdas)
!     for consistency, we'll allocate 3*ntmax and set lambdas = 0
      zsc = 1+ntmax; zcs = zcs+ntmax; zcc = zcc+ntmax; zss = zss+ntmax
      ALLOCATE(rzl_local(ns,0:ntor,0:mpol1,3*ntmax), stat=n)
      IF (n .ne. 0) STOP 'Allocation error in LoadRZL'
      rzl_local = 0

      DO mn = 1, mnmax
         m = NINT(xm(mn));  n = NINT(xn(mn))/nfp; n1 = ABS(n)
         sgn = SIGN(1, n)
         rzl_local(:,n1,m,rcc) = rzl_local(:,n1,m,rcc) + rmnc(mn,:)
         rzl_local(:,n1,m,zsc) = rzl_local(:,n1,m,zsc) + zmns(mn,:)
         IF (lthreed) THEN
            rzl_local(:,n1,m,rss) = rzl_local(:,n1,m,rss) + sgn*rmnc(mn,:)
            rzl_local(:,n1,m,zcs) = rzl_local(:,n1,m,zcs) - sgn*zmns(mn,:)
         END IF
         IF (lasym) THEN
            rzl_local(:,n1,m,rsc) = rzl_local(:,n1,m,rsc) + rmns(mn,:)
            rzl_local(:,n1,m,zcc) = rzl_local(:,n1,m,zcc) + zmnc(mn,:)
            IF (lthreed) THEN
                rzl_local(:,n1,m,rcs) = rzl_local(:,n1,m,rcs)           &
                                      - sgn*rmns(mn,:)
                rzl_local(:,n1,m,zss) = rzl_local(:,n1,m,zss)           &
                                      + sgn*zmnc(mn,:)
            END IF
         END IF
      END DO

      END SUBROUTINE LoadRZL

      END MODULE read_wout_mod

