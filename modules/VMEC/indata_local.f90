MODULE indata_local
USE vparams, ONLY: rprec, dp
IMPLICIT NONE

INTEGER, PARAMETER :: nsd = 10001    
INTEGER, PARAMETER :: mpold = 101    
INTEGER, PARAMETER :: ntord = 101    
INTEGER, PARAMETER :: ndatafmax  = 101
INTEGER, PARAMETER :: nstore_seq = 100
INTEGER, PARAMETER :: mpol1d = mpold - 1

INTEGER, PARAMETER :: mpol_default = 6
INTEGER, PARAMETER :: ntor_default = 0
INTEGER, PARAMETER :: ns_default   = 31

REAL(rprec), DIMENSION(-ntord:ntord,0:mpol1d) :: vmec_rbs, vmec_zbc, vmec_rbc, vmec_zbs
REAL(rprec), DIMENSION(0:ntord) :: vmec_raxis_cc, vmec_raxis_cs, vmec_zaxis_cc, vmec_zaxis_cs

CONTAINS
  
  SUBROUTINE assign()
    USE vmec_input, ONLY: rbc, rbs, zbc, zbs, raxis_cc, raxis_cs, zaxis_cc, zaxis_cs
    IMPLICIT NONE
    rbc = vmec_rbc
    rbs = vmec_rbs
    zbc = vmec_zbc
    zbs = vmec_zbs
    raxis_cc = vmec_raxis_cc
    raxis_cs = vmec_raxis_cs
    zaxis_cc = vmec_zaxis_cc
    zaxis_cs = vmec_zaxis_cs
    return
  END SUBROUTINE assign

  SUBROUTINE backup()
    USE vmec_input, ONLY: rbc, rbs, zbc, zbs, raxis_cc, raxis_cs, zaxis_cc, zaxis_cs
    IMPLICIT NONE
    vmec_rbc = rbc
    vmec_rbs = rbs
    vmec_zbc = zbc
    vmec_zbs = zbs
    vmec_raxis_cc = raxis_cc
    vmec_raxis_cs = raxis_cs
    vmec_zaxis_cc = zaxis_cc
    vmec_zaxis_cs = zaxis_cs
    return
  END SUBROUTINE backup

END MODULE indata_local

