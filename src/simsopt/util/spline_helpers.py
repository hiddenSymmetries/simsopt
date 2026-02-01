import numpy as np
from simsopt.mhd import Vmec
from simsopt.util.mpi import MpiPartition

mpi = MpiPartition()
mpi.write()

def vmec_from_surf(
        nfp,
        surf=None,
        M = 12,
        N = 12,
        ns=13,
        ntheta=32,
        nzeta=32,
        ftol=1e-7,
        phiedge = 1,
        verbose=False,
        niter=3000
    ):
    '''
    Generate VMEC object from any optimizable object with a
    `to_RZFourier` method. 
    '''
    # runtime params
    vmec = Vmec(mpi = mpi, verbose=verbose)
    # N=6
    # r_n, z_n = surf.centroid_axis_fourier_coeffs(N=6)
    # r_n = r_n[N:]
    # z_n = z_n[N:]

    vmec.indata.delt = 9e-1
    # vmec.indata.niter = 2000
    # vmec.indata.nstep = 1e2
    vmec.indata.tcon0 = 2
    vmec.indata.ns_array = np.append(np.array([ns]), np.zeros(99,))
    vmec.indata.niter_array = np.append(np.array([niter]), -1*np.ones(99,))
    vmec.indata.ftol_array = np.append(np.array([ftol]), np.zeros(99,))
    vmec.indata.precon_type = 'none'
    vmec.indata.prec2d_threshold = 1e-19
    # grid params
    vmec.indata.lasym = 0
    vmec.indata.nfp = nfp
    vmec.indata.mpol = M
    vmec.indata.ntor = N
    vmec.indata.ntheta = ntheta
    vmec.indata.nzeta = nzeta
    vmec.indata.phiedge = phiedge
    # free bdry params
    vmec.indata.lfreeb = 0
    vmec.indata.nvacskip = 6
    # pressure params
    vmec.indata.gamma = 0
    vmec.indata.bloat = 1
    vmec.indata.spres_ped = 1
    vmec.indata.pres_scale = 1
    vmec.indata.pmass_type = 'power_series'
    vmec.indata.am = 0
    # current/iota params
    vmec.indata.curtor = 0
    vmec.indata.ncurr = 1
    vmec.indata.piota_type = 'power_series'
    vmec.indata.pcurr_type = 'power_series'

    vmec.boundary = surf
    vmec.set_indata()
    return vmec