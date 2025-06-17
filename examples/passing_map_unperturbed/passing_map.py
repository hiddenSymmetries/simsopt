import numpy as np
import time
import sys 

from booz_xform import Booz_xform
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from simsopt.util.constants import ALPHA_PARTICLE_MASS, FUSION_ALPHA_PARTICLE_ENERGY,ALPHA_PARTICLE_CHARGE
from simsopt.plotting.plotting_helpers import passing_poincare_plot
from simsopt.util.functions import proc0_print

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.size 
except ImportError:
    comm = None
    comm_size = 1

boozmn_filename = '../inputs/boozmn_aten_rescaled.nc' 

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY

resolution = 48 # Resolution for field interpolation
sign_vpar = 1.0 # sign(vpar). should be +/- 1. 
lam = 0.0 # lambda = v_perp^2/(v^2 B) = const. along trajectory
ntheta_poinc = 1 # Number of zeta initial conditions for poincare
ns_poinc = 120 # Number of s initial conditions for poincare
Nmaps = 1000 # Number of Poincare return maps to compute
ns_interp = resolution # number of radial grid points for interpolation
ntheta_interp = resolution # number of poloidal grid points for interpolation
nzeta_interp = resolution # number of toroidal grid points for interpolation
order = 3 # order for interpolation
tol = 1e-8 # Tolerance for ODE solver

sys.stdout = open(f"stdout_passing_map_{resolution}_{comm_size}.txt", "a", buffering=1)

equil = Booz_xform()
equil.verbose = 0
equil.read_boozmn(boozmn_filename)
nfp = equil.nfp

time1 = time.time()

bri = BoozerRadialInterpolant(equil,order,no_K=True,comm=comm)

degree = 3
srange = (0, 1, ns_interp)
thetarange = (0, np.pi, ntheta_interp)
zetarange = (0, 2*np.pi/nfp, nzeta_interp)
field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)

s_all, thetas_all, vpars_all = passing_poincare_plot(field,lam,sign_vpar,mass,charge,Ekin,
                                                     ns_poinc=ns_poinc,ntheta_poinc=ntheta_poinc,
                                                     Nmaps=Nmaps,comm=comm,reltol=tol,abstol=tol)

time2 = time.time()

proc0_print('poincare time: ',time2-time1)