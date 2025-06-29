import numpy as np
import time
import sys

from simsopt.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenHarmonic,
)
from simsopt.util.constants import (
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
    ALPHA_PARTICLE_CHARGE,
)
from simsopt.util.functions import proc0_print
from simsopt.field.trajectory_helpers import PassingPerturbedPoincare

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_size = comm.size
    verbose = comm.rank == 0
except ImportError:
    comm = None
    comm_size = 1
    verbose = True

boozmn_filename = "../inputs/boozmn_beta2.5_QA.nc"

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY

resolution = 48  # Resolution for field interpolation
sign_vpar = 1.0  # sign(vpar). should be +/- 1.
lam = 0.1  # lambda = v_perp^2/(v^2 B) = const. along trajectory
nchi_poinc = 1  # Number of chi initial conditions for poincare
ns_poinc = 120  # Number of s initial conditions for poincare
Nmaps = 1000  # Number of Poincare return maps to compute
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for interpolation
tol = 1e-8  # Tolerance for ODE solver
degree = 3  # Degree for Lagrange interpolation
helicity_M = 1  # field strength helicity (QA)
helicity_N = 0  # field strength helicity (QA)

# SAW parameters
Phihat = -1.50119e3
Phim = 1
Phin = 1
omega = 136041
phase = 0

sys.stdout = open(f"stdout_passing_map_{resolution}_{comm_size}.txt", "a", buffering=1)

time1 = time.time()

bri = BoozerRadialInterpolant(
    boozmn_filename,
    order,
    no_K=True,
    comm=comm,
    helicity_M=helicity_M,
    helicity_N=helicity_N,
)

field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

saw = ShearAlfvenHarmonic(Phihat, Phim, Phin, omega, phase, field)

# Point for evaluation of Eprime
p0 = np.zeros((1, 3))
p0[0, 0] = 0.5  # s

poinc = PassingPerturbedPoincare(
    saw,
    sign_vpar,
    mass,
    charge,
    helicity_M,
    helicity_N,
    Ekin=Ekin,
    p0=p0,
    lam=lam,
    ns_poinc=ns_poinc,
    nchi_poinc=nchi_poinc,
    Nmaps=Nmaps,
    comm=comm,
    solver_options={"reltol": tol, "abstol": tol},
)

if verbose:
    poinc.plot_poincare()

time2 = time.time()

proc0_print("poincare time: ", time2 - time1)