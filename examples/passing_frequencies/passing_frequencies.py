import numpy as np
import time
import sys

from simsopt.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from simsopt.util.constants import (
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
    ALPHA_PARTICLE_CHARGE,
)
from simsopt.util.functions import proc0_print
from simsopt.field.trajectory_helpers import PassingPoincare

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_size = comm.size
    verbose = comm.rank == 0
except ImportError:
    comm = None
    comm_size = 1
    verbose = True

boozmn_filename = "../inputs/boozmn_aten_rescaled.nc"

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY

resolution = 48  # Resolution for field interpolation
sign_vpar = 1.0  # sign(vpar). should be +/- 1.
lam = 0  # lambda = v_perp^2/(v^2 B) = const. along trajectory
ntheta_poinc = 1  # Number of zeta initial conditions for poincare
ns_poinc = 120  # Number of s initial conditions for poincare
Nmaps = 100  # Number of Poincare return maps to compute
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for interpolation
tol = 1e-8  # Tolerance for ODE solver
degree = 3  # Degree for Lagrange interpolation

sys.stdout = open(f"stdout_passing_freq_{resolution}_{comm_size}.txt", "a", buffering=1)

time1 = time.time()
M = 1
N = 4
bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm, helicity_M=M, helicity_N=N)

field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

poinc = PassingPoincare(
    field,
    lam,
    sign_vpar,
    mass,
    charge,
    Ekin,
    ns_poinc=ns_poinc,
    ntheta_poinc=ntheta_poinc,
    Nmaps=Nmaps,
    comm=comm,
    solver_options={"reltol": tol, "abstol": tol, "axis": 0},
)

omega_theta_prof, omega_zeta_prof, s_prof = poinc.compute_frequencies()
points = np.zeros((len(s_prof), 3))
points[:, 0] = s_prof
field.set_points(points)
iota = field.iota()[:, 0]

if verbose:
    import matplotlib

    matplotlib.use("Agg")  # Don't use interactive backend
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(s_prof, omega_theta_prof / omega_zeta_prof, label="Passing frequency")
    plt.plot(s_prof, iota, "--", label="Rotational transform")
    plt.xlabel("s")
    plt.ylabel(r"$\omega_\theta/\omega_\zeta$")
    plt.legend()
    plt.savefig(f"passing_frequencies.png")
    plt.close()

time2 = time.time()

proc0_print("poincare time: ", time2 - time1)
