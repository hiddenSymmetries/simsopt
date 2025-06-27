import sys
import numpy as np
import time

from simsopt.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenHarmonic
)
from simsopt.field.tracing import (
    trace_particles_boozer_perturbed,
    MaxToroidalFluxStoppingCriterion,
)
from simsopt.field.tracing_helpers import (
    initialize_position_profile,
    initialize_velocity_uniform,
)
from simsopt.util.constants import (
    ALPHA_PARTICLE_MASS,
    ALPHA_PARTICLE_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from simsopt.util.functions import proc0_print

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    verbose = comm.rank == 0
    comm_size = comm.size
except ImportError:
    comm = None
    verbose = True
    comm_size = 1

resolution = 48  # Resolution for field interpolation
nParticles = 5000  # Number of particles to trace
reltol = 1e-8  # Relative tolerance for the ODE solver
abstol = 1e-8  # Absolute tolerance for the ODE solver
order = 3  # Order for radial interpolation
degree = 3  # Degree for 3d interpolation
boozmn_filename = "../inputs/boozmn_QA_bootstrap.nc"
tmax = 1e-2  # Time for integration
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution

# SAW parameters
nParticles = 5000
Phihat = -1.50119e3
Phim = 1
Phin = 1
omega = 136041 # (1/9) * 1.224365528939409647*10^6 = omega for (4/9) co-passing orbit
phase = 0

sys.stdout = open(f"stdout_{nParticles}_{resolution}_{comm_size}.txt", "a", buffering=1)

## Setup radial interpolation
bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm)

## Setup 3d interpolation
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

saw = ShearAlfvenHarmonic(
    Phihat,
    Phim,
    Phin,
    omega,
    phase,
    field 
)

# Define fusion birth distribution
# Bader, A., et al. "Modeling of energetic particle transport in optimized stellarators." Nuclear Fusion 61.11 (2021): 116060.
nD = lambda s: (1 - s**5)  # Normalized density
nT = nD
T = lambda s: 11.5 * (1 - s)  # Temperature in keV


# D-T cross-section
def sigmav(T):
    if T > 0:
        return T ** (-2 / 3) * np.exp(-19.94 * T ** (-1 / 3))
    else:
        return 0


# Reactivity profile
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))

points = initialize_position_profile(field, nParticles, reactivity, comm=comm)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Initialize uniformly distributed parallel velocities
vpar0 = np.sqrt(2 * Ekin / mass)
vpar_init = initialize_velocity_uniform(vpar0, nParticles, comm=comm)

field.set_points(points)
mu_init = (vpar0**2 - vpar_init**2)/(2*field.modB()[:,0])

proc0_print(np.shape(mu_init))

time1 = time.time()

## Trace alpha particles in Boozer coordinates until they hit the s = 1 surface
res_tys, res_zeta_hits = trace_particles_boozer_perturbed(
    saw,
    points,
    vpar_init,
    mu_init,
    mass=mass,
    charge=charge,
    comm=comm,
    Ekin=Ekin,
    stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
    forget_exact_path=True,
    abstol=abstol,
    reltol=reltol,
    tmax=tmax
)

time2 = time.time()
proc0_print("Elapsed time for tracing = ", time2 - time1)

## Post-process results to obtain lost particles
if verbose:
    from simsopt.field.trajectory_helpers import compute_loss_fraction

    times, loss_frac = compute_loss_fraction(res_tys, tmin=1e-5, tmax=1e-2)
    import matplotlib

    matplotlib.use("Agg")  # Don't use interactive backend
    import matplotlib.pyplot as plt

    plt.figure()
    plt.loglog(times, loss_frac)
    plt.xlim([1e-5, 1e-2])
    plt.ylim([1e-3, 1])
    plt.xlabel("Time [s]")
    plt.ylabel("Fraction of lost particles")
    plt.savefig("loss_fraction.png")
