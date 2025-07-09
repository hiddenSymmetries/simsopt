import sys
import numpy as np
import time

from simsopt.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from simsopt.field.tracing import (
    trace_particles_boozer,
    MaxToroidalFluxStoppingCriterion,
)
from simsopt.field.tracing_helpers import initialize_position_uniform_vol, initialize_velocity_uniform
from simsopt.util.constants import (
    ALPHA_PARTICLE_MASS,
    ALPHA_PARTICLE_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from simsopt.util.functions import proc0_print
from simsopt.field.trajectory_helpers import compute_peta
from simsopt._core.util import parallel_loop_bounds

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    verbose = comm.rank == 0
    comm_size = comm.size
except ImportError:
    comm = None
    verbose = True
    comm_size = 1

time1 = time.time()

order = 3  # Order for radial interpolation
degree = 3  # Degree for 3d interpolation
boozmn_filename = "../inputs/boozmn_beta2.5_QH.nc"
tmax = 1e-4  # Time for integration
nParticles = 10
helicity_M = 1 # Enforce quasihelical symmetry in radial interpolation
helicity_N = -4

sys.stdout = open(f"stdout_resolution_scan_{comm_size}.txt", "a", buffering=1)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Initialize uniformly distributed parallel velocities
vpar0 = np.sqrt(2 * Ekin / mass)
vpar_init = initialize_velocity_uniform(
    vpar0, nParticles,
)

## Setup radial interpolation with quasisymmetry explicitly enforced 
bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm, 
                              helicity_M=helicity_M,helicity_N=helicity_N)

resolutions = np.array([16, 32, 48, 64, 80, 96])
tols = np.array([1e-4, 1e-6, 1e-8, 1e-10, 1e-12])

resolutions_grid, tols_grid = np.meshgrid(resolutions, tols)
resolutions_grid = resolutions_grid.flatten()
tols_grid = tols_grid.flatten()

errors_grid = np.zeros((len(resolutions),len(tols)))

proc0_print("Starting resolution and tolerance scan with ", nParticles, " particles and ", comm_size, " MPI ranks")
for i in range(len(resolutions)):
    proc0_print("Resolution = ", resolutions[i])
    resolution = resolutions[i]

    ns_interp = resolution
    ntheta_interp = resolution
    nzeta_interp = resolution

    ## Setup 3d interpolation
    field = InterpolatedBoozerField(
        bri,
        degree,
        ns_interp=ns_interp,
        ntheta_interp=ntheta_interp,
        nzeta_interp=nzeta_interp,
    )
    
    points = initialize_position_uniform_vol(field, nParticles, comm=comm, seed=0)

    for j in range(len(tols)): # Tolerance for ODE solver
        proc0_print("  Tolerance = ", tols[j])
        reltol = tols[j]
        abstol = tols[j]

        first, last = parallel_loop_bounds(comm, nParticles)
        errors = [] 
        for k in range(first, last):
            point = np.zeros((1,3))
            point[0, :] = points[k,:]
            ## Trace alpha particle in Boozer coordinates until it hits the s = 1 surface
            res_tys, res_zeta_hits = trace_particles_boozer(
                field,
                point,
                [vpar_init[k]],
                tmax=tmax,
                mass=mass,
                charge=charge,
                Ekin=Ekin,
                stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
                forget_exact_path=False,
                abstol=abstol,
                reltol=reltol,
                dt_save=1e-5,
            )

            # Compute p_eta along trajectory and compare to initial value
            res_ty = res_tys[0] 
            nsteps = len(res_ty[:, 0])
            points = np.zeros((nsteps, 3))
            points[:, 0] = res_ty[:, 1]
            points[:, 1] = res_ty[:, 2]
            points[:, 2] = res_ty[:, 3]
            vpars = res_ty[:, 4]

            peta = compute_peta(field, points, vpars, mass, charge, helicity_M, helicity_N)
            peta_error = (peta - peta[0])/peta[0]
            errors.append(np.max(np.abs(peta_error)))
        
        errors = [i for o in comm.allgather(errors) for i in o]
        proc0_print("  Max error in p_eta = ", np.max(errors))
        errors_grid[i,j] = np.max(errors)

time2 = time.time()
proc0_print("Elapsed time for tracing = ", time2 - time1)

## Post-process results to obtain error in p_eta as a function of resolution and tolerance
if verbose:
    import matplotlib

    matplotlib.use("Agg")  # Don't use interactive backend
    import matplotlib.pyplot as plt

    plt.figure()
    plt.contourf(resolutions, tols, errors_grid.T, levels=20, norm=matplotlib.colors.LogNorm())
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Resolution")
    plt.ylabel("Tolerance")
    plt.colorbar(label="Max Error in $p_\\eta$")
    plt.savefig("error_grid.png")
