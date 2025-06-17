import sys
import numpy as np
import time

from booz_xform import Booz_xform
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from simsopt.field.tracing import trace_particles_boozer, MaxToroidalFluxStoppingCriterion
from simsopt.util.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
from simsopt.util.functions import proc0_print 
from simsopt.plotting.plotting_helpers import plot_trajectory_overhead_cyl, plot_trajectory_poloidal

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    verbose = (comm.rank == 0)
    comm_size = comm.size 
except ImportError:
    comm = None
    verbose = True
    comm_size = 1

time1 = time.time()

boozmn_filename = '../inputs/boozmn_aten_rescaled.nc' 
order = 3 # Order for radial interpolation
reltol = 1e-8 # Relative tolerance for the ODE solver
abstol = 1e-8 # Absolute tolerance for the ODE solver
tmax = 1e-4 # Time for integration
resolution = 48 # Resolution for field interpolation
degree = 3 # Degree for 3d interpolation
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution

sys.stdout = open(f"stdout_trajectory_{resolution}_{comm_size}.txt", "a", buffering=1)

## Setup Booz_xform object
equil = Booz_xform()
equil.verbose = 0
equil.read_boozmn(boozmn_filename)
nfp = equil.nfp

## Setup radial interpolation
bri = BoozerRadialInterpolant(equil,order,no_K=True,comm=comm)

## Setup 3d interpolation
srange = (0, 1, ns_interp)
thetarange = (0, np.pi, ntheta_interp)
zetarange = (0, 2*np.pi/nfp, nzeta_interp)
field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)

Ekin=FUSION_ALPHA_PARTICLE_ENERGY
mass=ALPHA_PARTICLE_MASS
charge=ALPHA_PARTICLE_CHARGE 
# Initialize single trapped particle on s = 0.5 surface with random theta and zeta, and zero parallel velocity
vpar0=np.sqrt(2*Ekin/mass)
vpar_init = [0] 
points = np.zeros((1, 3))
points[0, 0] = 0.5 # s = 0.5 surface
points[0, 1] = np.random.uniform(0, 2*np.pi) # Random theta
points[0, 2] = np.random.uniform(0, 2*np.pi/nfp) # Random zeta

solver_options = {'abstol': abstol, 'reltol': reltol, 'axis': 2}

## Trace alpha particle in Boozer coordinates until it hits the s = 1 surface 
## Set forget_exact_path=False to save the trajectory information.  
## Set the dt_save parameter to the time interval for trajectory data 
## to be saved. 
traj_booz, res_hits = trace_particles_boozer(
        field, points, vpar_init, tmax=tmax, mass=mass, charge=charge,
        Ekin=Ekin, stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        forget_exact_path=False, dt_save=1e-7, solver_options=solver_options)

time2 = time.time()

proc0_print('Elapsed time for tracing: ',time2-time1)  

ax = plot_trajectory_overhead_cyl(traj_booz[0],field)

if verbose:
        fig = ax.figure
        fig.savefig('trajectory_overhead_cyl.png', dpi=300, bbox_inches='tight')

ax = plot_trajectory_poloidal(traj_booz[0], helicity_N=nfp)

if verbose: 
        fig = ax.figure 
        fig.savefig('trajectory_poloidal.png', dpi=300, bbox_inches='tight')

        from simsopt.field.trajectory_helpers import trajectory_to_vtk
        trajectory_to_vtk(traj_booz[0], field, filename='trajectory')

time3 = time.time()
proc0_print('Elapsed time for plotting: ',time3-time2)

