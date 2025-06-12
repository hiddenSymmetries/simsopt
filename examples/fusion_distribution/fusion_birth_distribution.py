from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from simsopt.field.tracing import trace_particles_boozer, MaxToroidalFluxStoppingCriterion
from simsopt.field import initialize_position_profile
from simsopt.util.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
import os
import sys
import numpy as np
from booz_xform import Booz_xform
import builtins
import time
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    verbose = (comm.rank == 0)
    comm_size = comm.size 
except ImportError:
    comm = None
    verbose = True
    comm_size = 1
def print(text):
    builtins.print(text)
    os.fsync(sys.stdout)
time1 = time.time()

resolution = 48 # Resolution for field interpolation
nParticles = 5000 # Number of particles to trace
reltol = 1e-8 # Relative tolerance for the ODE solver
abstol = 1e-8 # Absolute tolerance for the ODE solver
order = 3 # Order for radial interpolation
degree = 3 # Degree for 3d interpolation
boozmn_filename = 'boozmn_aten_rescaled.nc' 
tmax = 1e-2 # Time for integration
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution

sys.stdout = open(f"stdout_{nParticles}_{resolution}_{comm_size}.txt", "a", buffering=1)

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
field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, 
    nfp=nfp, stellsym=True, initialize=["modB","psip", "G", "I", "dGds", "dIds", "iota", "modB_derivs"])

# Define fusion birth distribution 
# Bader, A., et al. "Modeling of energetic particle transport in optimized stellarators." Nuclear Fusion 61.11 (2021): 116060.
nD = lambda s: (1 - s**5) # Normalized density
nT = nD
T = lambda s: 11.5 * (1 - s) # Temperature in keV 
# D-T cross-section 
def sigmav(T):
    if T > 0: 
        return T**(-2/3) * np.exp(-19.94 * T**(-1/3)) 
    else: 
        return 0 
# Reactivity profile
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))

points = initialize_position_profile(field, nParticles, reactivity, nfp, comm=comm)

Ekin=FUSION_ALPHA_PARTICLE_ENERGY
mass=ALPHA_PARTICLE_MASS
charge=ALPHA_PARTICLE_CHARGE 
# Initialize uniformly distributed parallel velocities
vpar0=np.sqrt(2*Ekin/mass)
if verbose: 
    vpar_init = np.random.uniform(-vpar0,vpar0,(nParticles,))
else: 
    vpar_init = None
if (comm is not None):
    vpar_init = comm.bcast(vpar_init, root=0)

solver_options = {'abstol': abstol, 'reltol': reltol, 'axis': 2}

## Trace alpha particles in Boozer coordinates until they hit the s = 1 surface 
gc_tys, gc_zeta_hits = trace_particles_boozer(
        field, points, vpar_init, tmax=tmax, mass=mass, charge=charge, comm=comm,
        Ekin=Ekin, stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        forget_exact_path=True, mode='gc_vac', solver_options=solver_options)

## Post-process results to obtain lost particles
if (verbose):
    timelost = []
    theta0 = []
    zeta0 = []
    vpar0 = []
    s0 = []
    slost = []
    thetalost = []
    zetalost = []
    vparlost = []
    for im in range(len(gc_tys)):
        timelost.append(gc_tys[im][1,0])
        s0.append(gc_tys[im][0,1])
        theta0.append(gc_tys[im][0,2])
        zeta0.append(gc_tys[im][0,3])
        vpar0.append(gc_tys[im][0,4])
        slost.append(gc_tys[im][1,1])
        thetalost.append(gc_tys[im][1,2])
        zetalost.append(gc_tys[im][1,3])
        vparlost.append(gc_tys[im][1,4])

    np.savetxt('timelost.txt',timelost)
    np.savetxt('s0lost.txt',s0)
    np.savetxt('theta0lost.txt',theta0)
    np.savetxt('zeta0lost.txt',zeta0)
    np.savetxt('vpar0lost.txt',vpar0)
    np.savetxt('sflost.txt',slost)
    np.savetxt('thetaflost.txt',thetalost)
    np.savetxt('zetaflost.txt',zetalost)
    np.savetxt('vparflost.txt',vparlost)

    time2 = time.time()
    print("Elapsed time for tracing = "+str(time2-time1))
