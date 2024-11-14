"""
Tracing using InterpolatedBoozerField (The most common use case)
"""
"""
SOME BENCHMARKS FOR REFERENCE

M1 Macbook Pro
==============
res 48 interpolation ~60s
env OMP_NUM_THREADS=1 mpiexec -n 8 python tracing_interp_example.py

res 96 interpolation ~650s
env OMP_NUM_THREADS=2 mpiexec -n 4 python tracing_interp_example.py

res 96 interpolation ~900s
mpiexec -n 2 python tracing_interp_example.py


Ginsburg HPC
==============
res 48 interpolation ~20s
-N 1, --ntasks-per-node=32, --cpus-per-task=1

res 48 interpolation ~35s
-N 1, --ntasks-per-node=8, --cpus-per-task=4

res 96 interpolation ~280s
-N 1, --ntasks-per-node=8, --cpus-per-task=4

"""
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from simsopt.field.tracing import trace_particles_boozer, MinToroidalFluxStoppingCriterion, MaxToroidalFluxStoppingCriterion
from simsopt.util.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
import os
import sys
import numpy as np
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
from booz_xform import Booz_xform
import matplotlib.pyplot as plt
import time

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

resolution = 16 # 96 used for production runs

reltol = 1e-12 # relative tolerance for integration
abstol = 1e-12 # absolute tolerance for integration
boozmn_filename = 'boozmn_QH_boots.nc' # netcdf file, 4608 modes in total
tmax = 1e-3 # Time for integration
ns_interp = resolution # Number of radial grid points for interpolation
ntheta_interp = resolution # Number of poloidal angle grid points for interpolation
nzeta_interp = resolution # Number of toroidal angle grid points for interpolation
N = -4

## Setup Booz_xform object
equil = Booz_xform()
equil.verbose = 0
equil.read_boozmn(boozmn_filename)
nfp = equil.nfp

## Call boozxform and setup radial interpolation of magnetic field
order = 3
bri = BoozerRadialInterpolant(equil, order, no_K=True, N=N, comm=comm) # if equil is Vmec object, mpi will be obtained from equil directly

## Setup 3d interpolation of magnetic field
"""
It takes time to set up InterpolatedBoozerField, but future field
evalutions will be much faster comparing to using BoozerRadialInterpolant.

It is important to make sure that different mpi processes are not out of sync
and stuck while waiting for data communication.
It might be easier to set up all the quantities you might need using
the initialize parameter.

You can find what quantities are needed for each RHS class in tracing.cpp.

"""
degree = 3
srange = (0, 1, ns_interp)
thetarange = (0, np.pi, ntheta_interp)
zetarange = (0, 2*np.pi/nfp, nzeta_interp)
initialize = ["psip", "G", "I", "dGds", "dIds", "iota", "modB_derivs", "modB"]
t = time.time()
field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True, initialize=initialize)

Ekin=FUSION_ALPHA_PARTICLE_ENERGY
mass=ALPHA_PARTICLE_MASS
charge=ALPHA_PARTICLE_CHARGE # Alpha particle charge
vpar0=np.sqrt(2*Ekin/mass)

NParticles = 100 # 5000 used for production runs
np.random.seed(1)
stz_inits = np.random.uniform(size=(NParticles, 3))
vpar_inits = vpar0*np.random.uniform(size=(NParticles, 1))
smin = 0.2
smax = 0.6
thetamin = 0
thetamax = np.pi
zetamin = 0
zetamax = np.pi
stz_inits[:, 0] = stz_inits[:, 0]*(smax-smin) + smin
stz_inits[:, 1] = stz_inits[:, 1]*(thetamax-thetamin) + thetamin
stz_inits[:, 2] = stz_inits[:, 2]*(zetamax-zetamin) + zetamin

## using the default adaptive RK45 solver
solver_options = {'abstol': abstol, 'reltol': reltol}

t = time.time()

## Call tracing routine
gc_tys, res_hits = trace_particles_boozer(
        field, stz_inits, vpar_inits, tmax=tmax, mass=mass, charge=charge, comm=comm,
        Ekin=Ekin, stopping_criteria=[MinToroidalFluxStoppingCriterion(1e-5),MaxToroidalFluxStoppingCriterion(1.0)],
        forget_exact_path=True, mode='gc_noK',solver_options=solver_options)

if (comm.rank == 0):
   res_lost = []
   loss_ctr = 0
   for i in range(len(res_hits)):
      if (len(res_hits[i])!=0):
         loss_ctr += 1
         res_lost.append(res_hits[i][0,:])
   print(f'Particles lost {loss_ctr}/{NParticles}={(100*loss_ctr)//NParticles:d}%')
   np.savetxt('res_hits.txt',res_lost)
