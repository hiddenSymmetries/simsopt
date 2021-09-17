#!/usr/bin/env python3
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule
from simsopt.geo.surfacexyztensorfourier import SurfaceRZFourier
from simsopt.field.coil import coils_via_symmetries
from simsopt.field.tracing import trace_particles_starting_on_curve, SurfaceClassifier, \
    particles_to_vtk, LevelsetStoppingCriterion, plot_poincare_data
from simsopt.geo.curve import curves_to_vtk
from simsopt.util.zoo import get_ncsx_data
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
import simsoptpp as sopp
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

import numpy as np
import time
import os
import logging
import sys
sys.path.append(os.path.join("..", "tests", "geo"))
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

# check whether we're in CI, in that case we make the run a bit cheaper
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
nparticles = 3 if ci else 100
degree = 2 if ci else 3


"""
This examples demonstrate how to use SIMSOPT to compute guiding center
trajectories of particles
"""


curves, currents, ma = get_ncsx_data()
coils = coils_via_symmetries(curves, currents, 3, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
print("Mean(|B|) on axis =", np.mean(np.linalg.norm(bs.set_points(ma.gamma()).B(), axis=1)))
print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)))
curves_to_vtk(curves + [ma], '/tmp/coils')

mpol = 5
ntor = 5
stellsym = True
nfp = 3
phis = np.linspace(0, 1, nfp*2*ntor+1, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
s = SurfaceRZFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)


s.fit_to_curve(ma, 0.20, flip_theta=False)
sc_particle = SurfaceClassifier(s, h=0.1, p=2)
n = 16
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]

rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2*np.pi/3, n*2)
zrange = (0, np.max(zs), n//2)
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=3, stellsym=True
)


def trace_particles(bfield, label, mode='gc_vac'):
    t1 = time.time()
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    gc_tys, gc_phi_hits = trace_particles_starting_on_curve(
        ma, bfield, nparticles, tmax=1e-2, seed=1, mass=PROTON_MASS, charge=ELEMENTARY_CHARGE,
        Ekin=5000*ONE_EV, umin=-1, umax=+1, comm=comm,
        phis=phis, tol=1e-9,
        stopping_criteria=[LevelsetStoppingCriterion(sc_particle.dist)], mode=mode,
        forget_exact_path=True)
    t2 = time.time()
    print(f"Time for particle tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in gc_tys])//nparticles}", flush=True)
    if comm is None or comm.rank == 0:
        # particles_to_vtk(gc_tys, f'/tmp/particles_{label}_{mode}')
        plot_poincare_data(gc_phi_hits, phis, f'/tmp/poincare_particle_{label}_loss.png', mark_lost=True)
        plot_poincare_data(gc_phi_hits, phis, f'/tmp/poincare_particle_{label}.png', mark_lost=False)


print('Error in B', bsh.estimate_error_B(1000), flush=True)
print('Error in AbsB', bsh.estimate_error_GradAbsB(1000), flush=True)
trace_particles(bsh, 'bsh', 'gc_vac')
# trace_particles(bsh, 'bsh', 'full')
# trace_particles(bs, 'bs', 'gc')
