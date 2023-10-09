#!/usr/bin/env python3

"""
This examples demonstrate how to use SIMSOPT to compute guiding center
trajectories of particles in cylindrical coordinates for a vacuum field.

This example takes advantage of MPI if you launch it with multiple
processes (e.g. by mpirun -n or srun), but it also works on a single
process.
"""

import time
import os
import logging
import sys
import numpy as np

from simsopt.configs import get_ncsx_data
from simsopt.field import (BiotSavart, InterpolatedField, coils_via_symmetries, trace_particles_starting_on_curve,
                           SurfaceClassifier, LevelsetStoppingCriterion, plot_poincare_data)
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.util import in_github_actions, proc0_print, comm_world
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV

proc0_print("Running 1_Simple/tracing_particle.py")
proc0_print("====================================")

sys.path.append(os.path.join("..", "tests", "geo"))
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

# If we're in the CI, make the run a bit cheaper:
nparticles = 3 if in_github_actions else 100
degree = 2 if in_github_actions else 3

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

nfp = 3
curves, currents, ma = get_ncsx_data()
coils = coils_via_symmetries(curves, currents, nfp, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
proc0_print("Mean(|B|) on axis =", np.mean(np.linalg.norm(bs.set_points(ma.gamma()).B(), axis=1)))
proc0_print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)))
curves_to_vtk(curves + [ma], OUT_DIR + 'coils')

mpol = 5
ntor = 5
stellsym = True
s = SurfaceRZFourier.from_nphi_ntheta(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
                                      range="full torus", nphi=64, ntheta=24)


s.fit_to_curve(ma, 0.20, flip_theta=False)
sc_particle = SurfaceClassifier(s, h=0.1, p=2)
n = 16
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]

rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2*np.pi/nfp, n*2)
# exploit stellarator symmetry and only consider positive z values:
zrange = (0, np.max(zs), n//2)
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True
)


def trace_particles(bfield, label, mode='gc_vac'):
    t1 = time.time()
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    gc_tys, gc_phi_hits = trace_particles_starting_on_curve(
        ma, bfield, nparticles, tmax=1e-2, seed=1, mass=PROTON_MASS, charge=ELEMENTARY_CHARGE,
        Ekin=5000*ONE_EV, umin=-1, umax=+1, comm=comm_world,
        phis=phis, tol=1e-9,
        stopping_criteria=[LevelsetStoppingCriterion(sc_particle.dist)], mode=mode,
        forget_exact_path=True)
    t2 = time.time()
    proc0_print(f"Time for particle tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in gc_tys])//nparticles}", flush=True)
    if comm_world is None or comm_world.rank == 0:
        # particles_to_vtk(gc_tys, OUT_DIR + f'particles_{label}_{mode}')
        plot_poincare_data(gc_phi_hits, phis, OUT_DIR + f'poincare_particle_{label}_loss.png', mark_lost=True)
        plot_poincare_data(gc_phi_hits, phis, OUT_DIR + f'poincare_particle_{label}.png', mark_lost=False)


proc0_print('Error in B', bsh.estimate_error_B(1000), flush=True)
proc0_print('Error in AbsB', bsh.estimate_error_GradAbsB(1000), flush=True)
trace_particles(bsh, 'bsh', 'gc_vac')
# trace_particles(bsh, 'bsh', 'full')
# trace_particles(bs, 'bs', 'gc')

proc0_print("End of 1_Simple/tracing_particle.py")
proc0_print("====================================")
