#!/usr/bin/env python3
# import matplotlib; matplotlib.use('agg')  # noqa
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule
from simsopt.geo.surfacexyztensorfourier import SurfaceRZFourier
from simsopt.field.coil import coils_via_symmetries
from simsopt.field.tracing import SurfaceClassifier, \
    particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data
from simsopt.geo.curve import curves_to_vtk
from simsopt.util.zoo import get_ncsx_data
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

print("Running 1_Simple/tracing_fieldline.py")
print("=====================================")

# check whether we're in CI, in that case we make the run a bit cheaper
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
nfieldlines = 3 if ci else 30
tmax_fl = 10000 if ci else 40000
degree = 2 if ci else 4

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

"""
This examples demonstrate how to use SIMSOPT to compute Poincare plots and
guiding center trajectories of particles
"""


curves, currents, ma = get_ncsx_data()
coils = coils_via_symmetries(curves, currents, 3, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
print("Mean(|B|) on axis =", np.mean(np.linalg.norm(bs.set_points(ma.gamma()).B(), axis=1)))
print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)))
curves_to_vtk(curves + [ma], OUT_DIR + 'coils')

mpol = 5
ntor = 5
stellsym = True
nfp = 3
phis = np.linspace(0, 1, nfp*2*ntor+1, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
s = SurfaceRZFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
s.fit_to_curve(ma, 0.70, flip_theta=False)

s.to_vtk(OUT_DIR + 'surface')
sc_fieldline = SurfaceClassifier(s, h=0.1, p=2)
sc_fieldline.to_vtk(OUT_DIR + 'levelset', h=0.02)


def trace_fieldlines(bfield, label):
    t1 = time.time()
    R0 = np.linspace(ma.gamma()[0, 0], ma.gamma()[0, 0] + 0.14, nfieldlines)
    Z0 = [ma.gamma()[0, 2] for i in range(nfieldlines)]
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-7, comm=comm,
        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
    t2 = time.time()
    print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    if comm is None or comm.rank == 0:
        particles_to_vtk(fieldlines_tys, OUT_DIR + f'fieldlines_{label}')
        plot_poincare_data(fieldlines_phi_hits, phis, OUT_DIR + f'poincare_fieldline_{label}.png', dpi=150)


# uncomment this to run tracing using the biot savart field (very slow!)
# trace_fieldlines(bs, 'bs')


n = 16
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2*np.pi/3, n*2)
zrange = (0, np.max(zs), n//2)
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=3, stellsym=True
)
# print('Error in B', bsh.estimate_error_B(1000), flush=True)
trace_fieldlines(bsh, 'bsh')
print("End of 1_Simple/tracing_fieldline.py")
print("=====================================")
