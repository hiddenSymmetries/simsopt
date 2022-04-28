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
degree = 2 if ci else 5


"""
This examples demonstrate how to use SIMSOPT to compute Poincare plots and
guiding center trajectories of particles
"""


curves, currents, ma = get_ncsx_data(Nt_coils=10)
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
s.fit_to_curve(ma, 0.05, flip_theta=False)

sc_fieldline = SurfaceClassifier(s, h=0.1, p=2)


def skip(rs, phis, zs):
    rphiz = np.asarray([rs, phis, zs]).T.copy()
    dists = sc_fieldline.evaluate_rphiz(rphiz)
    skip = list((dists < -0.01).flatten())
    print("sum(skip) =", sum(skip), "out of ", len(skip), flush=True)
    # skip = [p < 0.5 for p in phis]
    return skip


n = 50
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]
# rrange = (np.min(rs), np.max(rs), n)
# phirange = (0, 2*np.pi/nfp, n)
# zrange = (0., np.max(zs), n)
# bsh = InterpolatedField(
#     bs, degree, rrange, phirange, zrange, True, nfp=3, stellsym=True, skip=skip
# )
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2*np.pi, 4*n)
zrange = (np.min(zs), np.max(zs), n)
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=1, stellsym=False, skip=skip
)
import time
t1 = time.time()
# bsh.estimate_error_B(100)


def compute_error_on_surface(s):
    bsh.set_points(s.gamma().reshape((-1, 3)))
    dBh = bsh.GradAbsB()
    Bh = bsh.B()
    bs.set_points(s.gamma().reshape((-1, 3)))
    dB = bs.GradAbsB()
    B = bs.B()
    logger.info("Mean(|B|) on surface   %s" % np.mean(bs.AbsB()))
    logger.info("B    errors on surface %s" % np.sort(np.abs(B-Bh).flatten()))
    logger.info("∇|B| errors on surface %s" % np.sort(np.abs(dB-dBh).flatten()))


compute_error_on_surface(s)
t2 = time.time()
print(t2-t1)
import sys; sys.exit()

bsh.to_vtk("/tmp/interp")
s.to_vtk("/tmp/surf")
sc_fieldline.to_vtk("/tmp/class")
