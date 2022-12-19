#!/usr/bin/env python3

"""
This example demonstrates how to use SIMSOPT to compute Poincare plots.

This script uses the NCSX coil shapes available in
``simsopt.util.zoo.get_ncsx_data()``. For an example in which coils
optimized from a simsopt stage-2 optimization are used, see the
example tracing_fieldlines_QA.py.
"""

import time
import os
import logging
import sys
import numpy as np
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

from simsopt.field import BiotSavart
from simsopt.field import InterpolatedField
from simsopt.geo import SurfaceRZFourier
from simsopt.field import coils_via_symmetries
from simsopt.field import SurfaceClassifier, \
    particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data
from simsopt.geo import curves_to_vtk
from simsopt.configs import get_ncsx_data

print("Running 1_Simple/tracing_fieldlines_NCSX.py")
print("===========================================")

sys.path.append(os.path.join("..", "tests", "geo"))
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

# check whether we're in CI, in that case we make the run a bit cheaper
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
nfieldlines = 3 if ci else 30
tmax_fl = 10000 if ci else 40000
degree = 2 if ci else 4

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)


nfp = 3
curves, currents, ma = get_ncsx_data()
coils = coils_via_symmetries(curves, currents, nfp, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
print("Mean(|B|) on axis =", np.mean(np.linalg.norm(bs.set_points(ma.gamma()).B(), axis=1)))
print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)))
curves_to_vtk(curves + [ma], OUT_DIR + 'coils')

mpol = 5
ntor = 5
stellsym = True

s = SurfaceRZFourier.from_nphi_ntheta(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
                                      range="full torus", nphi=64, ntheta=24)
s.fit_to_curve(ma, 0.70, flip_theta=False)

s.to_vtk(OUT_DIR + 'surface')
sc_fieldline = SurfaceClassifier(s, h=0.03, p=2)
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


# Bounds for the interpolated magnetic field chosen so that the surface is
# entirely contained in it
n = 20
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2*np.pi/nfp, n*2)
# exploit stellarator symmetry and only consider positive z values:
zrange = (0, np.max(zs), n//2)


def skip(rs, phis, zs):
    # The RegularGrindInterpolant3D class allows us to specify a function that
    # is used in order to figure out which cells to be skipped.  Internally,
    # the class will evaluate this function on the nodes of the regular mesh,
    # and if *all* of the eight corners are outside the domain, then the cell
    # is skipped.  Since the surface may be curved in a way that for some
    # cells, all mesh nodes are outside the surface, but the surface still
    # intersects with a cell, we need to have a bit of buffer in the signed
    # distance (essentially blowing up the surface a bit), to avoid ignoring
    # cells that shouldn't be ignored
    rphiz = np.asarray([rs, phis, zs]).T.copy()
    dists = sc_fieldline.evaluate_rphiz(rphiz)
    skip = list((dists < -0.05).flatten())
    print("Skip", sum(skip), "cells out of", len(skip), flush=True)
    return skip


print('Initializing InterpolatedField')
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True, skip=skip
)
print('Done initializing InterpolatedField')

bsh.set_points(ma.gamma().reshape((-1, 3)))
bs.set_points(ma.gamma().reshape((-1, 3)))
Bh = bsh.B()
B = bs.B()
print("|B-Bh| on axis", np.sort(np.abs(B-Bh).flatten()))

print('Beginning field line tracing')
trace_fieldlines(bsh, 'bsh')

print("End of 1_Simple/tracing_fieldlines_NCSX.py")
print("==========================================")
