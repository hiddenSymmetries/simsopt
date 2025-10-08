#!/usr/bin/env python3

"""
This example demonstrates how to use SIMSOPT to compute Poincare plots.

This script uses the NCSX coil shapes available in
``simsopt.util.zoo.get_data("ncsx")``. For an example in which coils
optimized from a simsopt stage-2 optimization are used, see the
example tracing_fieldlines_QA.py.

This example takes advantage of MPI if you launch it with multiple
processes (e.g. by mpirun -n or srun), but it also works on a single
process.
"""

import os
import logging
import sys
import numpy as np

from simsopt.configs import get_data
from simsopt.field import (InterpolatedField, SurfaceClassifier, SimsoptFieldlineIntegrator, PoincarePlotter,
                           particles_to_vtk)
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.util import in_github_actions, proc0_print, comm_world

proc0_print("Running 1_Simple/tracing_fieldlines_NCSX.py")
proc0_print("===========================================")

sys.path.append(os.path.join("..", "tests", "geo"))
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

# If we're in the CI, make the run a bit cheaper:
nfieldlines = 3 if in_github_actions else 20
n_transits = 50 if in_github_actions else 100
tmax_fl = 10000 if in_github_actions else 40000
degree = 2 if in_github_actions else 4

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)


base_curves, currents, ma, nfp, bs = get_data("ncsx")
# bs.coils includes all coils after symmetry expansion (not just the base coils).
# You can access them directly like this:
all_curves = [c.curve for c in bs.coils]

# evaluate the magnetic field on the magnetic axis
proc0_print("Mean(|B|) on axis =", np.mean(np.linalg.norm(bs.set_points(ma.gamma()).B(), axis=1)))
proc0_print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)))

# create an integrator, that performs tasks like integrating field lines.
integrator_bs = SimsoptFieldlineIntegrator(bs, comm=comm_world, nfp=nfp, R0=ma.gamma()[0, 0], stellsym=True, tmax=tmax_fl, tol=1e-9)

# create a Poincare plotter object, which can compute and plot Poincare sections
axis_RZ = ma.gamma()[0, 0:2]
poincareline_end_RZ = axis_RZ + [0.14, 0]
poincare_start_points = np.linspace(axis_RZ, poincareline_end_RZ, nfieldlines)

poincare_bs = PoincarePlotter(integrator_bs, poincare_start_points, phis=4, n_transits=n_transits, add_symmetry_planes=True)

# Integration is only performed if a plot is requested. Plot the phi=0 plane:
fig1, ax = poincare_bs.plot_poincare_single(0)
# plot all planes (computation already occured):
fig2, axs = poincare_bs.plot_poincare_all(mark_lost=False)

# Save the figures:
if comm_world is None or comm_world.rank == 0:
    fig1.savefig(OUT_DIR + 'poincare_bs_phi_0.png', dpi=150)
    fig2.savefig(OUT_DIR + 'poincare_bs_all.png', dpi=150)
    print(f"Saved poincare plots to {OUT_DIR}")

# not necessary, but output will be garbled if other threads race 
# ahead whilst proc0 plots.
if comm_world is not None:
    comm_world.Barrier()

# if you want faster integration, you can generate an InterpolatedField:
# create a surface inside of which we will keep the field
s = SurfaceRZFourier.from_nphi_ntheta(mpol=5, ntor=5, stellsym=True, nfp=nfp,
                                      range="full torus", nphi=64, ntheta=24)
# fit the surface to the magnetic axis, with a minor radius of about 0.7. Other stellarators 
# might need a different value here. Sometimes also flip_theta (lest the surface normal point inward).
s.fit_to_curve(ma, 0.70, flip_theta=False)

# Bounds for the interpolated magnetic field chosen so that the surface is
# entirely contained in it
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]
n = 20
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2*np.pi/nfp, n*2)
# exploit stellarator symmetry and only consider positive z values:
zrange = (0, np.max(zs), n//2)

sc_fieldline = SurfaceClassifier(s, h=0.03, p=2)


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
    proc0_print("Skip", sum(skip), "cells out of", len(skip), flush=True)
    return skip


#finally we can create the interpolated field, this takes a bit of time and a lot of memory:
proc0_print('Initializing InterpolatedField', flush=True)
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True, skip=skip
)
proc0_print('Done initializing InterpolatedField')


# confirm that the interpolated field is close to the original field by evaluating it 
# on the magnetic axis:
bsh.set_points(ma.gamma().reshape((-1, 3)))
bs.set_points(ma.gamma().reshape((-1, 3)))
Bh = bsh.B()
B = bs.B()
proc0_print("|B-Bh| on axis", np.sort(np.abs(B-Bh).flatten()))

# The integrator accepts any MagneticField, also our faster InterpolatedField:
integrator_bsh = SimsoptFieldlineIntegrator(bsh, comm=comm_world, nfp=nfp, R0=ma.gamma()[0, 0], stellsym=True)
# create a Poincare plotter object for the interpolated field
poincare_bsh = PoincarePlotter(integrator_bsh, poincare_start_points, phis=4, n_transits=n_transits, add_symmetry_planes=True)

# Integration is only performed if a plot is requested. Plot the phi=0 plane:
fig3, ax = poincare_bsh.plot_poincare_single(0)
# plot all planes (computation already occured):
fig4, axs = poincare_bsh.plot_poincare_all(mark_lost=False)

# Save the figures:
if comm_world is None or comm_world.rank == 0:
    fig3.savefig(OUT_DIR + 'poincare_bsh_phi_0.png', dpi=150)
    fig4.savefig(OUT_DIR + 'poincare_bsh_all.png', dpi=150)
    print(f"Saved poincare plots to {OUT_DIR}")


if comm_world is None or comm_world.rank == 0:
    curves_to_vtk(all_curves + [ma], OUT_DIR + 'coils')
    particles_to_vtk(poincare_bsh.res_tys, OUT_DIR + 'fieldlines_bsh')
    particles_to_vtk(poincare_bs.res_tys, OUT_DIR + 'fieldlines_bs')
    s.to_vtk(OUT_DIR + 'surface')
    sc_fieldline.to_vtk(OUT_DIR + 'levelset', h=0.02)

# not necessary, but output will be garbled if other threads race 
# ahead whilst proc0 plots.
if comm_world is not None:
    comm_world.Barrier()

# Because our PoincarePlotter depends on the magnetic field, changes 
# to the magnetic field will trigger a recomputation, but only
# when a plot is requested. 

# change the currents by a few percent for fun:
np.random.seed(3)  # CAREFUL: All ranks do this, and there is no sync (yet)
for coil_num, current in enumerate(currents): 
    random_jiggle = 1 + 2*(0.5-np.random.random())
    proc0_print(f'changing current of coil {coil_num} by a factor: {random_jiggle}')
    current.set('x0', current.get_value() * random_jiggle)


# The biot-savart integrator knows the field has changed, so it will
# recompute the field lines when a plot is requested.
fig_pert, axs = poincare_bs.plot_poincare_all(mark_lost=True)
if comm_world is None or comm_world.rank == 0:
    fig_pert.savefig(OUT_DIR + 'poincare_perturbed.png', dpi=150)


proc0_print("End of 1_Simple/tracing_fieldlines_NCSX.py")
proc0_print("==========================================")
