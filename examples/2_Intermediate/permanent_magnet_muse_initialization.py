#!/usr/bin/env python
r"""
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os
import pickle
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.field.magneticfieldclasses import DipoleField
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, Coil, coils_via_symmetries
from simsopt.geo.curveobjectives import CurveLength, MinimumDistance, \
    MeanSquaredCurvature, LpCurveCurvature
from simsopt.geo.plot import plot
from simsopt.util.permanent_magnet_optimizer import PermanentMagnetOptimizer
import time


def read_focus_coils(filename):
    ncoils = np.loadtxt(filename, skiprows=1, max_rows=1, dtype=int)
    order = np.loadtxt(filename, skiprows=8, max_rows=1, dtype=int)
    coilcurrents = np.zeros(ncoils)
    xc = np.zeros((ncoils, order + 1))
    xs = np.zeros((ncoils, order + 1))
    yc = np.zeros((ncoils, order + 1))
    ys = np.zeros((ncoils, order + 1))
    zc = np.zeros((ncoils, order + 1))
    zs = np.zeros((ncoils, order + 1))
    for i in range(ncoils):
        coilcurrents[i] = np.loadtxt(filename, skiprows=6 + 14 * i, max_rows=1, usecols=1)
        xc[i, :] = np.loadtxt(filename, skiprows=10 + 14 * i, max_rows=1, usecols=range(order + 1))
        xs[i, :] = np.loadtxt(filename, skiprows=11 + 14 * i, max_rows=1, usecols=range(order + 1))
        yc[i, :] = np.loadtxt(filename, skiprows=12 + 14 * i, max_rows=1, usecols=range(order + 1))
        ys[i, :] = np.loadtxt(filename, skiprows=13 + 14 * i, max_rows=1, usecols=range(order + 1))
        zc[i, :] = np.loadtxt(filename, skiprows=14 + 14 * i, max_rows=1, usecols=range(order + 1))
        zs[i, :] = np.loadtxt(filename, skiprows=15 + 14 * i, max_rows=1, usecols=range(order + 1))

    # CurveXYZFourier wants data in order sin_x, cos_x, sin_y, cos_y, ...
    coil_data = np.zeros((order + 1, ncoils * 6))
    for i in range(ncoils):
        coil_data[:, i * 6 + 0] = xs[i, :]
        coil_data[:, i * 6 + 1] = xc[i, :]
        coil_data[:, i * 6 + 2] = ys[i, :]
        coil_data[:, i * 6 + 3] = yc[i, :]
        coil_data[:, i * 6 + 4] = zs[i, :]
        coil_data[:, i * 6 + 5] = zc[i, :]
    # coilcurrents = coilcurrents * 1e3  # rescale from kA to A
    base_currents = [Current(coilcurrents[i]) for i in range(ncoils)]
    ppp = 20
    coils = [CurveXYZFourier(order*ppp, order) for i in range(ncoils)]
    for ic in range(ncoils):
        dofs = coils[ic].dofs
        dofs[0][0] = coil_data[0, 6*ic + 1]
        dofs[1][0] = coil_data[0, 6*ic + 3]
        dofs[2][0] = coil_data[0, 6*ic + 5]
        for io in range(0, min(order, coil_data.shape[0]-1)):
            dofs[0][2*io+1] = coil_data[io+1, 6*ic + 0]
            dofs[0][2*io+2] = coil_data[io+1, 6*ic + 1]
            dofs[1][2*io+1] = coil_data[io+1, 6*ic + 2]
            dofs[1][2*io+2] = coil_data[io+1, 6*ic + 3]
            dofs[2][2*io+1] = coil_data[io+1, 6*ic + 4]
            dofs[2][2*io+2] = coil_data[io+1, 6*ic + 5]
        coils[ic].local_x = np.concatenate(dofs)
    return coils, base_currents, ncoils


# File for the desired TF coils
t1 = time.time()
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'muse_tf_coils.focus'
base_curves, base_currents, ncoils = read_focus_coils(filename)
t2 = time.time()
print("Done loading in MUSE coils", t2 - t1, " s")
coils = []
for i in range(ncoils):
    coils.append(Coil(base_curves[i], base_currents[i]))
print("Done loading initializing coils in SIMSOPT")

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.MUSE'

# Directory for output
nphi = 16
ntheta = 8
dr = 0.01
coff = 0.01
poff = 0.05
OUT_DIR = "muse_nphi{0:d}_ntheta{1:d}_dr{2:.2e}_coff{3:.2e}_poff{4:.2e}/".format(nphi, ntheta, dr, coff, poff)
os.makedirs(OUT_DIR, exist_ok=True)
class_filename = "PM_optimizer_muse"

# Initialize the boundary magnetic surface:
t1 = time.time()
s = SurfaceRZFourier.from_focus(filename, range="half period", nphi=nphi, ntheta=ntheta)
t2 = time.time()
print("Done loading in MUSE plasma boundary surface, t = ", t2 - t1)

t1 = time.time()
bs = BiotSavart(coils)
bspoints = np.zeros((nphi, 3))
R0 = s.get_rc(0, 0)
for i in range(nphi):
    bspoints[i] = np.array([R0 * np.cos(s.quadpoints_phi[i]), R0 * np.sin(s.quadpoints_phi[i]), 0.0]) 
bs.set_points(bspoints)
B0 = np.linalg.norm(bs.B(), axis=-1)
B0avg = np.mean(np.linalg.norm(bs.B(), axis=-1))
surface_area = s.area()
bnormalization = B0avg * surface_area
print("Bmag at R = ", R0, ", Z = 0: ", B0) 
print("toroidally averaged Bmag at R = ", R0, ", Z = 0: ", B0avg) 
bs.set_points(s.gamma().reshape((-1, 3)))
t2 = time.time()
print("Done setting up biot savart, ", t2 - t1, " s")

t1 = time.time()
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init_muse")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init_muse", extra_data=pointData)
t2 = time.time()
print("Done writing coils and initial surface to vtk, ", t2 - t1, " s")

# Save optimized BiotSavart object
biotsavart_json_str = bs.save(filename=OUT_DIR + 'BiotSavart.json')

# permanent magnet setup 
t1 = time.time()
pm_opt = PermanentMagnetOptimizer(
    s, coil_offset=coff, dr=dr, plasma_offset=poff,
    B_plasma_surface=bs.B().reshape((nphi, ntheta, 3)),
    filename=filename, FOCUS=True, out_dir=OUT_DIR,
    cylindrical_flag=True,
)
t2 = time.time()
print('Done initializing the permanent magnet object')
print('Process took t = ', t2 - t1, ' s')

# to check nothing got mistranslated
t1 = time.time()
pm_opt.m = pm_opt.m0
pm_opt.m_proxy = pm_opt.m0
b_dipole_initial = DipoleField(pm_opt)
b_dipole_initial.set_points(s.gamma().reshape((-1, 3)))
b_dipole_initial._toVTK(OUT_DIR + "Dipole_Fields_muse_initial")
pm_opt._plot_final_dipoles()
t2 = time.time()
print('Done setting up the Dipole Field class')
print('Process took t = ', t2 - t1, ' s')

# Save PM class object to file for reuse
file_out = open(OUT_DIR + class_filename + ".pickle", "wb")
pm_opt.plasma_boundary = None
pm_opt.rz_inner_surface = None
pm_opt.rz_outer_surface = None
pickle.dump(pm_opt, file_out)

plt.show()
