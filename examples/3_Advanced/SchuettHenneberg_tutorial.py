#!/usr/bin/env python
r"""
"""

import os
import shutil
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.field import regularization_rect
from simsopt.field.force import LpCurveForce, \
    SquaredMeanForce, \
    SquaredMeanTorque, LpCurveTorque
from simsopt.util import calculate_on_axis_B, remove_inboard_dipoles, \
    remove_interlinking_dipoles_and_TFs, initialize_coils, \
    dipole_array_optimization_function, save_coil_sets
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier, create_planar_curves_between_two_toroidal_surfaces
)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty

t1 = time.time()

# Number of Fourier modes describing each Cartesian component of each coil:
order = 0
shape_fixed = True
spatially_fixed = False
currents_fixed = False

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
input_name = 'input.henneberg_nfp2'
filename = TEST_DIR / input_name

# Initialize the boundary magnetic surface:
range_param = "half period"
nphi = 32
ntheta = 32
poff = 1.5
coff = 1.5
s = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4)
s_outer = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4)

# Make the inner and outer surfaces by extending the plasma surface
s_inner.extend_via_normal(poff)
s_outer.extend_via_normal(poff + coff)

qphi = nphi * 2
qtheta = ntheta * 2
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)

# Make high resolution, full torus version of the plasma boundary for plotting
s_plot = SurfaceRZFourier.from_vmec_input(
    filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

# initialize the coils
base_curves_TF, curves_TF, coils_TF, currents_TF = initialize_coils(s, TEST_DIR, 'SchuettHennebergQAnfp2')
num_TF_unique_coils = len(base_curves_TF)
base_coils_TF = coils_TF[:num_TF_unique_coils]
currents_TF = np.array([coil.current.get_value() for coil in coils_TF])

# Set up BiotSavart fields
bs_TF = BiotSavart(coils_TF)

# # Calculate average, approximate on-axis B field strength
print(s, bs_TF, base_curves_TF)
calculate_on_axis_B(bs_TF, s)

# wire cross section for the TF coils is a square 20 cm x 20 cm
# This cross-section is not reflected in optimization except
# to calculate self-forces, self-torques, or self-inductances.
a = 0.2
b = 0.2
nturns = 100
nturns_TF = 200

# wire cross section for the dipole coils should be more like 10 cm x 10 cm
aa = 0.1
bb = 0.1

# Create the initial dipole coils:
Nx = 4
Ny = Nx
Nz = Nx
base_curves, all_curves = create_planar_curves_between_two_toroidal_surfaces(
    s, s_inner, s_outer, Nx, Ny, Nz, order=order, coil_coil_flag=True, jax_flag=False,
)

# Remove dipoles that are on the inboard side, since this plasma is very compact.
base_curves = remove_inboard_dipoles(s, base_curves)

# Remove dipoles that are initialized interlinked with the TF coils.
base_curves = remove_interlinking_dipoles_and_TFs(base_curves, base_curves_TF)

ncoils = len(base_curves)
print('Ncoils = ', ncoils)
# Fix the dipole coil locations, shapes, and orientations, so that
# only degree of freedom for each dipole is how much current it has
for i in range(len(base_curves)):

    if shape_fixed:
        # Fix shape of each coil
        for j in range(2 * order + 1):
            base_curves[i].fix('x' + str(j))

    if spatially_fixed:
        # Fix the orientation of each coil
        base_curves[i].fix('x' + str(2 * order + 2))
        base_curves[i].fix('x' + str(2 * order + 3))
        base_curves[i].fix('x' + str(2 * order + 4))
        # Fix center points of each coil
        base_curves[i].fix('x' + str(2 * order + 5))
        base_curves[i].fix('x' + str(2 * order + 6))
        base_curves[i].fix('x' + str(2 * order + 7))

base_currents = [Current(1.0) * 1e7 for i in range(ncoils)]
if currents_fixed:
    [c.fix_all() for c in base_currents]
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
base_coils = coils[:ncoils]
bs = BiotSavart(coils)
allcoils = coils + coils_TF

# Create the total Bfield object from both the TF and dipole coils
btot = bs + bs_TF
calculate_on_axis_B(btot, s)
btot.set_points(s.gamma().reshape((-1, 3)))
bs.set_points(s.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]
currents = [c.current.get_value() for c in coils]
a_list = np.hstack((np.ones(len(coils)) * aa, np.ones(len(coils_TF)) * a))
b_list = np.hstack((np.ones(len(coils)) * bb, np.ones(len(coils_TF)) * b))
base_a_list = np.hstack((np.ones(len(base_coils)) * aa, np.ones(len(base_coils_TF)) * a))
base_b_list = np.hstack((np.ones(len(base_coils)) * bb, np.ones(len(base_coils_TF)) * b))

LENGTH_WEIGHT = Weight(0.001)
# LENGTH_WEIGHT2 = Weight(0.01)
LENGTH_TARGET = 85
LINK_WEIGHT = 1e4
CC_THRESHOLD = 1.0
CC_WEIGHT = 1e2
CS_THRESHOLD = 1.5
CS_WEIGHT = 1e1
# Weight for the Coil Coil forces term
FORCE_WEIGHT = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
FORCE_WEIGHT2 = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT2 = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
# Directory for output
OUT_DIR = ("./SchuettHenneberg_tutorial/")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# Save the TF and dipole coils separately, along with pointwise and net
# forces and torques on the coils.def
save_coil_sets(btot, OUT_DIR, "_initial", a, b, nturns_TF, aa, bb, nturns)

# Save the total Bfield errors on the plasma surface
btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_initial", extra_data=pointData)
btot.set_points(s.gamma().reshape((-1, 3)))

# Define the individual terms objective function:
Jf = SquaredFlux(s, btot)
Jls = [CurveLength(c) for c in base_curves]
Jls_TF = [CurveLength(c) for c in base_curves_TF]
Jlength = QuadraticPenalty(sum(Jls_TF), LENGTH_TARGET, "max")
Jlength2 = QuadraticPenalty(sum(Jls), LENGTH_TARGET // 10, "max")

# coil-coil and coil-plasma distances should be between all coils
Jccdist = CurveCurveDistance(curves + curves_TF, CC_THRESHOLD / 2.0, num_basecurves=len(allcoils))
Jccdist2 = CurveCurveDistance(curves_TF, CC_THRESHOLD, num_basecurves=len(coils_TF))
Jcsdist = CurveSurfaceDistance(curves + curves_TF, s, CS_THRESHOLD)

# While the coil array is not moving around, they cannot interlink.
linkNum = LinkingNumber(curves + curves_TF, downsample=2)
all_base_coils = base_coils + base_coils_TF
Jforce = sum([LpCurveForce(c, allcoils, regularization_rect(a, b), p=4, threshold=8e5 * 100, downsample=1
                           ) for i, c in enumerate(all_base_coils)])
Jforce2 = sum([SquaredMeanForce(c, allcoils, downsample=1) for c in all_base_coils])

# Errors creep in when downsample = 2
Jtorque = sum([LpCurveTorque(c, allcoils, regularization_rect(a, b), p=2, threshold=4e5 * 100, downsample=1
                             ) for i, c in enumerate(all_base_coils)])
Jtorque2 = sum([SquaredMeanTorque(c, allcoils, downsample=1) for c in all_base_coils])

CURVATURE_THRESHOLD = 0.5
MSC_THRESHOLD = 0.05
CURVATURE_WEIGHT = 1e-2
MSC_WEIGHT = 1e-1
Jcs = [LpCurveCurvature(c.curve, 2, CURVATURE_THRESHOLD) for c in base_coils_TF]
Jmscs = [MeanSquaredCurvature(c.curve) for c in base_coils_TF]

JF = Jf \
    + CC_WEIGHT * Jccdist \
    + CC_WEIGHT * Jccdist2 \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
    + LINK_WEIGHT * linkNum \
    + LENGTH_WEIGHT * Jlength

# If dipole shapes can change, penalize the total length of the dipole coils
if not shape_fixed:
    JF += LENGTH_WEIGHT * Jlength2

if FORCE_WEIGHT.value > 0.0:
    JF += FORCE_WEIGHT.value * Jforce  # \

if FORCE_WEIGHT2.value > 0.0:
    JF += FORCE_WEIGHT2.value * Jforce2  # \

if TORQUE_WEIGHT.value > 0.0:
    JF += TORQUE_WEIGHT * Jtorque

if TORQUE_WEIGHT2.value > 0.0:
    JF += TORQUE_WEIGHT2 * Jtorque2

# Define dictionary of objectives and weights to pass to dipole array
# optimization function wrapper
obj_dict = {
    "JF": JF,
    "Jf": Jf,
    "Jlength": Jlength,
    "Jlength2": Jlength2,
    "Jls": Jls,
    "Jls_TF": Jls_TF,
    "Jmscs": Jmscs,
    "Jccdist": Jccdist,
    "Jccdist2": Jccdist2,
    "Jcsdist": Jcsdist,
    "linkNum": linkNum,
    "Jforce": Jforce,
    "Jforce2": Jforce2,
    "Jtorque": Jtorque,
    "Jtorque2": Jtorque2,
    "btot": btot,
    "s": s,
    "base_curves_TF": base_curves_TF,
}
weight_dict = {
    "length_weight": LENGTH_WEIGHT.value,
    "cc_weight": CC_WEIGHT,
    "cs_weight": CS_WEIGHT,
    "link_weight": LINK_WEIGHT,
    "force_weight": FORCE_WEIGHT.value,
    "net_force_weight": FORCE_WEIGHT2.value,
    "torque_weight": TORQUE_WEIGHT.value,
    "net_torque_weight": TORQUE_WEIGHT2.value,
}

# Run the optimization
dofs = JF.x
MAXITER = 500
res = minimize(dipole_array_optimization_function, dofs, args=(obj_dict, weight_dict), jac=True, method='L-BFGS-B',
               options={'maxiter': MAXITER, 'maxcor': 1000}, tol=1e-20)

# Save the optimized dipole and TF coils
save_coil_sets(btot, OUT_DIR, "_optimized", a, b, nturns_TF, aa, bb, nturns)

# Save optimized Bnormal errors on plasma surface
btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_optimized", extra_data=pointData)
btot.set_points(s.gamma().reshape((-1, 3)))
calculate_on_axis_B(btot, s)
t2 = time.time()
print('Total time = ', t2 - t1)
