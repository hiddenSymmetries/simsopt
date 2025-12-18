#!/usr/bin/env python

r"""
This example script runs joint optimization between a set of modular
toroidal field (TF) coils and a set of dipole coils (a dipole array). 
This procedure and various results are described in the following paper(s):
    A. A. Kaptanoglu et al. 
    Reactor-scale stellarators with force and torque minimized dipole coils
    https://arxiv.org/abs/2412.13937
    A. A. Kaptanoglu et al. 
    Optimization of passive superconductors for shaping stellarator magnetic fields
    https://arxiv.org/abs/2501.12468

Both of these papers have corresponding Zenodo datasets:
https://zenodo.org/records/14934093
https://zenodo.org/records/15236238

More advanced examples of dipole array optimization
can be found in examples/3_Advanced/dipole_coil_optimization/ 
and examples/3_Advanced/passive_coil_optimization/
"""

import os
import shutil
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries, regularization_rect, PSCArray
from simsopt.util import calculate_modB_on_major_radius, remove_inboard_dipoles, \
    remove_interlinking_dipoles_and_TFs, initialize_coils, in_github_actions, \
    dipole_array_optimization_function, save_coil_sets, align_dipoles_with_plasma
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier, create_planar_curves_between_two_toroidal_surfaces
)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.field.force import LpCurveForce, SquaredMeanForce, \
    SquaredMeanTorque, LpCurveTorque

t1 = time.time()

# Directory for output
OUT_DIR = ("./dipole_array_tutorial_advanced/")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

nphi = 64
ntheta = 64
MAXITER = 400

# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    MAXITER = 10
    nphi = 4
    ntheta = 4

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
input_name = 'input.schuetthenneberg_nfp2'
filename = TEST_DIR / input_name

# Whether to use active or passive coils
passive_coil_array = False

# Initialize the boundary magnetic surface:
range_param = "half period"
poff = 1.5
coff = 3.0
s = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4)
s_outer = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4)

# Make the inner and outer surfaces by extending the plasma surface
s_inner.extend_via_normal(poff)
s_outer.extend_via_normal(poff + coff)

# Make high resolution, full torus version of the plasma boundary for plotting
qphi = nphi * 2
qtheta = ntheta * 2
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)
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
calculate_modB_on_major_radius(bs_TF, s)

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

# Number of Fourier modes describing each Cartesian component of each coil:
order = 2

# Whether to fix the shapes, spatial locations/orientations, and currents of the dipole coils
shape_fixed = False
spatially_fixed = False
currents_fixed = False

# Create the initial dipole coils:
Nx = 4
Ny = Nx
Nz = Nx
base_curves, all_curves = create_planar_curves_between_two_toroidal_surfaces(
    s, s_inner, s_outer, Nx, Ny, Nz, order=order,
)

# Remove dipoles that are on the inboard side, since this plasma is very compact.
base_curves = remove_inboard_dipoles(s, base_curves)

# Remove dipoles that are initialized interlinked with the TF coils.
base_curves = remove_interlinking_dipoles_and_TFs(base_curves, base_curves_TF)

# Get the angles of the dipole coils corresponding to their normal vectors
# being aligned to point towards the nearest point on the plasma surface
alphas, deltas = align_dipoles_with_plasma(s, base_curves)

# print out total number of dipole coils remaining
ncoils = len(base_curves)
print('Ncoils = ', ncoils)

# Fix the dipole coil locations, shapes, and orientations, so that
# only degree of freedom for each dipole is how much current it has
for i in range(len(base_curves)):

    # Set curve orientations to be aligned with the plasma surface
    alpha2 = alphas[i] / 2.0
    delta2 = deltas[i] / 2.0
    calpha2 = np.cos(alpha2)
    salpha2 = np.sin(alpha2)
    cdelta2 = np.cos(delta2)
    sdelta2 = np.sin(delta2)
    # Set quaternion DOFs: q0, qi, qj, qk
    base_curves[i].set('q0', calpha2 * cdelta2)
    base_curves[i].set('qi', salpha2 * cdelta2)
    base_curves[i].set('qj', calpha2 * sdelta2)
    base_curves[i].set('qk', -salpha2 * sdelta2)

    if shape_fixed:
        # Fix shape of each coil (Fourier coefficients)
        for j in range(order + 1):
            base_curves[i].fix(f'rc({j})')
        for j in range(1, order + 1):
            base_curves[i].fix(f'rs({j})')

    if spatially_fixed:
        # Fix the orientation of each coil (quaternion components)
        base_curves[i].fix('q0')
        base_curves[i].fix('qi')
        base_curves[i].fix('qj')
        base_curves[i].fix('qk')
        # Fix center points of each coil
        base_curves[i].fix('X')
        base_curves[i].fix('Y')
        base_curves[i].fix('Z')

eval_points = s.gamma().reshape(-1, 3)
if passive_coil_array:
    # Initialize the PSCArray object
    ncoils = len(base_curves)
    a_list = np.ones(len(base_curves)) * aa
    b_list = np.ones(len(base_curves)) * aa
    psc_array = PSCArray(base_curves, coils_TF, eval_points, a_list, b_list, nfp=s.nfp, stellsym=s.stellsym)

    # Calculate average, approximate on-axis B field strength
    calculate_modB_on_major_radius(psc_array.biot_savart_TF, s)
    psc_array.biot_savart_TF.set_points(eval_points)
    btot = psc_array.biot_savart_total
    calculate_modB_on_major_radius(btot, s)
    coils = psc_array.coils
    base_coils = coils[:ncoils]
else:
    psc_array = None
    base_currents = [Current(1.0) * 1e7 for i in range(ncoils)]
    if currents_fixed:
        [c.fix_all() for c in base_currents]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    base_coils = coils[:ncoils]
    bs = BiotSavart(coils)

    # Create the total Bfield object from both the TF and dipole coils
    btot = bs + bs_TF
    calculate_modB_on_major_radius(btot, s)

allcoils = coils + coils_TF
btot.set_points(eval_points)
curves = [c.curve for c in coils]
currents = [c.current.get_value() for c in coils]

# Save the TF and dipole coils separately, along with pointwise and net
# forces and torques on the coils.def
save_coil_sets(btot, OUT_DIR, "_initial", a, b, nturns_TF, aa, bb, nturns)

# Save the total Bfield errors on the plasma surface
btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_initial", extra_data=pointData)
btot.set_points(eval_points)

# Define the objective function weights
LENGTH_WEIGHT = Weight(0.01)
LENGTH_WEIGHT2 = Weight(0.01)
LENGTH_TARGET = 85
LINK_WEIGHT = 1e4
CC_THRESHOLD = 0.8
CC_WEIGHT = 1e2
CS_THRESHOLD = 1.3
CS_WEIGHT = 1e1

# Define the individual terms objective function:
Jf = SquaredFlux(s, btot)
Jls = [CurveLength(c) for c in base_curves]
Jls_TF = [CurveLength(c) for c in base_curves_TF]
Jlength = QuadraticPenalty(sum(Jls_TF), LENGTH_TARGET, "max")
Jlength2 = QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max")

# coil-coil and coil-plasma distances should be between all coils
Jccdist = CurveCurveDistance(curves + curves_TF, CC_THRESHOLD / 2.0, num_basecurves=len(allcoils))
Jccdist2 = CurveCurveDistance(curves_TF, CC_THRESHOLD, num_basecurves=len(coils_TF))
Jcsdist = CurveSurfaceDistance(curves + curves_TF, s, CS_THRESHOLD)

# While the coil array is not moving around, they cannot interlink.
linkNum = LinkingNumber(curves + curves_TF, downsample=2)

CURVATURE_THRESHOLD = 0.5
MSC_THRESHOLD = 0.05
CURVATURE_WEIGHT = 1e-2
MSC_WEIGHT = 1e-1
Jcs = [LpCurveCurvature(c.curve, 2, CURVATURE_THRESHOLD) for c in base_coils_TF]
Jmscs = [MeanSquaredCurvature(c.curve) for c in base_coils_TF]

# Force and torque terms
all_coils = coils + coils_TF
all_base_coils = base_coils + base_coils_TF
FORCE_WEIGHT = 1e-18
FORCE_WEIGHT2 = 0.0
TORQUE_WEIGHT = 0.0
TORQUE_WEIGHT2 = 0.0
regularization_list = [regularization_rect(aa, bb) for i in range(len(base_coils))] + \
    [regularization_rect(a, b) for i in range(len(base_coils_TF))]
# Only compute the force and torque on the unique set of coils, otherwise
# you are doing too much work. Also downsample the coil quadrature points
# by a factor of 2 to save compute.
Jforce = LpCurveForce(all_base_coils, all_coils, regularization_list, downsample=2)
Jforce2 = SquaredMeanForce(all_base_coils, all_coils, downsample=2)
Jtorque = LpCurveTorque(all_base_coils, all_coils, regularization_list, downsample=2)
Jtorque2 = SquaredMeanTorque(all_base_coils, all_coils, downsample=2)

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
    JF += LENGTH_WEIGHT2 * Jlength2

if FORCE_WEIGHT > 0.0:
    JF += FORCE_WEIGHT * Jforce

if FORCE_WEIGHT2 > 0.0:
    JF += FORCE_WEIGHT2 * Jforce2

if TORQUE_WEIGHT > 0.0:
    JF += TORQUE_WEIGHT * Jtorque

if TORQUE_WEIGHT2 > 0.0:
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
    "Jcs": Jcs,
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
    "psc_array": psc_array,
}
weight_dict = {
    "length_weight": LENGTH_WEIGHT.value,
    "curvature_weight": CURVATURE_WEIGHT,
    "msc_weight": MSC_WEIGHT,
    "msc_threshold": MSC_THRESHOLD,
    "cc_weight": CC_WEIGHT,
    "cs_weight": CS_WEIGHT,
    "link_weight": LINK_WEIGHT,
    "force_weight": FORCE_WEIGHT,
    "torque_weight": TORQUE_WEIGHT,
    "net_force_weight": FORCE_WEIGHT2,
    "net_torque_weight": TORQUE_WEIGHT2,
}

# Run the optimization
dofs = JF.x
res = minimize(dipole_array_optimization_function, dofs, args=(obj_dict, weight_dict, psc_array), jac=True, method='L-BFGS-B',
               options={'maxiter': MAXITER, 'maxcor': 1000}, tol=1e-20)

if passive_coil_array:
    psc_array.recompute_currents()

# Save the optimized dipole and TF coils
save_coil_sets(btot, OUT_DIR, "_optimized", a, b, nturns_TF, aa, bb, nturns)

# Save optimized Bnormal errors on plasma surface
btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_optimized", extra_data=pointData)
btot.set_points(eval_points)
calculate_modB_on_major_radius(btot, s)
t2 = time.time()
print('Total time = ', t2 - t1)
