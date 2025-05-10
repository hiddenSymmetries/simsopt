#!/usr/bin/env python

"""
coil_force_objectives_scan.py
----------------------------

This script scans and compares different force and torque objective terms in stage-two coil optimization for stellarator design using SIMSOPT. It sets up a multi-objective optimization problem for magnetic coils, with selectable force/torque objectives and engineering constraints, and solves it using SciPy's L-BFGS-B optimizer. The script outputs VTK files for visualization and prints diagnostic information about the optimization process.

Main steps:
- Parse command-line arguments to select the force/torque objective and its weight.
- Define input parameters for coil geometry, penalties, and weights.
- Set up the magnetic surface and initial coil configuration.
- Construct the objective function as a weighted sum of physics and engineering terms, with the selected force/torque objective.
- Perform a Taylor test to verify gradient correctness.
- Run the optimization and save results.

Usage:
    python coil_force_objectives_scan.py <ObjectiveType> <ForceWeight> [<Threshold>]
    where <ObjectiveType> is one of: SquaredMeanForce, SquaredMeanTorque, LpCurveForce, LpCurveTorque, B2_Energy, NetFluxes
    and <ForceWeight> is the weight for the force/torque term.
    <Threshold> is optional for LpCurveForce/LpCurveTorque.

"""
import os
import sys
from pathlib import Path
import shutil
from scipy.optimize import minimize
import numpy as np
from simsopt.geo import create_equally_spaced_curves
from simsopt.geo import SurfaceRZFourier
from simsopt.field import Current, coils_via_symmetries, coils_to_vtk
from simsopt.objectives import SquaredFlux, Weight, QuadraticPenalty
from simsopt.geo import (CurveLength, CurveCurveDistance, CurveSurfaceDistance,
                         MeanSquaredCurvature, LpCurveCurvature)
from simsopt.field import BiotSavart
from simsopt.field.force import LpCurveForce, \
    SquaredMeanForce, SquaredMeanTorque, LpCurveTorque, B2_Energy, NetFluxes
from simsopt.field.selffield import regularization_circ

# --- Argument check and usage warning ---
if len(sys.argv) < 3:
    print("\nUsage: python coil_force_objectives_scan.py <ObjectiveType> <ForceWeight> [<Threshold>]")
    print("  <ObjectiveType>: SquaredMeanForce, SquaredMeanTorque, LpCurveForce, LpCurveTorque, B2_Energy, NetFluxes")
    print("  <ForceWeight>: weight for the force/torque term (float)")
    print("  <Threshold>: (optional) threshold for LpCurveForce/LpCurveTorque (float)")
    print("\nExample: python coil_force_objectives_scan.py LpCurveForce 1e-3 0.0\n")
    sys.exit(1)

###############################################################################
# INPUT PARAMETERS
###############################################################################

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 4

# Major radius for the initial circular coils:
R0 = 1.0

# Minor radius for the initial circular coils:
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5

# Weight on the curve lengths in the objective function. We use the `Weight`
# class here to later easily adjust the scalar value and rerun the optimization
# without having to rebuild the objective.
LENGTH_WEIGHT = Weight(1e-1)
LENGTH_TARGET = 15.0

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.1
CC_WEIGHT = 1e3

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.3
CS_WEIGHT = 1e2

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 5.
CURVATURE_WEIGHT = 1e-6

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 5
MSC_WEIGHT = 1e-6

# Number of iterations to perform:
MAXITER = 200

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./coil_forces_scan_" + sys.argv[1] + '_Weight' + sys.argv[2] + '/'
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

###############################################################################
# SET UP OBJECTIVE FUNCTION
###############################################################################

# Initialize the boundary magnetic surface:
nphi = 32
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

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

# Create the initial coils:
base_curves = create_equally_spaced_curves(
    ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, jax_flag=False)
base_currents = [Current(1e5) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
base_coils = coils[:ncoils]
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

a = 0.02
a_list = np.ones(len(coils)) * a
nturns = 100
curves = [c.curve for c in coils]
coils_to_vtk(coils, OUT_DIR + "coils_init", close=True)
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_full_init", extra_data=pointData)
bs.set_points(s.gamma().reshape((-1, 3)))

# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jlength = QuadraticPenalty(sum(Jls), LENGTH_TARGET)
for c in coils:
    c.regularization = regularization_circ(0.05)

if sys.argv[1] == 'SquaredMeanForce':
    Jforce = SquaredMeanForce(base_coils, coils)
elif sys.argv[1] == 'SquaredMeanTorque':
    Jforce = SquaredMeanTorque(base_coils, coils)
elif sys.argv[1] == 'LpCurveForce':
    print('If user did not specify the threshold as the third command line argument, it will be set to zero ')
    try:
        Jforce = LpCurveForce(base_coils, coils, p=2, threshold=float(sys.argv[3]))
    except:
        Jforce = LpCurveForce(base_coils, coils, p=2, threshold=0.0)
elif sys.argv[1] == 'LpCurveTorque':
    print('If user did not specify the threshold as the third command line argument, it will be set to zero ')
    try:
        Jforce = LpCurveTorque(base_coils, coils, p=2, threshold=float(sys.argv[3]))
    except:
        Jforce = LpCurveTorque(base_coils, coils, p=2, threshold=0.0)
elif sys.argv[1] == 'B2_Energy':
    Jforce = B2_Energy(coils)
elif sys.argv[1] == 'NetFluxes':
    Jforce = sum([NetFluxes(c, coils) for c in base_coils])
else:
    print('User did not input a valid Force/Torque objective. Defaulting to no force term')
    FORCE_WEIGHT = 1e-100

# Weight on the mean squared force penalty in the objective function
try:
    FORCE_WEIGHT = Weight(sys.argv[2])
except:
    FORCE_WEIGHT = Weight(1e-100)

JF = Jf \
    + LENGTH_WEIGHT * Jlength \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
    + FORCE_WEIGHT * Jforce

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs):
    """
    Wrapper for the total objective function and its gradient for use with SciPy's optimizer.

    Parameters
    ----------
    dofs : np.ndarray
        Array of degrees of freedom (optimization variables).

    Returns
    -------
    J : float
        Value of the total objective function.
    grad : np.ndarray
        Gradient of the objective function with respect to dofs.
    """
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    BdotN_over_B = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
                           ) / np.mean(bs.AbsB())
    outstr = f"J={J:.1e}, Jf={Jf.J():.1e}, ⟨B·n⟩={BdotN:.1e}, ⟨B·n⟩/⟨B⟩={BdotN_over_B:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.2f}"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    length_val = LENGTH_WEIGHT.value * Jlength.J()
    cc_val = CC_WEIGHT * Jccdist.J()
    cs_val = CS_WEIGHT * Jcsdist.J()
    forces_val = FORCE_WEIGHT.value * Jforce.J()
    valuestr = f"J={J:.2e}, Jf={Jf.J():.2e}"
    valuestr += f", LenObj={length_val:.2e}"
    valuestr += f", ccObj={cc_val:.2e}"
    valuestr += f", csObj={cs_val:.2e}"
    valuestr += f", forceObj={forces_val:.2e}"
    outstr += f", F={Jforce.J():.2e}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    print(valuestr)
    return J, grad


print("""
###############################################################################
# Perform a Taylor test
###############################################################################
""")
print("(It make take jax several minutes to compile the objective for the first evaluation.)")
f = fun
dofs = JF.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    print("err", (J1-J2)/(2*eps) - dJh)

###############################################################################
# RUN THE OPTIMIZATION
###############################################################################

dofs = JF.x
print(f"Optimization with FORCE_WEIGHT={FORCE_WEIGHT.value} and LENGTH_WEIGHT={LENGTH_WEIGHT.value}")
# print("INITIAL OPTIMIZATION")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': MAXITER}, tol=1e-12)
coils_to_vtk(coils, OUT_DIR + "coils_opt", close=True)

pointData_surf = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData_surf)
bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_full_opt", extra_data=pointData)
