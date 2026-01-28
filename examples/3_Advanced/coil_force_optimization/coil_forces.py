#!/usr/bin/env python
"""
coil_forces.py
--------------

This script demonstrates the use of force metrics in stage-two coil optimization for stellarator design 
using SIMSOPT. It sets up a multi-objective optimization problem for magnetic coils, including 
engineering constraints and force/energy penalties, and solves it using SciPy's L-BFGS-B optimizer. 
The script outputs VTK files for visualization and prints diagnostic information about the optimization 
process. This script was used to generate the results in the paper:
    
    Hurwitz, S., Landreman, M., Huslage, P. and Kaptanoglu, A., 2025. 
    Electromagnetic coil optimization for reduced Lorentz forces.
    Nuclear Fusion, 65(5), p.056044.
    https://iopscience.iop.org/article/10.1088/1741-4326/adc9bf/meta
    
Main steps:
- Define input parameters for coil geometry, penalties, and weights.
- Set up the magnetic surface and initial coil configuration.
- Construct the objective function as a weighted sum of physics and engineering terms.
- Perform a Taylor test to verify gradient correctness.
- Run the optimization in two stages (with different length penalties).
- Save results and print summary statistics.

"""
import os
import shutil
from pathlib import Path
from scipy.optimize import minimize
import numpy as np
from simsopt.geo import create_equally_spaced_curves
from simsopt.geo import SurfaceRZFourier
from simsopt.field import Current, coils_via_symmetries, coils_to_vtk
from simsopt.objectives import SquaredFlux, Weight, QuadraticPenalty
from simsopt.geo import (CurveLength, CurveCurveDistance, CurveSurfaceDistance,
                         MeanSquaredCurvature, LpCurveCurvature)
from simsopt.field import BiotSavart
from simsopt.field.force import LpCurveForce, B2Energy
from simsopt.field.selffield import regularization_circ
from simsopt.util import in_github_actions, calculate_modB_on_major_radius


###############################################################################
# INPUT PARAMETERS
###############################################################################

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 3

# Major radius for the initial circular coils:
R0 = 1.0

# Minor radius for the initial circular coils:
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5

# Weight on the curve lengths in the objective function. We use the `Weight`
# class here to later easily adjust the scalar value and rerun the optimization
# without having to rebuild the objective.
LENGTH_WEIGHT = Weight(1e-03)
LENGTH_TARGET = 17.4

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.1
CC_WEIGHT = 1000

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.3
CS_WEIGHT = 10

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 5.
CURVATURE_WEIGHT = 1e-6

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 5
MSC_WEIGHT = 1e-6

# Weight for forces and total vacuum energy
FORCE_WEIGHT = Weight(1e-26)
B2Energy_WEIGHT = Weight(1e-10)

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 400

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./coil_forces/"
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)


###############################################################################
# SET UP OBJECTIVE FUNCTION
###############################################################################

# Initialize the boundary magnetic surface:
nphi = 32 if not in_github_actions else 8
ntheta = 32 if not in_github_actions else 8
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

# Create the initial coils:
base_curves = create_equally_spaced_curves(
    ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, use_jax_curve=False
)
base_currents = [Current(1e5) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all()

regularizations = [regularization_circ(0.05) for _ in range(ncoils)]
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym, regularizations)
base_coils = coils[:ncoils]
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
calculate_modB_on_major_radius(bs, s)
bs.set_points(s.gamma().reshape((-1, 3)))

a = 0.05
nturns = 100
curves = [c.curve for c in coils]
coils_to_vtk(coils, OUT_DIR + "coils_init", close=True)
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jforce = LpCurveForce(base_coils, coils, p=4)
J_b2energy = B2Energy(coils)

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = Jf \
    + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max") \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
    + FORCE_WEIGHT * Jforce \
    + B2Energy_WEIGHT * J_b2energy

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
    outstr += f", F={Jforce.J():.2e}"
    outstr += f", B2Energy={J_b2energy.J():.2e}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
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
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
coils_to_vtk(coils, OUT_DIR + "coils_opt_short", close=True)

pointData_surf = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt_short", extra_data=pointData_surf)

# We now use the result from the optimization as the initial guess for a
# subsequent optimization with reduced penalty for the coil length. This will
# result in slightly longer coils but smaller `B·n` on the surface.
dofs = res.x
LENGTH_WEIGHT *= 0.1
# print("OPTIMIZATION WITH REDUCED LENGTH PENALTY\n")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
coils_to_vtk(coils, OUT_DIR + "coils_opt_force", close=True)
pointData_surf = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + f"surf_opt_force_WEIGHT={FORCE_WEIGHT.value:e}_LWEIGHT={LENGTH_WEIGHT.value*10:e}", extra_data=pointData_surf)

# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
bs.save(OUT_DIR + "biot_savart_opt.json")

#Print out final important info:
JF.x = dofs
J = JF.J()
grad = JF.dJ()
jf = Jf.J()
BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
jforce_string = f"{Jforce.J():.2e}"
outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}], Jforce=[{jforce_string}]"
outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
print(outstr)

calculate_modB_on_major_radius(bs, s)
print(sum([c.get_value() for c in base_currents]))
