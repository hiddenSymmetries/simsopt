#!/usr/bin/env python

"""
Example script for the force metric in a stage-two coil optimization
"""
import os
import shutil
from pathlib import Path
from scipy.optimize import minimize
import numpy as np
from simsopt.geo import curves_to_vtk, create_equally_spaced_curves
from simsopt.geo import SurfaceRZFourier
from simsopt.field import Current, coils_via_symmetries
from simsopt.objectives import SquaredFlux, Weight, QuadraticPenalty
from simsopt.geo import (CurveLength, CurveCurveDistance, CurveSurfaceDistance, 
                         MeanSquaredCurvature, LpCurveCurvature)
from simsopt.field import BiotSavart
from simsopt.field.force import coil_force, coil_torque, coil_net_forces, coil_net_torques, LpCurveForce
from simsopt.field.selffield import regularization_circ
from simsopt.util import in_github_actions, calculate_on_axis_B


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

# Weight on the mean squared force penalty in the objective function
FORCE_WEIGHT = Weight(1e-26)

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 400

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
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
nphi = 32
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, jax_flag=True)
base_currents = [Current(1e5) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
base_coils = coils[:ncoils]
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
calculate_on_axis_B(bs, s)
bs.set_points(s.gamma().reshape((-1, 3)))

a = 0.05

def pointData_forces_torques(coils):
    contig = np.ascontiguousarray
    forces = np.zeros((len(coils), len(coils[0].curve.gamma()) + 1, 3))
    torques = np.zeros((len(coils), len(coils[0].curve.gamma()) + 1, 3))
    for i, c in enumerate(coils):
        forces[i, :-1, :] = coil_force(c, coils, regularization_circ(a))
        torques[i, :-1, :] = coil_torque(c, coils, regularization_circ(a))
    
    forces[:, -1, :] = forces[:, 0, :]
    torques[:, -1, :] = torques[:, 0, :]
    forces = forces.reshape(-1, 3)
    torques = torques.reshape(-1, 3)
    point_data = {"Pointwise_Forces": (contig(forces[:, 0]), contig(forces[:, 1]), contig(forces[:, 2])), 
                  "Pointwise_Torques": (contig(torques[:, 0]), contig(torques[:, 1]), contig(torques[:, 2]))}
    return point_data

curves = [c.curve for c in coils]
a_list = regularization_circ(a) * np.ones(len(coils))
curves_to_vtk(
    curves, OUT_DIR + "curves_init", close=True, 
    extra_point_data=pointData_forces_torques(coils),
    NetForces=coil_net_forces(coils, coils, a_list),
    NetTorques=coil_net_torques(coils, coils, a_list)
    )
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jforce = [LpCurveForce(c, coils, regularization_circ(a), p=4) for c in base_coils]

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = Jf \
    + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max") \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
    + FORCE_WEIGHT * sum(Jforce)

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize
def fun(dofs):
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
    outstr += f", F={sum(J.J() for J in Jforce):.2e}"
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
curves_to_vtk(curves, OUT_DIR + "curves_opt_short", close=True, extra_point_data=pointData_forces_torques(coils),
    NetForces=coil_net_forces(coils, coils, a_list),
    NetTorques=coil_net_torques(coils, coils, a_list)
    )

pointData_surf = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt_short", extra_data=pointData_surf)

# We now use the result from the optimization as the initial guess for a
# subsequent optimization with reduced penalty for the coil length. This will
# result in slightly longer coils but smaller `B·n` on the surface.
dofs = res.x
LENGTH_WEIGHT *= 0.1
# print("OPTIMIZATION WITH REDUCED LENGTH PENALTY\n")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
curves_to_vtk(curves, OUT_DIR + f"curves_opt_force_FWEIGHT={FORCE_WEIGHT.value:e}_LWEIGHT={LENGTH_WEIGHT.value*10:e}", close=True, 
    extra_point_data=pointData_forces_torques(coils),
    NetForces=coil_net_forces(coils, coils, a_list),
    NetTorques=coil_net_torques(coils, coils, a_list),
    )
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
force = [np.max(np.linalg.norm(coil_force(c, coils, regularization_circ(a)), axis=1)) for c in base_coils]
outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
jforce_string = ", ".join(f"{J.J():.2e}" for J in Jforce)
force_string = ", ".join(f"{f:.2e}" for f in force)
outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}], Jforce=[{jforce_string}], force=[{force_string}]"
outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
print(outstr)

calculate_on_axis_B(bs, s)
print(sum([c.get_value() for c in base_currents]))
