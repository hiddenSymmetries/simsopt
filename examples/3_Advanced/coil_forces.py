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

Use --use-jax for the optional JAX field and objective adapters, and add
--fused for one compiled objective evaluation. Native execution is the default.
JAX CI runs 10 iterations per stage; native CI runs 50.

"""
import argparse
from typing import cast
from simsopt._core.optimizable import Optimizable
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

try:
    import jax
    from simsopt_jax.runtime.host_boundary import snapshot_host_tree
    from simsopt_jax.objectives import StageTwoObjectiveConfig, fused_stage_two_objective, make_stage_two_problem
    from simsopt_jax_adapters.geo import (
        JaxCurveCurveDistance, JaxCurveLength, JaxCurveSurfaceDistance,
        JaxLpCurveCurvature, JaxMeanSquaredCurvature,
    )
    from simsopt_jax_adapters.objectives import JaxSquaredFlux
    from simsopt_jax.backend import set_backend
    from simsopt_jax_adapters.field import JaxB2Energy, JaxBiotSavart, JaxLpCurveForce
except ImportError as error:
    jax_import_error = str(error)
else:
    jax_import_error = None

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--use-jax", action="store_true", help="Use the optional JAX field and objective adapters")
parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
parser.add_argument("--fused", action="store_true", help="Compile the whole objective; requires --use-jax")
args = parser.parse_args()
if args.fused and not args.use_jax:
    parser.error("--fused requires --use-jax")
if args.use_jax:
    if jax_import_error is not None:
        parser.error(f"--use-jax requires Python >= 3.11 and jax/jaxlib >= 0.10: {jax_import_error}")
    set_backend("jax", device="gpu" if args.device == "gpu" else "cpu", intent="parity")



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
FORCE_WEIGHT = Weight(1e-2)  # (MN/m)^4 units
B2Energy_WEIGHT = Weight(1e-4)  

# Number of iterations to perform:
MAXITER = (10 if args.use_jax else 50) if in_github_actions else 400

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
nphi = 32 if not in_github_actions else 8
ntheta = 32 if not in_github_actions else 8
s = SurfaceRZFourier.from_vmec_input(str(filename), range="half period", nphi=nphi, ntheta=ntheta)

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
bs = JaxBiotSavart(coils) if args.use_jax else BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
calculate_modB_on_major_radius(bs, s)
bs.set_points(s.gamma().reshape((-1, 3)))

a = 0.05
nturns = 100
curves = [c.curve for c in coils]
coils_to_vtk(coils, OUT_DIR + "coils_init", close=True)
pointData = {"B_N": np.sum(np.asarray(bs.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the individual terms objective function:
if args.use_jax:
    Jf = JaxSquaredFlux(s, bs)
    Jls = [JaxCurveLength(c) for c in base_curves]
    Jccdist = JaxCurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
    Jcsdist = JaxCurveSurfaceDistance(curves, s, CS_THRESHOLD)
    Jcs = [JaxLpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [JaxMeanSquaredCurvature(c) for c in base_curves]
    Jforce = JaxLpCurveForce(base_coils, coils, p=4)
    J_b2energy = JaxB2Energy(coils)
else:
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
    + LENGTH_WEIGHT * QuadraticPenalty(cast(Optimizable, sum(Jls)), LENGTH_TARGET, "max") \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
    + FORCE_WEIGHT * cast(Optimizable, Jforce) \
    + B2Energy_WEIGHT * cast(Optimizable, J_b2energy)

def stage_two_problem():
    """Snapshot shared geometry and the current length weight for fused evaluation."""
    return make_stage_two_problem(bs, Jf.fixed_surface_flux_spec(), StageTwoObjectiveConfig(
        num_basecurves=ncoils,
        length_weight=float(LENGTH_WEIGHT),
        length_target=LENGTH_TARGET,
        curve_curve_minimum_distance=CC_THRESHOLD,
        curve_curve_weight=CC_WEIGHT,
        curve_surface_minimum_distance=CS_THRESHOLD,
        curve_surface_weight=CS_WEIGHT,
        curvature_threshold=CURVATURE_THRESHOLD,
        curvature_weight=CURVATURE_WEIGHT,
        mean_squared_curvature_threshold=MSC_THRESHOLD,
        mean_squared_curvature_weight=MSC_WEIGHT,
        force_weight=float(FORCE_WEIGHT),
        force_p=4,
        vacuum_energy_weight=float(B2Energy_WEIGHT),
    ), regularizations=[c.regularization for c in coils])


if args.fused:
    problem = stage_two_problem()
    value_and_grad = jax.jit(jax.value_and_grad(fused_stage_two_objective, argnums=1))
    # The fused kernel uses the field's DOF order; reporting uses the composite's.
    field_order = np.array([JF.dof_names.index(name) for name in bs.dof_names])
    objective_order = np.argsort(field_order)


def fused_value_and_grad(dofs):
    """Evaluate the frozen problem and return a host gradient in composite DOF order."""
    value, gradient = jax.device_get(value_and_grad(
        problem, jax.device_put(snapshot_host_tree(dofs[field_order])),
    ))
    return float(value), gradient[objective_order]


if args.fused:
    initial_value, initial_gradient = fused_value_and_grad(np.asarray(JF.x))
    np.testing.assert_allclose(initial_value, JF.J(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(initial_gradient, JF.dJ(), rtol=1e-11, atol=1e-13)


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
    if args.fused:
        J, grad = fused_value_and_grad(dofs)
    else:
        J = JF.J()
        grad = JF.dJ()
    BdotN = np.mean(np.abs(np.sum(np.asarray(bs.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    BdotN_over_B = np.mean(np.abs(np.sum(np.asarray(bs.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
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
dofs = cast(np.ndarray, JF.x)
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


dofs = cast(np.ndarray, JF.x)
print(f"Optimization with FORCE_WEIGHT={FORCE_WEIGHT.value} and LENGTH_WEIGHT={LENGTH_WEIGHT.value}")
# print("INITIAL OPTIMIZATION")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
if args.fused:
    JF.x = res.x
coils_to_vtk(coils, OUT_DIR + "coils_opt_short", close=True)

pointData_surf = {"B_N": np.sum(np.asarray(bs.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt_short", extra_data=pointData_surf)

# We now use the result from the optimization as the initial guess for a
# subsequent optimization with reduced penalty for the coil length. This will
# result in slightly longer coils but smaller `B·n` on the surface.
dofs = res.x
LENGTH_WEIGHT *= 0.1
if args.fused:
    problem = stage_two_problem()
# print("OPTIMIZATION WITH REDUCED LENGTH PENALTY\n")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
if args.fused:
    JF.x = res.x
coils_to_vtk(coils, OUT_DIR + "coils_opt_force", close=True)
pointData_surf = {"B_N": np.sum(np.asarray(bs.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + f"surf_opt_force_WEIGHT={FORCE_WEIGHT.value:e}_LWEIGHT={LENGTH_WEIGHT.value*10:e}", extra_data=pointData_surf)

# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
bs.save(OUT_DIR + "biot_savart_opt.json")

#Print out final important info:
JF.x = res.x if args.fused else dofs
J = JF.J()
grad = JF.dJ()
jf = Jf.J()
BdotN = np.mean(np.abs(np.sum(np.asarray(bs.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
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
