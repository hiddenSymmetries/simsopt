#!/usr/bin/env python
r"""
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

The objective is given by

    J = (1/2) \int |B dot n|^2 ds
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)

if any of the weights are increased, or the thresholds are tightened, the coils
are more regular and better separated, but the target normal field may not be
achieved as well. This example demonstrates the adjustment of weights and
penalties via the use of the `Weight` class.

The target equilibrium is the QA configuration of arXiv:2108.03711.
Use --use-jax to select the optional JAX field and objective adapters; the native field is the default.
Add --fused to evaluate their sum in one compiled JAX program.
In CI the JAX path runs 10 iterations per stage; the native path runs 50.
"""

import argparse
from typing import cast
import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt._core.optimizable import Optimizable
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveSurfaceDistance)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions

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
    from simsopt_jax_adapters.field import JaxBiotSavart
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
LENGTH_WEIGHT = Weight(1e-6)

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

# Number of iterations to perform:
MAXITER = (10 if args.use_jax else 50) if in_github_actions else 400

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################

# Initialize the boundary magnetic surface:
nphi = 32
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(str(filename), range="half period", nphi=nphi, ntheta=ntheta)

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1e5) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = JaxBiotSavart(coils) if args.use_jax else BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
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
else:
    Jf = SquaredFlux(s, bs)
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]



# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = Jf \
    + LENGTH_WEIGHT * cast(Optimizable, sum(Jls)) \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)

def stage_two_problem():
    """Snapshot shared geometry and the current length weight for fused evaluation."""
    return make_stage_two_problem(bs, Jf.fixed_surface_flux_spec(), StageTwoObjectiveConfig(
        num_basecurves=ncoils,
        length_weight=float(LENGTH_WEIGHT),
        curve_curve_minimum_distance=CC_THRESHOLD,
        curve_curve_weight=CC_WEIGHT,
        curve_surface_minimum_distance=CS_THRESHOLD,
        curve_surface_weight=CS_WEIGHT,
        curvature_threshold=CURVATURE_THRESHOLD,
        curvature_weight=CURVATURE_WEIGHT,
        mean_squared_curvature_threshold=MSC_THRESHOLD,
        mean_squared_curvature_weight=MSC_WEIGHT,
    ))


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
    JF.x = dofs
    if args.fused:
        J, grad = fused_value_and_grad(dofs)
    else:
        J = JF.J()
        grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(np.asarray(bs.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad


print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
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

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
if args.fused:
    JF.x = res.x
curves_to_vtk(curves, OUT_DIR + "curves_opt_short")
pointData = {"B_N": np.sum(np.asarray(bs.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt_short", extra_data=pointData)


# We now use the result from the optimization as the initial guess for a
# subsequent optimization with reduced penalty for the coil length. This will
# result in slightly longer coils but smaller `B·n` on the surface.
dofs = res.x
LENGTH_WEIGHT *= 0.1
if args.fused:
    problem = stage_two_problem()
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
if args.fused:
    JF.x = res.x
curves_to_vtk(curves, OUT_DIR + "curves_opt_long")
pointData = {"B_N": np.sum(np.asarray(bs.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt_long", extra_data=pointData)

# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
bs.save(OUT_DIR + "biot_savart_opt.json")
