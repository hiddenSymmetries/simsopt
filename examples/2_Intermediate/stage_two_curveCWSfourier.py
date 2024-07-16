#!/usr/bin/env python
r"""
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface. The coils are bounded to stick to a particular surface. In this
particular case we consider a vacuum field, so the target is just zero.

The objective is given by

    J = (1/2) \int |B dot n|^2 ds
        + LENGTH_WEIGHT * (Maximum CurveLength THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)

if any of the weights are increased, or the thresholds are tightened, the coils
are more regular and better separated, but the target normal field may not be
achieved as well. This example demonstrates the adjustment of weights and
penalties via the use of the `Weight` class.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""


import os
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import curves_to_vtk
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveCWSFourier
)
from simsopt.util import in_github_actions

OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 4

# Number of Fourier modes describing each Cartesian component of each coil:
order = 8

# Choose a circular CWS for the initial coils. If false, the CWS will be
# extended via the normal from the plasma boundary distance_cws_plasma away:
circular_cws = False
distance_cws_plasma = 0.25

# Threshold and weight for the maximum length of each individual coil:
LENGTH_THRESHOLD = 7.0
LENGTH_WEIGHT = 0.01

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.07
CC_WEIGHT = 10

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 25
CURVATURE_WEIGHT = 1e-3

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 25
MSC_WEIGHT = 1e-6

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 400

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

#######################################################
# End of input parameters.
#######################################################

# Initialize the boundary magnetic surface:
nphi = 32
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

# Initialize the coil winding surface
if circular_cws:
    cws = SurfaceRZFourier.from_nphi_ntheta(nphi, ntheta, "half period", s.nfp, mpol=1, ntor=1)
    cws.stellsym = s.stellsym
    R = s.get_rc(0, 0)
    cws.rc[0, 1] = R
    cws.rc[1, 1] = s.get_zs(1, 0)+distance_cws_plasma
    cws.zs[1, 1] = s.get_zs(1, 0)+distance_cws_plasma
    cws.local_full_x = cws.get_dofs()
else:
    cws = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
    cws.extend_via_normal(distance_cws_plasma)

# Initialize a set of base curves that lie on top of the boundary magnetic surface:
base_curves = []
for i in range(ncoils):
    curve_cws = CurveCWSFourier(
        mpol=cws.mpol,
        ntor=cws.ntor,
        idofs=cws.x,
        quadpoints=150,
        order=order,
        nfp=cws.nfp,
        stellsym=cws.stellsym,
    )
    angle = (i+0.5)*(2*np.pi)/((2)*s.nfp*ncoils)
    curve_dofs = np.zeros(len(curve_cws.get_dofs()),)
    curve_dofs[0] = 1
    curve_dofs[2*order+2] = 0
    curve_dofs[2*order+3] = angle
    curve_cws.set_dofs(curve_dofs)
    curve_cws.fix(0)
    curve_cws.fix(2*order+2)
    base_curves.append(curve_cws)
base_currents = [Current(1)*1e5 for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
curves_to_vtk(base_curves, OUT_DIR + "base_curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
cws.to_vtk(OUT_DIR + "cws")

# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = (
    Jf
    + LENGTH_WEIGHT * sum(QuadraticPenalty(J, LENGTH_THRESHOLD, "max") for J in Jls)
    + CC_WEIGHT * Jccdist
    + CURVATURE_WEIGHT * sum(Jcs)
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
)

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J:.3e}, Jf={jf:.3e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
    print(outstr)
    return J, grad


print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
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

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
curves_to_vtk(base_curves, OUT_DIR + "base_curves_opt")
curves_to_vtk(curves, OUT_DIR + "curves_opt")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData)

# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
bs.save(OUT_DIR + "biot_savart_opt.json")
