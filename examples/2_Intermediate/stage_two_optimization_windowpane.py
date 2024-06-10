#!/usr/bin/env python
r"""
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero. There are modular coils in addition to windowpane coils

The objective is given by

    J = (1/2) \int |B dot n|^2 ds
        + LENGTH_WEIGHT * (sum CurveLength)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)

if any of the weights are increased, or the thresholds are tightened, the coils
are more regular and better separated, but the target normal field may not be
achieved as well. This example demonstrates the adjustment of weights and
penalties via the use of the `Weight` class.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveCWSFourier)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 2

# Number of unique windowpane coil shapes, i.e. the number of windowpane coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils_windowpane = 4

# Major radius for the initial circular coils:
R0 = 1.0

# Minor radius for the initial circular coils:
R1 = 0.4

# Number of Fourier modes describing each Cartesian component of each coil:
order = 2

# Choose a circular CWS for the initial coils. If false, the CWS will be
# extended via the projected normal from the plasma boundary distance_cws_plasma away:
circular_cws = False
distance_cws_plasma = 0.15

# Weight on the curve lengths in the objective function. We use the `Weight`
# class here to later easily adjust the scalar value and rerun the optimization
# without having to rebuild the objective.
LENGTH_WEIGHT = Weight(1e-5)

# Minimum length for the windowpane coils:
LENGTH_WEIGHT_WINDOWPANE = 1e-1
LENGTH_THRESHOLD_WINDOWPANE_MIN = 0.3
LENGTH_THRESHOLD_WINDOWPANE_MAX = 0.6

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.1
CC_WEIGHT = 1e5

# Threshold and weight for the curvature penalty in the objective function for the modular coils:
CURVATURE_THRESHOLD = 5.
CURVATURE_THRESHOLD_WINDOWPANE = 30.
CURVATURE_WEIGHT = 1e-4

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 5
MSC_THRESHOLD_windowpane = 250
MSC_WEIGHT = 1e-4

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 600

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

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=130)
base_currents = [Current(1)*1e5 for i in range(ncoils)]

base_curves_windowpane = []
base_currents_windowpane = [Current(1)*1e4 for i in range(ncoils_windowpane)]
for i in range(ncoils_windowpane):
    # angle = i*2*np.pi/4
    windowpane_coil = CurveCWSFourier(mpol=cws.mpol, ntor=cws.ntor, idofs=cws.x, quadpoints=100, order=1, nfp=cws.nfp, stellsym=cws.stellsym)
    windowpane_coil.set_dofs(np.array([0, np.mod(i,2*np.pi), 0.1, 0.001, 0, np.mod(i,2*np.pi/2/s.nfp), 0.001, 0.1]))
    windowpane_coil.fix(0)
    windowpane_coil.fix(3)
    windowpane_coil.fix(4)
    windowpane_coil.fix(6)
    base_curves_windowpane.append(windowpane_coil)

coils_modular  = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
coils_windowpane = coils_via_symmetries(base_curves_windowpane, base_currents_windowpane, s.nfp, True)
coils = coils_modular + coils_windowpane
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

curves_modular = [c.curve for c in coils_modular]
curves_windowpane = [c.curve for c in coils_windowpane]
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init", close=True)
if ncoils_windowpane > 0:
    curves_to_vtk(base_curves_windowpane, OUT_DIR + "curves_windowpane_init", close=True)
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
cws.to_vtk(OUT_DIR + "cws")

# Define the individual terms objective function:
Jf = SquaredFlux(s, bs, definition="local")
Jls = [CurveLength(c) for c in base_curves]
Jls_windowpane = [CurveLength(c) for c in base_curves_windowpane]
Jccdist = CurveCurveDistance(curves_modular, CC_THRESHOLD, num_basecurves=ncoils)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jcs_windowpane = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD_WINDOWPANE) for c in base_curves_windowpane]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jmscs_windowpane = [MeanSquaredCurvature(c) for c in base_curves_windowpane]

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = Jf \
    + LENGTH_WEIGHT * sum(Jls) \
    + CC_WEIGHT * Jccdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
if ncoils_windowpane > 0:
    JF += LENGTH_WEIGHT_WINDOWPANE * sum(QuadraticPenalty(J, LENGTH_THRESHOLD_WINDOWPANE_MIN, "min") for J in Jls_windowpane) \
        + LENGTH_WEIGHT_WINDOWPANE * sum(QuadraticPenalty(J, LENGTH_THRESHOLD_WINDOWPANE_MAX, "max") for J in Jls_windowpane) \
        + CURVATURE_WEIGHT * sum(Jcs_windowpane) \
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD_windowpane, "max") for J in Jmscs_windowpane)

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls+Jls_windowpane])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in np.concatenate((base_curves,base_curves_windowpane)))
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs+Jmscs_windowpane)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls+Jls_windowpane):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
    outstr += f", Currents=[{', '.join([f'{c._current.get_value()/1e5:.2f}' for c in coils[0:ncoils]])}]"
    outstr += f", Currents windowpane=[{', '.join([f'{c._current.get_value()/1e4:.2f}' for c in coils_windowpane[0:ncoils_windowpane]])}]"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
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
curves_to_vtk(curves, OUT_DIR + "curves_opt", close=True)
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData)

# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
bs.save(OUT_DIR + "biot_savart_opt.json")
