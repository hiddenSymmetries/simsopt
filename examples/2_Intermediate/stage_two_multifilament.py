#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux, FOCUSObjective
from simsopt.geo.curve import RotatedCurve, curves_to_vtk
from simsopt.geo.multifilament import CurveShiftedRotated, FilamentRotation
from simsopt.field.biotsavart import BiotSavart, Current
from simsopt.geo.coilcollection import coils_via_symmetries, create_equally_spaced_curves
from simsopt.geo.curveobjectives import CurveLength, CoshCurveCurvature
from simsopt.geo.curveobjectives import MinimumDistance
from scipy.optimize import minimize
import numpy as np
from pathlib import Path
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

import os
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']

"""
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

The objective is given by

    J = \int |Bn| ds + alpha * (sum CurveLength) + beta * MininumDistancePenalty

if alpha or beta are increased, the coils are more regular and better
separated, but the target normal field may not be achieved as well.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

nfp = 2
nphi = 128
ntheta = 128
phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
thetas = np.linspace(0, 1., ntheta, endpoint=False)
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)

ncoils = 4
R0 = 1.0
R1 = 0.5
order = 6
PPP = 15
ALPHA = 1e-8
MIN_DIST = 0.1
DIST_ALPHA = 10.
BETA = 10
MAXITER = 50 if ci else 400
KAPPA_MAX = 10.
KAPPA_ALPHA = 1.
KAPPA_WEIGHT = .1

L = 1
h = 0.02
NFIL = (2*L+1)*(2*L+1)


def create_multifil(c):
    rotation = FilamentRotation(order)
    cs = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            cs.append(CurveShiftedRotated(c, i*h, j*h, rotation))
    return cs


base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=True, R0=R0, R1=R1, order=order, PPP=PPP)
base_currents = []
for i in range(ncoils):
    curr = Current(1e5/NFIL)
    # since the target field is zero, one possible solution is just to set all
    # currents to 0. to avoid the minimizer finding that solution, we fix one
    # of the currents
    if i == 0:
        curr.fix_all()
    base_currents.append(curr)

rep_curves = []
rep_currents = []

for i in range(ncoils):
    rep_curves += create_multifil(base_curves[i])
    rep_currents += NFIL * [base_currents[i]]


coils_rep = coils_via_symmetries(rep_curves, rep_currents, nfp, True)
coils = coils_via_symmetries(base_curves, base_currents, nfp, True)

curves_rep = [c.curve for c in coils_rep]
curves = [c.curve for c in coils]
curves_to_vtk(curves_rep, "/tmp/curves_init")
bs = BiotSavart(coils_rep)
bs.set_points(s.gamma().reshape((-1, 3)))

pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk("/tmp/surf_init", extra_data=pointData)


Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jdist = MinimumDistance(curves, MIN_DIST, penalty_type="cosh", alpha=DIST_ALPHA)

JF = FOCUSObjective(Jf, Jls, ALPHA, Jdist, BETA)

Jkappas = [CoshCurveCurvature(c, kappa_max=KAPPA_MAX, alpha=KAPPA_ALPHA) for c in base_curves]

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize
def fun(dofs):
    JF.x = dofs
    J = JF.J() + KAPPA_WEIGHT*sum(Jk.J() for Jk in Jkappas)
    dJ = JF.dJ()
    for Jk in Jkappas:
        dJ += KAPPA_WEIGHT * Jk.dJ()
    grad = dJ(JF)
    cl_string = ", ".join([f"{J.J():.3f}" for J in Jls])
    mean_AbsB = np.mean(bs.AbsB())
    jf = Jf.J()
    print(f"J={J:.3e}, Jflux={jf:.3e}, sqrt(Jflux)/Mean(|B|)={np.sqrt(jf)/mean_AbsB:.3e}, CoilLengths=[{cl_string}], ||∇J||={np.linalg.norm(grad):.3e}")
    return J, grad


print("Curvatures", [np.max(c.kappa()) for c in base_curves])
print("Shortest distance", Jdist.shortest_distance())

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
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 400}, tol=1e-15)
curves_to_vtk(curves_rep, "/tmp/curves_opt_3")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk("/tmp/surf_opt_3", extra_data=pointData)
print("Curvatures", [np.max(c.kappa()) for c in base_curves])
print("Shortest distance", Jdist.shortest_distance())
