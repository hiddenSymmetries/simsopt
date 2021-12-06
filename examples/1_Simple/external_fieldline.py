#!/usr/bin/env python3
# import matplotlib; matplotlib.use('agg')  # noqa
from pathlib import Path
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux, CoilOptObjective
from simsopt.geo.curve import RotatedCurve, curves_to_vtk, create_equally_spaced_curves
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries, ScaledCurrent
from simsopt.geo.curveobjectives import CurveLength, MinimumDistance
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule
from simsopt.geo.surfacexyztensorfourier import SurfaceRZFourier
from simsopt.field.coil import coils_via_symmetries
from simsopt.field.tracing import SurfaceClassifier, \
    particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data
from simsopt.geo.curve import curves_to_vtk
from simsopt.util.zoo import get_ncsx_data
from simsopt._core.derivative import derivative_dec
from scipy.optimize import minimize
import simsoptpp as sopp
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

import numpy as np
import time
import os
import logging
import sys
sys.path.append(os.path.join("..", "tests", "geo"))
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

print("Running 1_Simple/tracing_fieldline.py")
print("=====================================")

# check whether we're in CI, in that case we make the run a bit cheaper
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
nfieldlines = 50
tmax_fl = 800


"""
This examples demonstrate how to use SIMSOPT to compute Poincare plots and
guiding center trajectories of particles
"""


# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 4
# Major radius for the initial circular coils:
R0 = 1.0
# Minor radius for the initial circular coils:
R1 = 0.5
# Number of Fourier modes describing each Cartesian component of each coil:
order = 5
# Weight on the curve lengths in the objective function:
ALPHA = 1e-5
# Threshhold for the coil-to-coil distance penalty in the objective function:
MIN_DIST = 0.1
# Weight on the coil-to-coil distance penalty term in the objective function:
BETA = 1000

nfp = 2

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = []
for i in range(ncoils):
    curr = Current(1)
    if i == 0:
        curr.fix_all()
    base_currents.append(ScaledCurrent(curr, 1e5))

coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
bs_fl = BiotSavart(coils)
# print("Mean(|B|) on axis =", np.mean(np.linalg.norm(bs.set_points(ma.gamma()).B(), axis=1)))
# print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)))

curve = base_curves[0]
# curve.set("xc(0)", 1.9 * curve.get("xc(0)"))
curve.set("xc(1)", 1.9 * curve.get("xc(1)"))
curve.set("yc(1)", 1.9 * curve.get("yc(1)"))

curves_to_vtk(curves, '/tmp/coils')

def trace_fieldlines(bfield, label, R0=None, Z0=None):
    if R0 is None:
        R0 = np.linspace(1.2, 2.0, nfieldlines)
    if Z0 is None:
        Z0 = [0. for i in range(nfieldlines)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-7, comm=comm)
    if comm is None or comm.rank == 0:
        particles_to_vtk(fieldlines_tys, f'/tmp/fieldlines_{label}')


# trace_fieldlines(bs, 'init')

fl = CurveXYZFourier(100, 5)
fl.set("xc(0)", 2.5)
fl.set("xc(1)", 0.7)
fl.set("ys(1)", 2.0)
curves_to_vtk([fl], '/tmp/fl')

from simsopt._core.graph_optimizable import Optimizable


class FieldlineObjective(Optimizable):
    def __init__(self, curve, field):
        self.curve = curve
        self.field = field
        xyz = self.curve.gamma()
        self.field.set_points(xyz)
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[curve, field])

    def J(self):
        self.field.set_points(self.curve.gamma())
        B = self.field.B()
        l = self.curve.gammadash()
        Absl = self.curve.incremental_arclength()
        AbsB = self.field.AbsB()
        J = np.mean(0.5*np.sum((B/AbsB - l/Absl[:, None])**2, axis=1) * Absl)
        return J

    @derivative_dec
    def dJ(self):
        self.field.set_points(self.curve.gamma())
        dBdX = self.field.dB_by_dX()
        B = self.field.B()
        AbsB = self.field.AbsB()
        l = self.curve.gammadash()
        Absl = self.curve.incremental_arclength()
        npoints = Absl.size
        term1 = -((B/AbsB - l/Absl[:, None]) * (1/Absl[:, None])) * Absl[:, None]
        term2 = -np.sum(
            (B/AbsB - l/Absl[:, None]) * (-l/(Absl[:, None]**2)), axis=1) * Absl
        term3 = 0.5 * np.sum((B/AbsB - l/Absl[:, None])**2, axis=1)
        term4 = np.sum((dBdX/AbsB[:, :, None]) * (B/AbsB - l/Absl[:, None])[:, None, :], axis=2) * Absl[:, None]
        term4 -= np.sum(dBdX*B[:, None, :], axis=2) * np.sum((B/AbsB**3) * (B/AbsB - l/Absl[:, None]), axis=1)[:, None] * Absl[:, None]
        term5 = (1/AbsB) * (B/AbsB - l/Absl[:, None]) * Absl[:, None]
        term5 -= B * np.sum((B/AbsB**3) * (B/AbsB - l/Absl[:, None]), axis=1)[:, None] * Absl[:, None]

        return self.curve.dgammadash_by_dcoeff_vjp(term1/npoints) \
            + self.curve.dincremental_arclength_by_dcoeff_vjp((term2+term3)/npoints) \
            + self.curve.dgamma_by_dcoeff_vjp(term4/npoints) \
            + self.field.B_vjp(term5/npoints)

obj = FieldlineObjective(fl, bs_fl)
fllen = CurveLength(fl)
l0 = fllen.J()
def fun(dofs):
    obj.x = dofs
    J = obj.J() + 0.5 * (fllen.J()-l0)**2
    dj = obj.dJ(partials=True) + float((fllen.J()-l0))*fllen.dJ(partials=True)
    grad = dj(obj)
    return float(J), np.asarray(grad).copy()

f = fun
dofs = obj.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    print("err", (J1-J2)/(2*eps) - dJh)

for c in base_curves:
    c.fix_all()
for c in base_currents:
    c._ScaledCurrent__basecurrent.fix_all()

dofs = obj.x
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': 100, 'maxcor': 400}, tol=1e-15)
curves_to_vtk([fl], '/tmp/fl2')

nphi = 32
ntheta = 32
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jdist = MinimumDistance(curves, MIN_DIST)
JF = CoilOptObjective(Jf, Jls, ALPHA, Jdist, BETA)

def fun2(dofs):
    obj.x = dofs
    JF.x = JF.x
    J = JF.J() + 1e-2 * obj.J()
    dJ = JF.dJ(partials=True) + 1e-2 * obj.dJ(partials=True)
    grad = dJ(obj)
    cl_string = ", ".join([f"{J.J():.3f}" for J in Jls])
    mean_AbsB = np.mean(bs.AbsB())
    jf = Jf.J()
    print(f"J={J:.3e}, Jflux={jf:.3e}, sqrt(Jflux)/Mean(|B|)={np.sqrt(jf)/mean_AbsB:.3e}, CoilLengths=[{cl_string}], ||∇J||={np.linalg.norm(grad):.3e}")
    return J, grad


for c in base_curves:
    c.unfix_all()
for c in base_currents[1:]:
    c._ScaledCurrent__basecurrent.fix_all()

f = fun2
dofs = obj.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    print("err", (J1-J2)/(2*eps) - dJh)


dofs = obj.x
res = minimize(fun2, dofs, jac=True, method='L-BFGS-B', options={'maxiter': 400, 'maxcor': 400}, tol=1e-15)

curves_to_vtk(curves, '/tmp/coils_fin')
curves_to_vtk([fl], '/tmp/fl_fin')
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk("/tmp/surf_fin", extra_data=pointData)
# trace_fieldlines(bs, 'fin')

rs = np.sqrt(fl.gamma()[:, 0]**2 + fl.gamma()[:, 1]**2)
closest = np.argmin(rs)
r0 = np.sqrt(fl.gamma()[closest, 0]**2 + fl.gamma()[closest, 1]**2)
z0 = fl.gamma()[closest, 2]
R0 = list(r0 + 0.1*np.random.standard_normal(size=(20, )))
Z0 = [z0] * len(R0)
R0 += [0.95 * np.sqrt(s.gamma()[0, 0, 0]**2 + s.gamma()[0, 0, 1]**2)]
Z0 += [s.gamma()[0, 0, 2]]
trace_fieldlines(bs, 'fin', R0=R0, Z0=Z0)
