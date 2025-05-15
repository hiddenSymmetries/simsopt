#!/usr/bin/env python

r"""
This example script runs joint optimization between a set of modular
toroidal field (TF) coils and a set of dipole coils (a dipole array). 
It corresponds to a SIMSOPT tutorial on the wiki page. 

This script is substantially simplified to provide a simple
illustration of performing dipole array optimization in SIMSOPT. Note that
for a reactor-scale stellarator such as the one here, the optimization
should include terms to reduce the forces and torques. You will find the solution
can have intolerably large forces and torques without adding this into optimization.

File: dipole_array_tutorial.py
Author: Jake Halpern, Alan Kaptanoglu
Last Edit Date: 04/2025
Description: This script shows how to set up planar non-encircling coils tangent to an 
             axisymmetric winding surface and planar toroidal field coils that can be 
             shifted and tilted radially 
             
"""

from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.util import calculate_on_axis_B, save_coil_sets, generate_curves, in_github_actions
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    SurfaceRZFourier
)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
import os

outdir = './dipole_array_tutorial/'
os.makedirs(outdir, exist_ok=True)
nphi = 32
ntheta = 32
MAXITER = 500

# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    MAXITER = 10
    nphi = 4
    ntheta = 4

qphi = 2 * nphi
qtheta = 2 * ntheta
# load in a sample hybrid torus equilibria
filename = 'input.LandremanPaul2021_QA_reactorScale_lowres'
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
s = SurfaceRZFourier.from_vmec_input(TEST_DIR / filename, range='half period', nphi=nphi, ntheta=ntheta)
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)
s_plot = SurfaceRZFourier.from_vmec_input(
    TEST_DIR / filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)
n_plot = s_plot.unitnormal()
# create a vacuum vessel to place coils on with sufficient plasma-coil distance
VV = SurfaceRZFourier.from_vmec_input(
    TEST_DIR / filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)
VV.extend_via_projected_normal(1.75)
base_wp_curves, base_tf_curves = generate_curves(s, VV, outdir=outdir)
ncoils = len(base_wp_curves)

# wire cross section for the TF coils is a square 25 cm x 25 cm
# Only need this if make self forces and TVE nonzero in the objective!
a = 0.25
b = 0.25
nturns = 100
nturns_TF = 200

# wire cross section for the dipole coils should be at least 10 cm x 10 cm
aa = 0.1
bb = 0.1
total_current = 70000000
print('Total current = ', total_current)

ncoils_TF = len(base_tf_curves)
base_currents_TF = [(Current(total_current / ncoils_TF * 1e-7) * 1e7) for _ in range(ncoils_TF - 1)]
total_current = Current(total_current)
total_current.fix_all()
base_currents_TF += [total_current - sum(base_currents_TF)]
coils_TF = coils_via_symmetries(base_tf_curves, base_currents_TF, s.nfp, True)
base_coils_TF = coils_TF[:ncoils_TF]
curves_TF = [c.curve for c in coils_TF]
bs_TF = BiotSavart(coils_TF)

# Fix the window pane curve dofs
[c.fix_all() for c in base_wp_curves]
base_wp_currents = [Current(1.0) * 1e6 for i in range(ncoils)]
coils = coils_via_symmetries(base_wp_curves, base_wp_currents, s.nfp, True)
base_coils = coils[:ncoils]

bs = BiotSavart(coils)
btot = bs + bs_TF

calculate_on_axis_B(btot, s)
btot.set_points(s.gamma().reshape((-1, 3)))
bs.set_points(s.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]
currents = [c.current.get_value() for c in coils]
save_coil_sets(btot, outdir, "_initial", a, b, nturns_TF, aa, bb, nturns)
btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {
    "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * n_plot, axis=2
                       ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(outdir + "surf_initial", extra_data=pointData)
btot.set_points(s.gamma().reshape((-1, 3)))

# Define the individual terms objective function:
LENGTH_WEIGHT = Weight(0.01)
CC_THRESHOLD = 0.8
CC_WEIGHT = 1e2
LENGTH_TARGET = 120
Jf = SquaredFlux(s, btot)
Jls_TF = [CurveLength(c) for c in base_tf_curves]
Jlength = QuadraticPenalty(sum(Jls_TF), LENGTH_TARGET, "max")
Jccdist = CurveCurveDistance(
    curves_TF, CC_THRESHOLD,
    num_basecurves=len(coils_TF)
)

JF = Jf \
    + CC_WEIGHT * Jccdist \
    + LENGTH_WEIGHT * Jlength


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    length_val = LENGTH_WEIGHT.value * Jlength.J()
    cc_val = CC_WEIGHT * Jccdist.J()
    BdotN_over_B = np.mean(np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
                           ) / np.mean(btot.AbsB())
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩/⟨B⟩={BdotN_over_B:.1e}"
    valuestr = f"J={J:.2e}, Jf={jf:.2e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls_TF])
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls_TF):.2f}"
    valuestr += f", LenObj={length_val:.2e}"
    valuestr += f", ccObj={cc_val:.2e}"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    print(valuestr)
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

res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
               options={'maxiter': MAXITER, 'maxcor': 500}, tol=1e-10)
save_coil_sets(btot, outdir, "_optimized", a, b, nturns_TF, aa, bb, nturns)

btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {
    "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * n_plot, axis=2
                       ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(outdir + "surf_optimized", extra_data=pointData)
