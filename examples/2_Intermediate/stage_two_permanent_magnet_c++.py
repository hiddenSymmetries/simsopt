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
achieved as well.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.objectives.utilities import QuadraticPenalty
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule, DipoleField
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.geo.curveobjectives import CurveLength, MinimumDistance, \
    MeanSquaredCurvature, LpCurveCurvature
from simsopt.geo.plot import plot
from simsopt.util.permanent_magnet_optimizer import PermanentMagnetOptimizer
import time

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 2

# Major radius for the initial circular coils:
R0 = 1.0

# Minor radius for the initial circular coils:
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5

# Number of iterations to perform:
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
ci = True
nfieldlines = 20 if ci else 30
tmax_fl = 30000 if ci else 40000
degree = 2 if ci else 4

MAXITER = 50 if ci else 400

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'  # _lowres

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
print(s.rc.shape, s.zs.shape, s.quadpoints_phi, s.quadpoints_theta)
print(s.quadpoints_phi.shape, s.quadpoints_theta.shape)

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1e5) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
base_currents[0].fix_all()

# Uncomment if want to keep the coils circular
for i in range(ncoils):
    base_curves[i].fix_all()

bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
# b_target_pm = -np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the regular coil optimization objective function:
Jf = SquaredFlux(s, bs)

# Form the total objective function. In this case we only optimize
# the currents and not the shapes, so only the squared flux is needed.
JF = Jf 

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
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad


print("dofs: ", JF.dof_names)
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
# Plot the optimized results
curves_to_vtk(curves, OUT_DIR + f"curves_opt")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData)

# Basic TF coil currents now optimized, turning to 
# permanent magnet optimization now. 
pm_opt = PermanentMagnetOptimizer(
    s, coil_offset=0.1, dr=0.15,
    B_plasma_surface=bs.B().reshape((nphi, ntheta, 3))
)
max_iter_MwPGP = 1000
reg_l0 = 0.1
print('Done initializing the permanent magnet object')
# Run code in python
t1 = time.time()
_, m_history, _, dipoles = pm_opt._optimize(max_iter_MwPGP=max_iter_MwPGP, reg_l0=reg_l0, py_flag=True)
t2 = time.time()
print(0.5 * np.linalg.norm(pm_opt.A_obj @ dipoles - pm_opt.b_obj, ord=2) ** 2)
print('Python MwPGP took {0:.2e}'.format(t2 - t1), ' s')
# Run code in c++ with openmp
t1 = time.time()
_, m_history, _, dipoles = pm_opt._optimize(max_iter_MwPGP=max_iter_MwPGP, reg_l0=reg_l0, py_flag=False)
t2 = time.time()
print(0.5 * np.linalg.norm(pm_opt.A_obj @ dipoles - pm_opt.b_obj, ord=2) ** 2)
print('C++ MwPGP took {0:.2e}'.format(t2 - t1), ' s')
#b_dipole = DipoleField(pm_opt.dipole_grid, dipoles, pm_opt)
#b_dipole.set_points(s.gamma().reshape((-1, 3)))

make_plots = False
if make_plots:
    # Make plot of ATA element values
    plt.figure()
    plt.hist(np.ravel(np.abs(pm_opt.ATA)), bins=np.logspace(-20, -2, 100), log=True)
    plt.grid(True)
    plt.savefig('histogram_ATA_values.png')

    # make histogram of the dipoles, normalized by their maximum values
    plt.figure()
    plt.hist(abs(dipoles) / np.ravel(np.outer(pm_opt.m_maxima, np.ones(3))), bins=np.linspace(0, 1, 30), log=True)
    plt.savefig('m_histogram.png')
    # plt.show()
    print('Done optimizing the permanent magnets')
