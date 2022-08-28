#!/usr/bin/env python
r"""
In this example we solve a FOCUS like Stage II coil optimisation problem for finite build coils.
We approximate each finite build coil using a multifilament approach. To model
the multilament pack we follow the approach of

    Optimization of finite-build stellarator coils,
    Singh, Luquant, et al.  Journal of Plasma Physics 86.4 (2020).

This means, that in addition to the degrees of freedom for the shape of the
coils, we have additional degrees of freedom for the rotation of the coil pack.

The objective is given by

    J = (1/2) ∫ |B_{BiotSavart}·n - B_{External}·n|^2 ds
        + LENGTH_PEN * Σ ½(CurveLength - L0)^2
        + DIST_PEN * PairwiseDistancePenalty

The target equilibrium is the QA configuration of

    Magnetic fields with precise quasisymmetry for plasma confinement,
    Landreman, M., & Paul, E. (2022), Physical Review Letters, 128(3), 035001.

"""

import os
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from simsopt.field import BiotSavart
from simsopt.field import Current, Coil, apply_symmetries_to_curves, apply_symmetries_to_currents
from simsopt.geo import curves_to_vtk, create_equally_spaced_curves
from simsopt.geo import create_multifilament_grid
from simsopt.geo import CurveLength, CurveCurveDistance
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 4

# Major radius for the initial circular coils:
R0 = 1.00

# Minor radius for the initial circular coils:
R1 = 0.70

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5

# Weight on the curve length penalty in the objective function:
LENGTH_PEN = 1e-2

# Threshhold and weight for the coil-to-coil distance penalty in the objective function:
DIST_MIN = 0.1
DIST_PEN = 10

# Settings for multifilament approximation.  In the following
# parameters, note that "normal" and "binormal" refer not to the
# Frenet frame but rather to the "coil centroid frame" defined by
# Singh et al., before rotation.
numfilaments_n = 2  # number of filaments in normal direction
numfilaments_b = 3  # number of filaments in bi-normal direction
gapsize_n = 0.02  # gap between filaments in normal direction
gapsize_b = 0.04  # gap between filaments in bi-normal direction
rot_order = 1  # order of the Fourier expression for the rotation of the filament pack, i.e. maximum Fourier mode number

# Number of iterations to perform:
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
MAXITER = 50 if ci else 400

#######################################################
# End of input parameters.
#######################################################

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

config_str = f"rot_order_{rot_order}_nfn_{numfilaments_n}_nfb_{numfilaments_b}"

# Initialize the boundary magnetic surface:
nphi = 32
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

nfil = numfilaments_n * numfilaments_b
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = []
for i in range(ncoils):
    curr = Current(1.)
    # since the target field is zero, one possible solution is just to set all
    # currents to 0. to avoid the minimizer finding that solution, we fix one
    # of the currents
    if i == 0:
        curr.fix_all()
    base_currents.append(curr * (1e5/nfil))

# use sum here to concatenate lists
base_curves_finite_build = sum([
    create_multifilament_grid(c, numfilaments_n, numfilaments_b, gapsize_n, gapsize_b, rotation_order=rot_order) for c in base_curves], [])
base_currents_finite_build = sum([[c]*nfil for c in base_currents], [])

# apply stellarator and rotation symmetries
curves_fb = apply_symmetries_to_curves(base_curves_finite_build, s.nfp, True)
currents_fb = apply_symmetries_to_currents(base_currents_finite_build, s.nfp, True)
# also apply symmetries to the underlying base curves, as we use those in the
# curve-curve distance penalty
curves = apply_symmetries_to_curves(base_curves, s.nfp, True)

coils_fb = [Coil(c, curr) for (c, curr) in zip(curves_fb, currents_fb)]
bs = BiotSavart(coils_fb)
bs.set_points(s.gamma().reshape((-1, 3)))

curves_to_vtk(curves, OUT_DIR + "curves_init")
curves_to_vtk(curves_fb, OUT_DIR + f"curves_init_fb_{config_str}")

pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + f"surf_init_fb_{config_str}", extra_data=pointData)

# Define the objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jdist = CurveCurveDistance(curves, DIST_MIN)

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = Jf \
    + LENGTH_PEN * sum(QuadraticPenalty(Jls[i], Jls[i].J()) for i in range(len(base_curves))) \
    + DIST_PEN * Jdist

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    cl_string = ", ".join([f"{J.J():.3f}" for J in Jls])
    mean_AbsB = np.mean(bs.AbsB())
    jf = Jf.J()
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    print(f"J={J:.3e}, Jflux={jf:.3e}, sqrt(Jflux)/Mean(|B|)={np.sqrt(jf)/mean_AbsB:.3e}, CoilLengths=[{cl_string}], [{kap_string}], ||∇J||={np.linalg.norm(grad):.3e}")
    return 1e-4*J, 1e-4*grad


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
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    print("err", (J1-J2)/(2*eps) - dJh)

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")

res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 400, 'gtol': 1e-20, 'ftol': 1e-20}, tol=1e-20)

curves_to_vtk(curves_fb, OUT_DIR + f"curves_opt_fb_{config_str}")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + f"surf_opt_fb_{config_str}", extra_data=pointData)
