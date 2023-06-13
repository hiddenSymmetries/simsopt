"""
This script performs a stage two coil optimization
with additional terms for torsion and binormal curvature in the cost function.
We use a multifilament approach that initializes the coils using the Frener-Serret Frame
"""

import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart
from simsopt.field import Current, Coil, apply_symmetries_to_curves, apply_symmetries_to_currents
from simsopt.geo import curves_to_vtk, create_equally_spaced_curves
from simsopt.geo import CurveLength, CurveCurveDistance, CurveSurfaceDistance
from simsopt.geo import SurfaceRZFourier
from simsopt.geo import ArclengthVariation
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo.strain_optimization_classes import create_multifilament_grid_frenet, StrainOpt
# from exportcoils import export_coils, import_coils, import_coils_fb, export_coils_fb


# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 4

# Major radius for the initial circular coils:
R0 = 1.00

# Minor radius for the initial circular coils:
R1 = 0.40

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5

# Set up the winding pack
numfilaments_n = 1  # number of filaments in normal direction
numfilaments_b = 1  # number of filaments in bi-normal direction
gapsize_n = 0.02  # gap between filaments in normal direction
gapsize_b = 0.03  # gap between filaments in bi-normal direction
rot_order = 5  # order of the Fourier expression for the rotation of the filament pack

scale = 1
width = 12

# Weight on the curve length penalty in the objective function:
LENGTH_PEN = 1e-2

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.1
CC_WEIGHT = 1e-4  # 2e-1

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.1
CS_WEIGHT = 0  # 2e-2

# weight for penalty on winding pack strain
STRAIN_WEIGHT = 1e-13  # 3e-7
# weight for arclength error.
# Arclength parametrization is crucial otherwise the formulas for torsion and curvature are wrong
ARCLENGTH_WEIGHT = 1


# Number of iterations to perform:
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
MAXITER = 500 if ci else 500

#######################################################
# End of input parameters.
#######################################################

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." /
            "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

config_str = f"{ncoils}_coils_rot_order_{rot_order}_nfn_{numfilaments_n}_nfb_{numfilaments_b}\
    _strain_weight_{STRAIN_WEIGHT}"

# Initialize the boundary magnetic surface:
nphi = 64
ntheta = 64
s = SurfaceRZFourier.from_vmec_input(
    filename, range="half period", nphi=nphi, ntheta=ntheta)

nfil = numfilaments_n * numfilaments_b
base_curves = create_equally_spaced_curves(
    ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = []
for i in range(ncoils):
    curr = Current(1.)
    # since the target field is zero, one possible solution is just to set all
    # currents to 0. to avoid the minimizer finding that solution, we fix one
    # of the currents
    if i == 0:
        curr.fix_all()
    base_currents.append(curr * (5e4/nfil))

# use sum here to concatenate lists
base_curves_finite_build = sum([
    create_multifilament_grid_frenet(c, numfilaments_n, numfilaments_b, gapsize_n,
                                     gapsize_b, rotation_order=rot_order) for c in base_curves], [])
base_currents_finite_build = sum([[c]*nfil for c in base_currents], [])

# apply stellarator and rotation symmetries
curves_fb = apply_symmetries_to_curves(base_curves_finite_build, s.nfp, True)
currents_fb = apply_symmetries_to_currents(
    base_currents_finite_build, s.nfp, True)
# also apply symmetries to the underlying base curves, as we use those in the
# curve-curve distance penalty
curves = apply_symmetries_to_curves(base_curves, s.nfp, True)

coils_fb = [Coil(c, curr) for (c, curr) in zip(curves_fb, currents_fb)]
coils_exp = [Coil(c, curr) for (c, curr) in zip(
    base_curves_finite_build, base_currents_finite_build)]
bs = BiotSavart(coils_fb)
bs.set_points(s.gamma().reshape((-1, 3)))
curves_to_vtk(curves, OUT_DIR + "curves_init")
curves_to_vtk(curves_fb, OUT_DIR + f"curves_init_fb_{config_str}")

pointData = {"B_N": np.sum(bs.B().reshape(
    (nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + f"surf_init_fb_{config_str}", extra_data=pointData)

# Define the individual terms of the objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jarc = [ArclengthVariation(c) for c in base_curves]
Jstrain_list = [StrainOpt(c, width=width, scale=scale)
                for c in base_curves_finite_build]
Jstrain = sum(QuadraticPenalty(Jstrain_list[i], Jstrain_list[i].J(
)) for i in range(len(base_curves_finite_build)))


# Assemble the objective function
JF = Jf \
    + STRAIN_WEIGHT * Jstrain \
    + LENGTH_PEN * sum(QuadraticPenalty(Jls[i], Jls[i].J()) for i in range(len(base_curves))) \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + ARCLENGTH_WEIGHT * sum(Jarc) \



def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    # cl_string = ", ".join([f"{J.J():.3f}" for J in Jls])
    # mean_AbsB = np.mean(bs.AbsB())
    # jf = Jf.J()
    # strain_string = Jstrain.J()
    # print(f"J={J:.3e}, Jflux={jf:.3e}, sqrt(Jflux)/Mean(|B|)={np.sqrt(jf)/mean_AbsB:.3e}, CoilLengths=[{cl_string}], {strain_string:.3e}, ||∇J||={np.linalg.norm(grad):.3e}")
    return 1e-4*J, 1e-4*grad


f = fun
dofs = JF.x


print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
               options={'maxiter': MAXITER, 'maxcor': 10, 'gtol': 1e-20, 'ftol': 1e-20}, tol=1e-20)
curves_to_vtk(curves_fb, OUT_DIR + f"curves_opt_fb_{config_str}")
curves_to_vtk(base_curves_finite_build, OUT_DIR +
              f"curves_opt_fb_hfp_{config_str}"+"whereisthecoil")
pointData = {"B_N": np.sum(bs.B().reshape(
    (nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + f"surf_opt_fb_{config_str}", extra_data=pointData)
