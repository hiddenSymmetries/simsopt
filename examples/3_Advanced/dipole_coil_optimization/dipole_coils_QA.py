#!/usr/bin/env python
r"""
"""

import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.field import regularization_rect
from simsopt.field.force import LpCurveForce, \
    SquaredMeanForce, \
    SquaredMeanTorque, LpCurveTorque
from simsopt.util import calculate_modB_on_major_radius, remove_inboard_dipoles, \
    initialize_coils, save_coil_sets, in_github_actions
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier, create_planar_curves_between_two_toroidal_surfaces
)
from simsopt._core import Optimizable
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty

t1 = time.time()

continuation_run = False
nphi = 32
ntheta = 32
if continuation_run:
    file_suffix = "_continuation"
    MAXITER = 2000
else:
    file_suffix = ""
    MAXITER = 600

# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    MAXITER = 10
    nphi = 4
    ntheta = 4

# Directory for output
OUT_DIR = ("./dipole_coils_QA/")
os.makedirs(OUT_DIR, exist_ok=True)

# Number of Fourier modes describing each Cartesian component of each coil:
order = 0

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
input_name = 'input.LandremanPaul2021_QA_reactorScale_lowres'
filename = TEST_DIR / input_name

# Initialize the boundary magnetic surface:
range_param = "half period"
poff = 1.5
coff = 1.5
s = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4)
s_outer = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4)

# Make the inner and outer surfaces by extending the plasma surface
s_inner.extend_via_normal(poff)
s_outer.extend_via_normal(poff + coff)

qphi = nphi * 2
qtheta = ntheta * 2
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)

# Make high resolution, full torus version of the plasma boundary for plotting
s_plot = SurfaceRZFourier.from_vmec_input(
    filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

# wire cross section for the TF coils is a square 20 cm x 20 cm
# Only need this if make self forces and B2Energy nonzero in the objective!
a = 0.2
b = 0.2
nturns = 100
nturns_TF = 200

# wire cross section for the dipole coils should be more like 5 cm x 5 cm
aa = 0.05
bb = 0.05

if not continuation_run:
    # initialize the TF coils
    # Use rectangular regularization for force/torque calculations
    ncoils_TF_init = 3  # LandremanPaulQA has 3 base coils
    regularization_TF = regularization_rect(a, b)
    regularizations_TF = [regularization_TF for _ in range(ncoils_TF_init)]
    base_curves_TF, curves_TF, coils_TF, currents_TF = initialize_coils(s, TEST_DIR, "LandremanPaulQA", regularizations=regularizations_TF)
    num_TF_unique_coils = len(base_curves_TF)
    base_coils_TF = coils_TF[:num_TF_unique_coils]
    currents_TF = np.array([coil.current.get_value() for coil in coils_TF])

    # # Set up BiotSavart fields
    bs_TF = BiotSavart(coils_TF)

    # # Calculate average, approximate on-axis B field strength
    calculate_modB_on_major_radius(bs_TF, s)

    # Create the initial dipole coils:
    Nx = 6
    Ny = Nx
    Nz = Nx
    base_curves, all_curves = create_planar_curves_between_two_toroidal_surfaces(
        s, s_inner, s_outer, Nx, Ny, Nz, order=order, use_jax_curve=False,
    )
    base_curves = remove_inboard_dipoles(s, base_curves, eps=0.05)

    ncoils = len(base_curves)
    print('Ncoils = ', ncoils)
    for i in range(len(base_curves)):
        # Fix shape of each coil (Fourier coefficients)
        for j in range(order + 1):
            base_curves[i].fix(f'rc({j})')
        for j in range(1, order + 1):
            base_curves[i].fix(f'rs({j})')
        # Fix center points of each coil
        # base_curves[i].fix('X')
        # base_curves[i].fix('Y')
        # base_curves[i].fix('Z')

    base_currents = [Current(1.0) * 2e7 for i in range(ncoils)]
    # Fix currents in each coil
    # for i in range(ncoils):
    #     base_currents[i].fix_all()

    # Use rectangular regularization for force/torque calculations
    regularization = regularization_rect(aa, bb)
    regularizations = [regularization for _ in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True, regularizations=regularizations)
    base_coils = coils[:ncoils]

    bs = BiotSavart(coils)
    btot = bs + bs_TF
else:
    btot = Optimizable.from_file("QA_dipole_array/biot_savart_optimized.json")
    bs = btot.Bfields[0]
    bs_TF = btot.Bfields[1]
    coils = bs.coils
    currents = [c.current.get_value() for c in coils]
    base_coils = coils[:len(coils) // 4]
    coils_TF = bs_TF.coils
    base_coils_TF = coils_TF[:len(coils_TF) // 4]
    curves = [c.curve for c in coils]
    base_curves = curves[:len(curves) // 4]
    curves_TF = [c.curve for c in coils_TF]
    currents_TF = [c.current.get_value() for c in coils_TF]
    base_curves_TF = curves_TF[:len(curves_TF) // 4]
    ncoils = len(curves)

    btot.set_points(s.gamma().reshape((-1, 3)))
    bs.set_points(s.gamma().reshape((-1, 3)))
    bs_TF.set_points(s.gamma().reshape((-1, 3)))

    # bs = BiotSavart(coils)  # + coils_TF)
    # btot = bs + bs_TF
    # calculate_modB_on_major_radius(btot, s)
    # btot.set_points(s.gamma().reshape((-1, 3)))
    # bs.set_points(s.gamma().reshape((-1, 3)))
    curves = [c.curve for c in coils]
    currents = [c.current.get_value() for c in coils]

calculate_modB_on_major_radius(btot, s)
btot.set_points(s.gamma().reshape((-1, 3)))
bs.set_points(s.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]
currents = [c.current.get_value() for c in coils]
a_list = np.hstack((np.ones(len(coils)) * aa, np.ones(len(coils_TF)) * a))
b_list = np.hstack((np.ones(len(coils)) * bb, np.ones(len(coils_TF)) * b))
base_a_list = np.hstack((np.ones(len(base_coils)) * aa, np.ones(len(base_coils_TF)) * a))
base_b_list = np.hstack((np.ones(len(base_coils)) * bb, np.ones(len(base_coils_TF)) * b))

LENGTH_WEIGHT = Weight(0.01)
CC_THRESHOLD = 0.8
CC_WEIGHT = 1e2
CS_THRESHOLD = 1.5
CS_WEIGHT = 1e2
if continuation_run:
    LENGTH_TARGET = 115
    LINK_WEIGHT = 1e4
else:
    LENGTH_TARGET = 100
    LINK_WEIGHT = 1e3

# Weight for the Coil Coil forces term
FORCE_WEIGHT = Weight(1e-34)  # 1e-34 Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
FORCE_WEIGHT2 = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT2 = Weight(1e-23)  # 1e-22 Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons

save_coil_sets(btot, OUT_DIR, "_initial" + file_suffix)
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_initial" + file_suffix, extra_data=pointData)
btot.set_points(s.gamma().reshape((-1, 3)))

# Define the individual terms objective function:
Jf = SquaredFlux(s, btot)
# Separate length penalties on the dipole coils and the TF coils
# since they have very different sizes
# Jls = [CurveLength(c) for c in base_curves]
Jls_TF = [CurveLength(c) for c in base_curves_TF]
Jlength = QuadraticPenalty(sum(Jls_TF), LENGTH_TARGET, "max")

# coil-coil and coil-plasma distances should be between all coils
Jccdist = CurveCurveDistance(curves + curves_TF, CC_THRESHOLD / 2.0, num_basecurves=len(coils + coils_TF))
Jccdist2 = CurveCurveDistance(curves_TF, CC_THRESHOLD, num_basecurves=len(coils_TF))
Jcsdist = CurveSurfaceDistance(curves + curves_TF, s, CS_THRESHOLD)

# While the coil array is not moving around, they cannot
# interlink.
linkNum = LinkingNumber(curves + curves_TF, downsample=2)

# Currently, all force terms involve all the coils
all_coils = coils + coils_TF
all_base_coils = base_coils + base_coils_TF
regularization_list = [regularization_rect(aa, bb) for i in range(len(base_coils))] + \
    [regularization_rect(a, b) for i in range(len(base_coils_TF))]
Jforce = LpCurveForce(all_base_coils, all_coils, regularization_list, p=4, threshold=4e5 * 100, downsample=2)
Jforce2 = SquaredMeanForce(all_base_coils, all_coils, downsample=1)
Jtorque = LpCurveTorque(all_base_coils, all_coils, regularization_list, p=2, threshold=4e5 * 100, downsample=2)
Jtorque2 = SquaredMeanTorque(all_base_coils, all_coils, downsample=1)

CURVATURE_THRESHOLD = 0.5
MSC_THRESHOLD = 0.05
if continuation_run:
    CURVATURE_WEIGHT = 1e-5
    MSC_WEIGHT = 1e-6
else:
    CURVATURE_WEIGHT = 1e-4
    MSC_WEIGHT = 1e-5
Jcs = [LpCurveCurvature(c.curve, 2, CURVATURE_THRESHOLD) for c in base_coils_TF]
Jmscs = [MeanSquaredCurvature(c.curve) for c in base_coils_TF]

JF = Jf \
    + CC_WEIGHT * Jccdist \
    + CC_WEIGHT * Jccdist2 \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + LINK_WEIGHT * linkNum \
    + LENGTH_WEIGHT * Jlength

if FORCE_WEIGHT.value > 0.0:
    JF += FORCE_WEIGHT.value * Jforce  # \

if FORCE_WEIGHT2.value > 0.0:
    JF += FORCE_WEIGHT2.value * Jforce2  # \

if TORQUE_WEIGHT.value > 0.0:
    JF += TORQUE_WEIGHT * Jtorque

if TORQUE_WEIGHT2.value > 0.0:
    JF += TORQUE_WEIGHT2 * Jtorque2


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    length_val = LENGTH_WEIGHT.value * Jlength.J()
    cc_val = CC_WEIGHT * (Jccdist.J() + Jccdist2.J())
    cs_val = CS_WEIGHT * Jcsdist.J()
    link_val = LINK_WEIGHT * linkNum.J()
    forces_val = FORCE_WEIGHT.value * Jforce.J()
    forces_val2 = FORCE_WEIGHT2.value * Jforce2.J()
    torques_val = TORQUE_WEIGHT.value * Jtorque.J()
    torques_val2 = TORQUE_WEIGHT2.value * Jtorque2.J()
    BdotN = np.mean(np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    BdotN_over_B = np.mean(np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
                           ) / np.mean(btot.AbsB())
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}, ⟨B·n⟩/⟨B⟩={BdotN_over_B:.1e}"
    valuestr = f"J={J:.2e}, Jf={jf:.2e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls_TF])
    kap_string = ", ".join(f"{np.max(c.kappa()):.2f}" for c in base_curves_TF)
    msc_string = ", ".join(f"{J.J():.2f}" for J in Jmscs)
    outstr += f", ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls_TF):.2f}"
    valuestr += f", LenObj={length_val:.2e}"
    valuestr += f", ccObj={cc_val:.2e}"
    valuestr += f", csObj={cs_val:.2e}"
    valuestr += f", Lk1Obj={link_val:.2e}"
    valuestr += f", forceObj={forces_val:.2e}"
    valuestr += f", forceObj2={forces_val2:.2e}"
    valuestr += f", torqueObj={torques_val:.2e}"
    valuestr += f", torqueObj2={torques_val2:.2e}"
    outstr += f", F={Jforce.J():.2e}"
    outstr += f", Fnet={Jforce2.J():.2e}"
    outstr += f", T={Jtorque.J():.2e}"
    outstr += f", Tnet={Jtorque2.J():.2e}"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-C-Sep2={Jccdist2.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f", Link Number = {linkNum.J()}"
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
    t1 = time.time()
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    t2 = time.time()
    print("err", (J1-J2)/(2*eps) - dJh)

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")

res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
               options={'maxiter': MAXITER, 'maxcor': 500}, tol=1e-10)
save_coil_sets(btot, OUT_DIR, "_optimized" + file_suffix)
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_optimized" + file_suffix, extra_data=pointData)

btot.set_points(s.gamma().reshape((-1, 3)))
calculate_modB_on_major_radius(btot, s)

t2 = time.time()
print('Total time = ', t2 - t1)
btot.save(OUT_DIR + "biot_savart_optimized" + file_suffix + ".json")
print(OUT_DIR)
