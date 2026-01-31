#!/usr/bin/env python
r"""
"""

import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, coils_to_vtk, regularization_rect
from simsopt.field.force import LpCurveForce, SquaredMeanForce, SquaredMeanTorque, LpCurveTorque
from simsopt.util import calculate_modB_on_major_radius, initialize_coils
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier
)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty

t1 = time.time()

# LandremanPaulQA
# LandremanPaulQH
config_flag = "SchuettHennebergQAnfp2"

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()

if config_flag[-2:] == 'QA':
    input_name = 'input.LandremanPaul2021_QA_reactorScale_lowres'
elif config_flag[-2:] == 'QH':
    input_name = 'input.LandremanPaul2021_QH_reactorScale_lowres'
elif config_flag == 'SchuettHennebergQAnfp2':
    input_name = 'input.schuetthenneberg_nfp2'
filename = TEST_DIR / input_name

# Initialize the boundary magnetic surface:
range_param = "half period"
nphi = 32
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi, ntheta=ntheta)
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

# initialize the coils
# Use rectangular regularization for force/torque calculations
a = 0.2
b = 0.2
regularization = regularization_rect(a, b)
ncoils_init = 2 if config_flag == 'SchuettHennebergQAnfp2' else (3 if config_flag[-2:] == 'QA' else 2)
regularizations = [regularization for _ in range(ncoils_init)]
base_curves_TF, curves_TF, coils_TF, currents_TF = initialize_coils(s, TEST_DIR, config_flag, regularizations=regularizations)
num_TF_unique_coils = len(base_curves_TF)
base_coils_TF = coils_TF[:num_TF_unique_coils]
currents_TF = np.array([coil.current.get_value() for coil in coils_TF])

# # Set up BiotSavart fields
bs_TF = BiotSavart(coils_TF)

# # Calculate average, approximate on-axis B field strength
calculate_modB_on_major_radius(bs_TF, s)

# wire cross section for the TF coils is a square 20 cm x 20 cm
# Only need this if make self forces and B2Energy nonzero in the objective!
a = 0.2
b = 0.2
nturns_TF = 200
btot = bs_TF
calculate_modB_on_major_radius(btot, s)
btot.set_points(s.gamma().reshape((-1, 3)))

if config_flag[-2:] == 'QA':
    LENGTH_WEIGHT = Weight(0.001)
    LENGTH_TARGET = 125
    LINK_WEIGHT = 1e3
    CC_THRESHOLD = 0.8
    CC_WEIGHT = 1e1
    CS_THRESHOLD = 1.5
    CS_WEIGHT = 1e2
    CURVATURE_THRESHOLD = 0.5
    MSC_THRESHOLD = 0.05
    CURVATURE_WEIGHT = 1e-2
    MSC_WEIGHT = 1e-3
elif config_flag[-2:] == 'QH':
    LENGTH_WEIGHT = Weight(0.001)
    LENGTH_TARGET = 110
    LINK_WEIGHT = 1e3
    CC_THRESHOLD = 0.8
    CC_WEIGHT = 1e1
    CS_THRESHOLD = 1.5
    CS_WEIGHT = 1e2
    CURVATURE_THRESHOLD = 0.5
    MSC_THRESHOLD = 0.05
    CURVATURE_WEIGHT = 1e-2
    MSC_WEIGHT = 1e-3
elif config_flag == 'SchuettHennebergQAnfp2':
    LENGTH_WEIGHT = Weight(0.01)
    LENGTH_TARGET = 95
    LINK_WEIGHT = 1e4
    CC_THRESHOLD = 0.8
    CC_WEIGHT = 1e2
    CS_THRESHOLD = 1.5
    CS_WEIGHT = 1e1
    CURVATURE_THRESHOLD = 0.5
    MSC_THRESHOLD = 0.05
    CURVATURE_WEIGHT = 1e-2
    MSC_WEIGHT = 1e-3
# Weight for the Coil Coil forces term
FORCE_WEIGHT = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
FORCE_WEIGHT2 = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT2 = Weight(1e-22)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
# Directory for output
OUT_DIR = ("./nodipoles_comparison/")
os.makedirs(OUT_DIR, exist_ok=True)

coils_to_vtk(
    coils_TF,
    OUT_DIR + "coils_TF_initial_" + config_flag,
    close=True,
)
# Force and Torque calculations spawn a bunch of spurious BiotSavart child objects -- erase them!
for c in (coils_TF):
    c._children = set()

# Repeat for whole B field
btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_initial_" + config_flag, extra_data=pointData)
btot.set_points(s.gamma().reshape((-1, 3)))

# Define the individual terms objective function:
Jf = SquaredFlux(s, btot)
Jls_TF = [CurveLength(c) for c in base_curves_TF]
Jlength = QuadraticPenalty(sum(Jls_TF), LENGTH_TARGET, "max")

# coil-coil and coil-plasma distances should be between all coils
Jccdist = CurveCurveDistance(curves_TF, CC_THRESHOLD, num_basecurves=len(coils_TF))
Jcsdist = CurveSurfaceDistance(curves_TF, s, CS_THRESHOLD)

# While the coil array is not moving around, they cannot
# interlink.
linkNum = LinkingNumber(curves_TF, downsample=2)

# Currently, all force terms involve all the coils
all_coils = coils_TF
all_base_coils = base_coils_TF
regularization_list = [regularization_rect(a, b) for i in range(len(base_coils_TF))]
Jforce = LpCurveForce(all_base_coils, all_coils, regularization_list, p=4, threshold=4e5 * 100, downsample=2)
Jforce2 = SquaredMeanForce(all_base_coils, all_coils, downsample=2)
Jtorque = LpCurveTorque(all_base_coils, all_coils, regularization_list, p=2, threshold=4e5 * 100, downsample=2)
Jtorque2 = SquaredMeanTorque(all_base_coils, all_coils, downsample=2)

Jcs = [LpCurveCurvature(c.curve, 2, CURVATURE_THRESHOLD) for c in base_coils_TF]
Jmscs = [MeanSquaredCurvature(c.curve) for c in base_coils_TF]

JF = Jf \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
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
    cc_val = CC_WEIGHT * Jccdist.J()
    cs_val = CS_WEIGHT * Jcsdist.J()
    link_val = LINK_WEIGHT * linkNum.J()
    forces_val = Jforce.J()
    forces_val2 = Jforce2.J()
    torques_val = Jtorque.J()
    torques_val2 = Jtorque2.J()
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
    valuestr += f", forceObj={FORCE_WEIGHT.value * forces_val:.2e}"
    valuestr += f", forceObj2={FORCE_WEIGHT2.value * forces_val2:.2e}"
    valuestr += f", torqueObj={TORQUE_WEIGHT.value * torques_val:.2e}"
    valuestr += f", torqueObj2={TORQUE_WEIGHT2.value * torques_val2:.2e}"
    outstr += f", F={forces_val:.2e}"
    outstr += f", Fnet={forces_val2:.2e}"
    outstr += f", T={torques_val:.2e}"
    outstr += f", Tnet={torques_val2:.2e}"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
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

MAXITER = 600
res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
               options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
coils_to_vtk(
    bs_TF.coils,
    OUT_DIR + "coils_TF_optimized_" + config_flag,
    close=True,
)

btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_optimized_" + config_flag, extra_data=pointData)

btot.set_points(s.gamma().reshape((-1, 3)))
calculate_modB_on_major_radius(btot, s)

t2 = time.time()
print('Total time = ', t2 - t1)
btot.save(OUT_DIR + "biot_savart_optimized_" + config_flag + ".json")
print(OUT_DIR)
