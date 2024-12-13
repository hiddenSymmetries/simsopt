#!/usr/bin/env python
r"""
"""

import os
import shutil
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.field import regularization_rect
from simsopt.field.force import MeanSquaredForce, coil_force, coil_torque, coil_net_torques, coil_net_forces, LpCurveForce, \
    SquaredMeanForce, \
    MeanSquaredTorque, SquaredMeanTorque, LpCurveTorque, MixedSquaredMeanForce, MixedLpCurveForce
from simsopt.util import calculate_on_axis_B
from simsopt.geo import (
    CurveLength, CurveCurveDistance, create_equally_spaced_curves,
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier, curves_to_vtk, create_equally_spaced_planar_curves,
    create_planar_curves_between_two_toroidal_surfaces
)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions
from simsopt._core import Optimizable

t1 = time.time()

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
input_name = 'input.LandremanPaul2021_QH_reactorScale_lowres'
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

# wire cross section for the TF coils is a square 20 cm x 20 cm
# Only need this if make self forces and TVE nonzero in the objective!
a = 0.2
b = 0.2
nturns = 100
nturns_TF = 200

# wire cross section for the dipole coils should be more like 5 cm x 5 cm
aa = 0.05
bb = 0.05


def pointData_forces_torques(coils, allcoils, aprimes, bprimes, nturns_list):
    contig = np.ascontiguousarray
    forces = np.zeros((len(coils), len(coils[0].curve.gamma()) + 1, 3))
    torques = np.zeros((len(coils), len(coils[0].curve.gamma()) + 1, 3))
    for i, c in enumerate(coils):
        aprime = aprimes[i]
        bprime = bprimes[i]
        forces[i, :-1, :] = coil_force(c, allcoils, regularization_rect(aprime, bprime), nturns_list[i])
        torques[i, :-1, :] = coil_torque(c, allcoils, regularization_rect(aprime, bprime), nturns_list[i])

    forces[:, -1, :] = forces[:, 0, :]
    torques[:, -1, :] = torques[:, 0, :]
    forces = forces.reshape(-1, 3)
    torques = torques.reshape(-1, 3)
    point_data = {"Pointwise_Forces": (contig(forces[:, 0]), contig(forces[:, 1]), contig(forces[:, 2])),
                  "Pointwise_Torques": (contig(torques[:, 0]), contig(torques[:, 1]), contig(torques[:, 2]))}
    return point_data


# btot = Optimizable.from_file("QH_minimal_TForder4_n25_p1.75e+00_c2.50e+00_lw1.00e-02_lt1.00e+02_lkw1.00e+04_cct8.00e-01_ccw1.00e+01_cst1.50e+00_csw1.00e+02_fw1.00e-33_fww0.000000e+00_tw0.00e+00_tww1.000000e-23/biot_savart_optimized_QH.json")
# btot = Optimizable.from_file("QH_minimal_TForder4_n25_p1.75e+00_c2.50e+00_lw1.00e-02_lt9.00e+01_lkw1.00e+04_cct8.00e-01_ccw1.00e+02_cst1.50e+00_csw1.00e+02_fw1.00e-35_fww0.000000e+00_tw0.00e+00_tww1.000000e-23/"
# btot = Optimizable.from_file("QH_minimal_TForder4_n25_p1.75e+00_c2.50e+00_lw1.00e-02_lt9.00e+01_lkw1.00e+04_cct8.00e-01_ccw1.00e+02_cst1.50e+00_csw1.00e+02_fw1.00e-34_fww0.000000e+00_tw0.00e+00_tww1.000000e-22/biot_savart_optimized_QH.json")
# btot = Optimizable.from_file("QH_minimal_TForder4_n27_p1.50e+00_c2.50e+00_lw1.00e-02_lt1.00e+02_lkw1.00e+04_cct8.00e-01_ccw1.00e+01_cst1.50e+00_csw1.00e+02_fw1.00e-34_fww0.000000e+00_tw0.00e+00_tww1.000000e-22/biot_savart_optimized_QH.json")
btot = Optimizable.from_file("QH_minimal_TForder4_n27_p1.50e+00_c2.50e+00_lw1.00e-02_lt8.00e+01_lkw1.00e+04_cct8.00e-01_ccw1.00e+01_cst1.50e+00_csw1.00e+02_fw1.00e-36_fww0.000000e+00_tw0.00e+00_tww1.000000e-24/biot_savart_optimized_QH.json")
bs = btot.Bfields[0]
bs_TF = btot.Bfields[1]
coils = bs.coils
currents = [c.current.get_value() for c in coils]
base_coils = coils[:len(coils) // 8]
coils_TF = bs_TF.coils
base_coils_TF = coils_TF[:len(coils_TF) // 8]
curves = [c.curve for c in coils]
base_curves = curves[:len(curves) // 8]
curves_TF = [c.curve for c in coils_TF]
currents_TF = [c.current.get_value() for c in coils_TF]
base_curves_TF = curves_TF[:len(curves_TF) // 8]
ncoils = len(curves)

btot.set_points(s.gamma().reshape((-1, 3)))
bs.set_points(s.gamma().reshape((-1, 3)))
bs_TF.set_points(s.gamma().reshape((-1, 3)))

a_list = np.hstack((np.ones(len(coils)) * aa, np.ones(len(coils_TF)) * a))
b_list = np.hstack((np.ones(len(coils)) * bb, np.ones(len(coils_TF)) * b))
base_a_list = np.hstack((np.ones(len(base_coils)) * aa, np.ones(len(base_coils_TF)) * a))
base_b_list = np.hstack((np.ones(len(base_coils)) * bb, np.ones(len(base_coils_TF)) * b))

LENGTH_WEIGHT = Weight(0.01)
LENGTH_TARGET = 90
LINK_WEIGHT = 1e4
CC_THRESHOLD = 0.8
CC_WEIGHT = 1e0
CS_THRESHOLD = 1.5
CS_WEIGHT = 1e1
# Weight for the Coil Coil forces term
FORCE_WEIGHT2 = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
FORCE_WEIGHT = Weight(1e-36)  # 1e-36 Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT2 = Weight(1e-24)  # 1e-24 Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
# Directory for output
OUT_DIR = ("./QH_continuation_fixed_TForder{:d}_n{:d}_lw{:.2e}_lt{:.2e}_lkw{:.2e}" +
           "_cct{:.2e}_ccw{:.2e}_cst{:.2e}_csw{:.2e}_fw{:.2e}_fww{:2e}_tw{:.2e}_tww{:2e}/").format(
    curves_TF[0].order, ncoils, LENGTH_WEIGHT.value, LENGTH_TARGET, LINK_WEIGHT,
    CC_THRESHOLD, CC_WEIGHT, CS_THRESHOLD, CS_WEIGHT, FORCE_WEIGHT.value,
    FORCE_WEIGHT2.value,
    TORQUE_WEIGHT.value,
    TORQUE_WEIGHT2.value)
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

curves_to_vtk(
    curves_TF,
    OUT_DIR + "curves_TF_0",
    close=True,
    extra_point_data=pointData_forces_torques(coils_TF, coils + coils_TF, np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b, np.ones(len(coils_TF)) * nturns_TF),
    I=currents_TF,
    NetForces=coil_net_forces(coils_TF, coils + coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF),
    NetTorques=coil_net_torques(coils_TF, coils + coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF)
)
curves_to_vtk(
    curves,
    OUT_DIR + "curves_0",
    close=True,
    extra_point_data=pointData_forces_torques(coils, coils + coils_TF, np.ones(len(coils)) * aa, np.ones(len(coils)) * bb, np.ones(len(coils)) * nturns),
    I=currents,
    NetForces=coil_net_forces(coils, coils + coils_TF, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb), np.ones(len(coils)) * nturns),
    NetTorques=coil_net_torques(coils, coils + coils_TF, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb), np.ones(len(coils)) * nturns)
)
# Force and Torque calculations spawn a bunch of spurious BiotSavart child objects -- erase them!
for c in (coils + coils_TF):
    c._children = set()

# Repeat for whole B field
pointData = {"B_N": np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_full_init", extra_data=pointData)
btot.set_points(s.gamma().reshape((-1, 3)))

# Define the individual terms objective function:
Jf = SquaredFlux(s, btot)
Jls_TF = [CurveLength(c) for c in base_curves_TF]
Jlength = QuadraticPenalty(sum(Jls_TF), LENGTH_TARGET, "max")

### Jcc below removed the dipoles!
Jccdist = CurveCurveDistance(curves + curves_TF, CC_THRESHOLD / 2.0, num_basecurves=len(coils + coils_TF))
Jccdist2 = CurveCurveDistance(curves_TF, CC_THRESHOLD, num_basecurves=len(coils_TF))

# Jcsdist = CurveSurfaceDistance(curves_TF, s, CS_THRESHOLD)
Jcsdist = CurveSurfaceDistance(curves + curves_TF, s, CS_THRESHOLD)
# Jcsdist2 = CurveSurfaceDistance(curves + curves_TF, s, CS_THRESHOLD)

# While the coil array is not moving around, they cannot
# interlink.
linkNum = LinkingNumber(curves + curves_TF, downsample=2)

# Currently, all force terms involve all the coils
all_coils = coils + coils_TF
all_base_coils = base_coils + base_coils_TF
Jforce = sum([LpCurveForce(c, all_coils, regularization_rect(a_list[i], b_list[i]), p=4, threshold=4e5 * 100, downsample=1
                           ) for i, c in enumerate(all_base_coils)])
Jforce2 = sum([SquaredMeanForce(c, all_coils, downsample=1) for c in all_base_coils])

# Errors creep in when downsample = 2
Jtorque = sum([LpCurveTorque(c, all_coils, regularization_rect(a_list[i], b_list[i]), p=2, threshold=4e5 * 100, downsample=1
                             ) for i, c in enumerate(all_base_coils)])
Jtorque2 = sum([SquaredMeanTorque(c, all_coils, downsample=1) for c in all_base_coils])

CURVATURE_THRESHOLD = 0.5
MSC_THRESHOLD = 0.05
CURVATURE_WEIGHT = 1e-2
MSC_WEIGHT = 1e-4
Jcs = [LpCurveCurvature(c.curve, 2, CURVATURE_THRESHOLD) for c in base_coils_TF]
Jmscs = [MeanSquaredCurvature(c.curve) for c in base_coils_TF]

JF = Jf \
    + CC_WEIGHT * Jccdist \
    + CC_WEIGHT * Jccdist2 \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
    + CS_WEIGHT * Jcsdist \
    + LENGTH_WEIGHT * Jlength \
    + LINK_WEIGHT * linkNum

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
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-C-Sep={Jccdist2.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
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

n_saves = 1
MAXITER = 5000
for i in range(1, n_saves + 1):
    print('Iteration ' + str(i) + ' / ' + str(n_saves))
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
                   options={'maxiter': MAXITER, 'maxcor': 1000}, tol=1e-20)
    # dofs = res.x

    dipole_currents = [c.current.get_value() for c in bs.coils]
    curves_to_vtk(
        [c.curve for c in bs.coils],
        OUT_DIR + "curves_{0:d}".format(i),
        close=True,
        extra_point_data=pointData_forces_torques(coils, coils + coils_TF, np.ones(len(coils)) * aa, np.ones(len(coils)) * bb, np.ones(len(coils)) * nturns),
        I=dipole_currents,
        NetForces=coil_net_forces(coils, coils + coils_TF, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb), np.ones(len(coils)) * nturns),
        NetTorques=coil_net_torques(coils, coils + coils_TF, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb), np.ones(len(coils)) * nturns),
    )
    curves_to_vtk(
        [c.curve for c in bs_TF.coils],
        OUT_DIR + "curves_TF_{0:d}".format(i),
        close=True,
        extra_point_data=pointData_forces_torques(coils_TF, coils + coils_TF, np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b, np.ones(len(coils_TF)) * nturns_TF),
        I=[c.current.get_value() for c in bs_TF.coils],
        NetForces=coil_net_forces(coils_TF, coils + coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF),
        NetTorques=coil_net_torques(coils_TF, coils + coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF),
    )
    for c in (coils + coils_TF):
        c._children = set()

    btot.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "surf_full_{0:d}".format(i), extra_data=pointData)

    pointData = {"B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                    ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "surf_full_normalizedBn_{0:d}".format(i), extra_data=pointData)

    btot.set_points(s.gamma().reshape((-1, 3)))
    print('Max I = ', np.max(np.abs(dipole_currents)))
    print('Min I = ', np.min(np.abs(dipole_currents)))
    calculate_on_axis_B(btot, s)
    # LENGTH_WEIGHT *= 0.01
    # JF = Jf \
    #     + CC_WEIGHT * Jccdist \
    #     + CS_WEIGHT * Jcsdist \
    #     + LINK_WEIGHT * linkNum \
    #     + LINK_WEIGHT2 * linkNum2 \
    #     + LENGTH_WEIGHT * sum(Jls_TF)


t2 = time.time()
print('Total time = ', t2 - t1)
btot.save(OUT_DIR + "biot_savart_optimized_QH.json")
print(OUT_DIR)
