#!/usr/bin/env python
r"""
"""

import os
import shutil
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.field import regularization_rect
from simsopt.field.force import coil_force, coil_torque, coil_net_torques, coil_net_forces, LpCurveForce, \
    SquaredMeanForce, \
    SquaredMeanTorque, LpCurveTorque, pointData_forces_torques
from simsopt.util import calculate_on_axis_B, remove_inboard_dipoles, remove_interlinking_dipoles_and_TFs
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier, curves_to_vtk, create_planar_curves_between_two_toroidal_surfaces
)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty

t1 = time.time()

continuation_run = False
if continuation_run:
    file_suffix = "_continuation"
else:
    file_suffix = ""

# Number of Fourier modes describing each Cartesian component of each coil:
order = 0

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
# input_name = 'wout_henneberg.nc'
input_name = 'input.henneberg_nfp2'
filename = TEST_DIR / input_name

# Initialize the boundary magnetic surface:
range_param = "half period"
nphi = 32
ntheta = 32
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

if continuation_run:
    btot = Optimizable.from_file("./QAnfp2_SchuettHenneberg/biot_savart_optimized.json")
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
else:
    ### Initialize some TF coils
    def initialize_coils_QA(TEST_DIR, s):
        """
        Initializes coils for each of the target configurations that are
        used for permanent magnet optimization.

        Args:
            TEST_DIR: String denoting where to find the input files.
            s: plasma boundary surface.
        Returns:
            base_curves: List of CurveXYZ class objects.
            curves: List of Curve class objects.
            coils: List of Coil class objects.
        """
        from simsopt.geo import create_equally_spaced_curves
        from simsopt.field import Current, coils_via_symmetries

        # generate planar TF coils
        ncoils = 2
        R0 = s.get_rc(0, 0) * 1.4
        R1 = s.get_rc(1, 0) * 4
        order = 16

        # Hard-coded the total current that gives about B ~ 5.7 T on axis with the initial coils
        total_current = 35000000
        print('Total current = ', total_current)

        # Only need Jax flag for CurvePlanarFourier class
        base_curves = create_equally_spaced_curves(
            ncoils, s.nfp, stellsym=True,
            R0=R0, R1=R1, order=order, numquadpoints=256,
        )

        base_currents = [(Current(total_current / ncoils * 1e-7) * 1e7) for _ in range(ncoils - 1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

        # Initialize the coil curves and save the data to vtk
        curves = [c.curve for c in coils]
        return base_curves, curves, coils, base_currents


    # initialize the coils
    base_curves_TF, curves_TF, coils_TF, currents_TF = initialize_coils_QA(TEST_DIR, s)
    num_TF_unique_coils = len(base_curves_TF)
    base_coils_TF = coils_TF[:num_TF_unique_coils]
    currents_TF = np.array([coil.current.get_value() for coil in coils_TF])

    # # Set up BiotSavart fields
    bs_TF = BiotSavart(coils_TF)

    # Calculate average, approximate on-axis B field strength
    calculate_on_axis_B(bs_TF, s)

    Nx = 5
    Ny = Nx
    Nz = Nx
    # Create the initial coils:
    base_curves, all_curves = create_planar_curves_between_two_toroidal_surfaces(
        s, s_inner, s_outer, Nx, Ny, Nz, order=order, coil_coil_flag=True, jax_flag=False,
    )

    base_curves = remove_inboard_dipoles(s, base_curves, eps=-0.4)
    base_curves = remove_interlinking_dipoles_and_TFs(base_curves, base_curves_TF)

    ncoils = len(base_curves)
    print('Ncoils = ', ncoils)
    for i in range(len(base_curves)):
        # Fix shape of each coil
        for j in range(2 * order + 1):
            base_curves[i].fix('x' + str(j))
        # Fix center points of each coil
        # base_curves[i].fix('x' + str(2 * order + 5))
        # base_curves[i].fix('x' + str(2 * order + 6))
        # base_curves[i].fix('x' + str(2 * order + 7))

    base_currents = [Current(1.0) * 1e7 for i in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    base_coils = coils[:ncoils]
    bs = BiotSavart(coils)
    btot = bs + bs_TF
    calculate_on_axis_B(btot, s)
    btot.set_points(s.gamma().reshape((-1, 3)))
    bs.set_points(s.gamma().reshape((-1, 3)))
    curves = [c.curve for c in coils]
    currents = [c.current.get_value() for c in coils]

a_list = np.hstack((np.ones(len(coils)) * aa, np.ones(len(coils_TF)) * a))
b_list = np.hstack((np.ones(len(coils)) * bb, np.ones(len(coils_TF)) * b))
base_a_list = np.hstack((np.ones(len(base_coils)) * aa, np.ones(len(base_coils_TF)) * a))
base_b_list = np.hstack((np.ones(len(base_coils)) * bb, np.ones(len(base_coils_TF)) * b))

LENGTH_WEIGHT = Weight(0.01)  # 0.001 for continuation run?
LINK_WEIGHT = 1e4
CS_THRESHOLD = 1.5
CS_WEIGHT = 1e1
FORCE_WEIGHT2 = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
if continuation_run:
    LENGTH_TARGET = 75
    CC_THRESHOLD = 0.8
    CC_WEIGHT = 1
    # Weight for the Coil Coil forces term
    FORCE_WEIGHT = Weight(4e-33)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
    TORQUE_WEIGHT2 = Weight(1e-23) 
else:
    LENGTH_TARGET = 85
    CC_THRESHOLD = 1.0
    CC_WEIGHT = 1e2
    # Weight for the Coil Coil forces term
    FORCE_WEIGHT = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
    TORQUE_WEIGHT2 = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons

# Directory for output
OUT_DIR = ("./QAnfp2_SchuettHenneberg)
os.makedirs(OUT_DIR, exist_ok=True)

curves_to_vtk(
    curves_TF,
    OUT_DIR + "curves_TF_initial" + file_suffix,
    close=True,
    extra_point_data=pointData_forces_torques(coils_TF, coils + coils_TF, np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b, np.ones(len(coils_TF)) * nturns_TF),
    I=currents_TF,
    NetForces=coil_net_forces(coils_TF, coils + coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF),
    NetTorques=coil_net_torques(coils_TF, coils + coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF)
)
curves_to_vtk(
    curves,
    OUT_DIR + "curves_dipoles_initial" + file_suffix,
    close=True,
    extra_point_data=pointData_forces_torques(coils, coils + coils_TF, np.ones(len(coils)) * aa, np.ones(len(coils)) * bb, np.ones(len(coils)) * nturns),
    I=currents,
    NetForces=coil_net_forces(coils, coils + coils_TF, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb), np.ones(len(coils)) * nturns),
    NetTorques=coil_net_torques(coils, coils + coils_TF, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb), np.ones(len(coils)) * nturns)
)
# Force and Torque calculations spawn a bunch of spurious BiotSavart child objects -- erase them!
for c in (coils + coils_TF):
    c._children = set()

btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_initial" + file_suffix, extra_data=pointData)
btot.set_points(s.gamma().reshape((-1, 3)))

# Define the individual terms objective function:
Jf = SquaredFlux(s, btot)
Jls = [CurveLength(c) for c in base_curves]
Jls_TF = [CurveLength(c) for c in base_curves_TF]
Jlength = QuadraticPenalty(sum(Jls_TF), LENGTH_TARGET, "max")
# Jlength2 = QuadraticPenalty(sum(Jls), LENGTH_TARGET2, "max")

# coil-coil and coil-plasma distances should be between all coils
Jccdist = CurveCurveDistance(curves + curves_TF, CC_THRESHOLD / 2.0, num_basecurves=len(coils + coils_TF))
Jccdist2 = CurveCurveDistance(curves_TF, CC_THRESHOLD, num_basecurves=len(coils_TF))
Jcsdist = CurveSurfaceDistance(curves + curves_TF, s, CS_THRESHOLD)

# While the coil array is not moving around, they cannot
# interlink.
linkNum = LinkingNumber(curves + curves_TF, downsample=2)

all_coils = coils + coils_TF
all_base_coils = base_coils + base_coils_TF
Jforce = sum([LpCurveForce(c, all_coils, regularization_rect(a, b), p=4, threshold=8e5 * 100, downsample=1
                           ) for i, c in enumerate(all_base_coils)])
Jforce2 = sum([SquaredMeanForce(c, all_coils, downsample=1) for c in all_base_coils])

# Errors creep in when downsample = 2
Jtorque = sum([LpCurveTorque(c, all_coils, regularization_rect(a, b), p=2, threshold=4e5 * 100, downsample=1
                             ) for i, c in enumerate(all_base_coils)])
Jtorque2 = sum([SquaredMeanTorque(c, all_coils, downsample=1) for c in all_base_coils])

CURVATURE_THRESHOLD = 0.5
MSC_THRESHOLD = 0.05
CURVATURE_WEIGHT = 1e-2
MSC_WEIGHT = 1e-1
Jcs = [LpCurveCurvature(c.curve, 2, CURVATURE_THRESHOLD) for c in base_coils_TF]
Jmscs = [MeanSquaredCurvature(c.curve) for c in base_coils_TF]

JF = Jf \
    + CC_WEIGHT * Jccdist \
    + CC_WEIGHT * Jccdist2 \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
    + LINK_WEIGHT * linkNum \
    + LENGTH_WEIGHT * Jlength  # \
# + LENGTH_WEIGHT2 * Jlength2

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
    # length_val2 = LENGTH_WEIGHT2.value * Jlength2.J()
    cc_val = CC_WEIGHT * Jccdist.J()
    cc_val2 = CC_WEIGHT * Jccdist.J()
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
    # cl_string2 = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.2f}" for c in base_curves_TF)
    msc_string = ", ".join(f"{J.J():.2f}" for J in Jmscs)
    outstr += f", ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls_TF):.2f}"
    # outstr += f", Len2=sum([{cl_string2}])={sum(J.J() for J in Jls):.2f}"
    valuestr += f", LenObj={length_val:.2e}"
    # valuestr += f", LenObj2={length_val2:.2e}"
    valuestr += f", ccObj={cc_val:.2e}"
    valuestr += f", ccObj2={cc_val2:.2e}"
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

MAXITER = 2000
res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
                options={'maxiter': MAXITER, 'maxcor': 1000}, tol=1e-20)

dipole_currents = [c.current.get_value() for c in bs.coils]
curves_to_vtk(
    [c.curve for c in bs.coils],
    OUT_DIR + "curves_dipoles_optimized" + file_suffix,
    close=True,
    extra_point_data=pointData_forces_torques(coils, coils + coils_TF, np.ones(len(coils)) * aa, np.ones(len(coils)) * bb, np.ones(len(coils)) * nturns),
    I=dipole_currents,
    NetForces=coil_net_forces(coils, coils + coils_TF, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb), np.ones(len(coils)) * nturns),
    NetTorques=coil_net_torques(coils, coils + coils_TF, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb), np.ones(len(coils)) * nturns),
)
curves_to_vtk(
    [c.curve for c in bs_TF.coils],
    OUT_DIR + "curves_TF_optimized" + file_suffix,
    close=True,
    extra_point_data=pointData_forces_torques(coils_TF, coils + coils_TF, np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b, np.ones(len(coils_TF)) * nturns_TF),
    I=[c.current.get_value() for c in bs_TF.coils],
    NetForces=coil_net_forces(coils_TF, coils + coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF),
    NetTorques=coil_net_torques(coils_TF, coils + coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF),
)

btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_optimized" + file_suffix, extra_data=pointData)

btot.set_points(s.gamma().reshape((-1, 3)))
print('Max I = ', np.max(np.abs(dipole_currents)))
print('Min I = ', np.min(np.abs(dipole_currents)))
calculate_on_axis_B(btot, s)

t2 = time.time()
print('Total time = ', t2 - t1)
btot.save(OUT_DIR + "biot_savart_optimized" + file_suffix + ".json")
print(OUT_DIR)
