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
from simsopt.field import regularization_rect, PSCArray
from simsopt.field.force import coil_force, coil_torque, coil_net_torques, coil_net_forces, LpCurveForce, \
    SquaredMeanForce, \
    SquaredMeanTorque, LpCurveTorque
from simsopt.util import calculate_on_axis_B
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier, curves_to_vtk, create_planar_curves_between_two_toroidal_surfaces
)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty

t1 = time.time()

# Number of Fourier modes describing each Cartesian component of each coil:
order = 0

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
input_name = 'input.LandremanPaul2021_QA_reactorScale_lowres'
filename = TEST_DIR / input_name

# Initialize the boundary magnetic surface:
range_param = "half period"
nphi = 32
ntheta = 32
poff = 1.5
coff = 1.0
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

### Initialize some TF coils
def initialize_coils_QA(TEST_DIR, s):
    """
    Initializes coils for each of the target configurations that are
    used for permanent magnet optimization.

    Args:
        config_flag: String denoting the stellarator configuration 
          being initialized.
        TEST_DIR: String denoting where to find the input files.
        out_dir: Path or string for the output directory for saved files.
        s: plasma boundary surface.
    Returns:
        base_curves: List of CurveXYZ class objects.
        curves: List of Curve class objects.
        coils: List of Coil class objects.
    """
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import Current, coils_via_symmetries

    # generate planar TF coils
    ncoils = 4
    R0 = s.get_rc(0, 0) * 1.2
    R1 = s.get_rc(1, 0) * 5
    order = 4

    from simsopt.mhd.vmec import Vmec
    vmec_file = 'wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc'
    total_current = Vmec(TEST_DIR / vmec_file).external_current() / (2 * s.nfp) / 1.2
    print('Total current = ', total_current)

    # Only need Jax flag for CurvePlanarFourier class
    base_curves = create_equally_spaced_curves(
        ncoils, s.nfp, stellsym=True,
        R0=R0, R1=R1, order=order, numquadpoints=256,
        jax_flag=False,
    )

    base_currents = [(Current(total_current / ncoils * 1e-7) * 1e7) for _ in range(ncoils - 1)]
    # [base_currents[i].fix_all() for i in range(len(base_currents))]
    # [base_curves[i].fix_all() for i in range(len(base_curves))]

    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

    # Initialize the coil curves and save the data to vtk
    curves = [c.curve for c in coils]
    return base_curves, curves, coils, base_currents

# initialize the TF coils
base_curves_TF, curves_TF, coils_TF, currents_TF = initialize_coils_QA(TEST_DIR, s)
num_TF_unique_coils = len(base_curves_TF)
base_coils_TF = coils_TF[:num_TF_unique_coils]
currents_TF = np.array([coil.current.get_value() for coil in coils_TF])

# wire cross section for the TF coils is a square 20 cm x 20 cm
# Only need this if make self forces and TVE nonzero in the objective!
a = 0.2
b = 0.2
nturns = 100
nturns_TF = 200

# wire cross section for the dipole coils should be more like 5 cm x 5 cm
aa = 0.05
bb = 0.05

Nx = 6
Ny = Nx
Nz = Nx
# Create the initial coils:
base_curves, all_curves = create_planar_curves_between_two_toroidal_surfaces(
    s, s_inner, s_outer, Nx, Ny, Nz, order=order, coil_coil_flag=True, jax_flag=False,
)

# Remove if within a radius of a TF coil
# import warnings
# keep_inds = []
# for ii in range(len(base_curves)):
#     counter = 0
#     for i in range(base_curves[0].gamma().shape[0]):
#         eps = 0.05
#         for j in range(len(base_curves_TF)):
#             for k in range(base_curves_TF[j].gamma().shape[0]):
#                 dij = np.sqrt(np.sum((base_curves[ii].gamma()[i, :] - base_curves_TF[j].gamma()[k, :]) ** 2))
#                 conflict_bool = (dij < (1.0 + eps) * base_curves[0].x[0])
#                 if conflict_bool:
#                     print('bad indices = ', i, j, dij, base_curves[0].x[0])
#                     warnings.warn(
#                         'There is a PSC coil initialized such that it is within a radius'
#                         'of a TF coil. Deleting these PSCs now.')
#                     counter += 1
#                     break
#     if counter == 0:
#         keep_inds.append(ii)
# base_curves = np.array(base_curves)[keep_inds]

ncoils = len(base_curves)
coil_normals = np.zeros((ncoils, 3))
plasma_points = s.gamma().reshape(-1, 3)
plasma_unitnormals = s.unitnormal().reshape(-1, 3)
for i in range(ncoils):
    point = (base_curves[i].get_dofs()[-3:])
    dists = np.sum((point - plasma_points) ** 2, axis=-1)
    min_ind = np.argmin(dists)
    coil_normals[i, :] = plasma_unitnormals[min_ind, :]
    # coil_normals[i, :] = (plasma_points[min_ind, :] - point)
coil_normals = coil_normals / np.linalg.norm(coil_normals, axis=-1)[:, None]
alphas = np.arcsin(
    -coil_normals[:, 1],
)
deltas = np.arctan2(coil_normals[:, 0], coil_normals[:, 2])
for i in range(len(base_curves)):
    alpha2 = alphas[i] / 2.0
    delta2 = deltas[i] / 2.0
    calpha2 = np.cos(alpha2)
    salpha2 = np.sin(alpha2)
    cdelta2 = np.cos(delta2)
    sdelta2 = np.sin(delta2)
    base_curves[i].set('x' + str(2 * order + 1), calpha2 * cdelta2)
    base_curves[i].set('x' + str(2 * order + 2), salpha2 * cdelta2)
    base_curves[i].set('x' + str(2 * order + 3), calpha2 * sdelta2)
    base_curves[i].set('x' + str(2 * order + 4), -salpha2 * sdelta2)
    # Fix orientations of each coil
    # base_curves[i].fix('x' + str(2 * order + 1))
    # base_curves[i].fix('x' + str(2 * order + 2))
    # base_curves[i].fix('x' + str(2 * order + 3))
    # base_curves[i].fix('x' + str(2 * order + 4))

    # Fix shape of each coil
    for j in range(2 * order + 1):
        base_curves[i].fix('x' + str(j))
    # Fix center points of each coil
    # base_curves[i].fix('x' + str(2 * order + 5))
    # base_curves[i].fix('x' + str(2 * order + 6))
    # base_curves[i].fix('x' + str(2 * order + 7))

# q_initial = base_curves.x[2 * order + 1:2 * order + 5]  # get the rotational dofs
# eps = 1e-6
# opt_bounds = [(None, None), (None, None)]
# for i in range(len(base_curves)):
#     for j in range(4):
#         opt_bounds.append((base_curves[i].x[j] - eps,
#             base_curves[i].x[j] + eps))
# opt_bounds = np.vstack((opt_bounds1, opt_bounds2))
# opt_bounds = tuple(map(tuple, opt_bounds))

ncoils = len(base_curves)
a_list = np.ones(len(base_curves)) * a
b_list = np.ones(len(base_curves)) * a
print('Num dipole coils = ', ncoils)

# Define function to compute the pointwise forces and torques
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

# Initialize the PSCArray object
eval_points = s.gamma().reshape(-1, 3)
psc_array = PSCArray(base_curves, coils_TF, eval_points, a_list, b_list, nfp=s.nfp, stellsym=s.stellsym)

# Calculate average, approximate on-axis B field strength
calculate_on_axis_B(psc_array.biot_savart_TF, s)
psc_array.biot_savart_TF.set_points(eval_points)
btot = psc_array.biot_savart_total
calculate_on_axis_B(btot, s)
btot.set_points(s.gamma().reshape((-1, 3)))

# bs.set_points(s.gamma().reshape((-1, 3)))
coils = psc_array.coils
base_coils = coils[:ncoils]
curves = [c.curve for c in coils]
currents = [c.current.get_value() for c in coils]
a_list = np.hstack((np.ones(len(coils)) * aa, np.ones(len(coils_TF)) * a))
b_list = np.hstack((np.ones(len(coils)) * bb, np.ones(len(coils_TF)) * b))

LENGTH_WEIGHT = Weight(0.01)
LENGTH_TARGET = 150
LINK_WEIGHT = 1e4
CC_THRESHOLD = 0.8
CC_WEIGHT = 1e2
CS_THRESHOLD = 1.5
CS_WEIGHT = 1e2
# Weight for the Coil Coil forces term
FORCE_WEIGHT = Weight(0.0)  # 1e-34 Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
FORCE_WEIGHT2 = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT2 = Weight(0.0)  # 1e-22 Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
# Directory for output
OUT_DIR = "./passive_coils_angle_scan/"
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

curves_to_vtk(
    [c.curve for c in btot.Bfields[1].coils],
    OUT_DIR + "curves_TF_0",
    close=True,
    extra_point_data=pointData_forces_torques(coils_TF, coils + coils_TF, np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b, np.ones(len(coils_TF)) * nturns_TF),
    I=[c.current.get_value() for c in btot.Bfields[1].coils],
    NetForces=coil_net_forces(coils_TF, coils + coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF),
    NetTorques=coil_net_torques(coils_TF, coils + coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF)
)
curves_to_vtk(
    [c.curve for c in btot.Bfields[0].coils],
    OUT_DIR + "curves_0",
    close=True,
    extra_point_data=pointData_forces_torques(coils, coils + coils_TF, np.ones(len(coils)) * aa, np.ones(len(coils)) * bb, np.ones(len(coils)) * nturns),
    I=[c.current.get_value() for c in btot.Bfields[0].coils],
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
s_plot.to_vtk(OUT_DIR + "surf_full_init", extra_data=pointData)
btot.set_points(s.gamma().reshape((-1, 3)))

bpsc = btot.Bfields[0]
bpsc.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bpsc.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
    "B_N / B": (np.sum(bpsc.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                    ) / np.linalg.norm(bpsc.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_init_PSC", extra_data=pointData)
bpsc.set_points(s.gamma().reshape((-1, 3)))

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
Jforce = sum([LpCurveForce(c, all_coils, regularization_rect(a_list[i], b_list[i]), p=4, threshold=4e5 * 100, downsample=1
                           ) for i, c in enumerate(all_base_coils)])
Jforce2 = sum([SquaredMeanForce(c, all_coils, downsample=1) for c in all_base_coils])

# Errors creep in when downsample = 2
Jtorque = sum([LpCurveTorque(c, all_coils, regularization_rect(a_list[i], b_list[i]), p=2, threshold=4e5 * 100, downsample=1
                             ) for i, c in enumerate(all_base_coils)])
Jtorque2 = sum([SquaredMeanTorque(c, all_coils, downsample=1) for c in all_base_coils])

CURVATURE_THRESHOLD = 0.5
MSC_THRESHOLD = 0.05
CURVATURE_WEIGHT = 1e-5
MSC_WEIGHT = 1e-6
Jcs = [LpCurveCurvature(c.curve, 2, CURVATURE_THRESHOLD) for c in base_coils_TF]
Jmscs = [MeanSquaredCurvature(c.curve) for c in base_coils_TF]

JF = Jf \
    + CS_WEIGHT * Jcsdist \
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

print(JF.dof_names)
# for i in range(len(JF.dof_names) - len(opt_bounds)):
#     opt_bounds.append((None, None))
# print(opt_bounds)
# print(opt_bounds, np.shape(opt_bounds), np.shape(JF.dof_names))
# exit()

def fun(dofs):
    JF.x = dofs
    # absolutely essential line that updates the PSC currents even though they are not
    # being directly optimized. 
    psc_array.recompute_currents()
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
    BdotN_over_B = BdotN / np.mean(btot.AbsB())
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
for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    t1 = time.time()
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    t2 = time.time()
    print("err", (J1-J2)/(2*eps) - dJh)
    print((J1-J2)/(2*eps), dJh)

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")

n_saves = 1
MAXITER = 600
for i in range(1, n_saves + 1):
    print('Iteration ' + str(i) + ' / ' + str(n_saves))
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B',   # bounds=opt_bounds,
                   options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
    # dofs = res.x

    bpsc = btot.Bfields[0]
    bpsc.set_points(s_plot.gamma().reshape((-1, 3)))
    dipole_currents = [c.current.get_value() for c in bpsc.coils]
    print(dipole_currents)
    curves_to_vtk(
        [c.curve for c in bpsc.coils],
        OUT_DIR + "curves_{0:d}".format(i),
        close=True,
        extra_point_data=pointData_forces_torques(coils, coils + coils_TF, np.ones(len(coils)) * aa, np.ones(len(coils)) * bb, np.ones(len(coils)) * nturns),
        I=dipole_currents,
        NetForces=coil_net_forces(coils, coils + coils_TF, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb), np.ones(len(coils)) * nturns),
        NetTorques=coil_net_torques(coils, coils + coils_TF, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb), np.ones(len(coils)) * nturns),
    )
    curves_to_vtk(
        [c.curve for c in btot.Bfields[1].coils],
        OUT_DIR + "curves_TF_{0:d}".format(i),
        close=True,
        extra_point_data=pointData_forces_torques(coils_TF, coils + coils_TF, np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b, np.ones(len(coils_TF)) * nturns_TF),
        I=[c.current.get_value() for c in btot.Bfields[1].coils],
        NetForces=coil_net_forces(coils_TF, coils + coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF),
        NetTorques=coil_net_torques(coils_TF, coils + coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF),
    )

    btot.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
        "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                    ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "surf_full_final", extra_data=pointData)

    btf = btot.Bfields[1]
    btf.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(btf.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
        "B_N / B": (np.sum(btf.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                    ) / np.linalg.norm(btf.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "surf_full_TF", extra_data=pointData)

    bpsc.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(bpsc.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
        "B_N / B": (np.sum(bpsc.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                    ) / np.linalg.norm(bpsc.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "surf_full_PSC", extra_data=pointData)

    btot.set_points(s.gamma().reshape((-1, 3)))
    print('Max I = ', np.max(dipole_currents))
    print('Min I = ', np.min(dipole_currents))
    calculate_on_axis_B(btot, s)

t2 = time.time()
print('Total time = ', t2 - t1)
btot.save(OUT_DIR + "biot_savart_optimized_QA.json")
print(OUT_DIR)
