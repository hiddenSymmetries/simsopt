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
# from simsopt.field import CoilCoilNetForces, CoilCoilNetTorques, \
#     TotalVacuumEnergy, CoilSelfNetForces, CoilCoilNetForces12, CoilCoilNetTorques12
from simsopt.field import regularization_rect
from simsopt.field.force import MeanSquaredForce, MeanSquaredForce2, coil_force, coil_torque, coil_net_torques, coil_net_forces, LpCurveForce, \
    SquaredMeanForce, MeanSquaredTorque, SquaredMeanTorque, LpCurveTorque
from simsopt.util import calculate_on_axis_B
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier, curves_to_vtk, create_equally_spaced_planar_curves,
    create_planar_curves_between_two_toroidal_surfaces
)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions

t1 = time.time()

# Number of Fourier modes describing each Cartesian component of each coil:
order = 0

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
input_name = 'input.LandremanPaul2021_QA_reactorScale_lowres'
filename = TEST_DIR / input_name

# Directory for output
OUT_DIR = "./ReactorScaleQA_DipoleArrays/"
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# Initialize the boundary magnetic surface:
range_param = "half period"
nphi = 32
ntheta = 32
poff = 1.5
coff = 3.5
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
    from simsopt.field import Current, Coil, coils_via_symmetries
    from simsopt.geo import curves_to_vtk

    # generate planar TF coils
    ncoils = 1
    R0 = s.get_rc(0, 0) * 2
    R1 = s.get_rc(1, 0) * 10
    order = 4

    from simsopt.mhd.vmec import Vmec
    vmec_file = 'wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc'
    total_current = Vmec(TEST_DIR / vmec_file).external_current() / (2 * s.nfp) / 1.2
    print('Total current = ', total_current)

    # Only need Jax flag for CurvePlanarFourier class
    base_curves = create_equally_spaced_curves(
        ncoils, s.nfp, stellsym=True, 
        R0=R0, R1=R1, order=order, numquadpoints=256,
        jax_flag=True,
    )

    # base_currents = [(Current(total_current / ncoils * 1e-7) * 1e7) for _ in range(ncoils - 1)]
    base_currents = [(Current(total_current / ncoils * 1e-7) * 1e7) for _ in range(ncoils)]
    base_currents[0].fix_all()

    # total_current = Current(total_current)
    # total_current.fix_all()
    # base_currents += [total_current - sum(base_currents)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    # for c in coils:
    #     c.current.fix_all()
    #     c.curve.fix_all()

    # Initialize the coil curves and save the data to vtk
    curves = [c.curve for c in coils]
    currents = [c.current.get_value() for c in coils]
    return base_curves, curves, coils, base_currents

# initialize the coils
base_curves_TF, curves_TF, coils_TF, currents_TF = initialize_coils_QA(TEST_DIR, s)
base_coils_TF = [coils_TF[0]]
currents_TF = np.array([coil.current.get_value() for coil in coils_TF])

# Set up BiotSavart fields
bs_TF = BiotSavart(coils_TF)

# Calculate average, approximate on-axis B field strength
calculate_on_axis_B(bs_TF, s)

# wire cross section for the TF coils is a square 20 cm x 20 cm
# Only need this if make self forces and TVE nonzero in the objective! 
a = 0.25
b = 0.25

# wire cross section for the dipole coils should be more like 5 cm x 5 cm
aa = 0.05
bb = 0.05

Nx = 5
Ny = Nx
Nz = Nx
# Create the initial coils:
base_curves, all_curves = create_planar_curves_between_two_toroidal_surfaces(
    s, s_inner, s_outer, Nx, Ny, Nz, order=order, coil_coil_flag=True, jax_flag=True,
    # numquadpoints=10  # Defaults is (order + 1) * 40 so this halves it
)
ncoils = len(base_curves)
print('Ncoils = ', ncoils)
for i in range(len(base_curves)):
    # base_curves[i].set('x' + str(2 * order + 1), np.random.rand(1) - 0.5)
    # base_curves[i].set('x' + str(2 * order + 2), np.random.rand(1) - 0.5)
    # base_curves[i].set('x' + str(2 * order + 3), np.random.rand(1) - 0.5)
    # base_curves[i].set('x' + str(2 * order + 4), np.random.rand(1) - 0.5)

    # Fix shape of each coil
    for j in range(2 * order + 1):
        base_curves[i].fix('x' + str(j))
    # Fix center points of each coil
    # base_curves[i].fix('x' + str(2 * order + 5))
    # base_curves[i].fix('x' + str(2 * order + 6))
    # base_curves[i].fix('x' + str(2 * order + 7))
base_currents = [Current(1e-1) * 2e7 for i in range(ncoils)]
# Fix currents in each coil
# for i in range(ncoils):
#     base_currents[i].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
base_coils = coils[:ncoils]

def pointData_forces_torques(coils, all_coils, aprimes, bprimes):
    contig = np.ascontiguousarray
    forces = np.zeros((len(coils), len(coils[0].curve.gamma()) + 1, 3))
    torques = np.zeros((len(coils), len(coils[0].curve.gamma()) + 1, 3))
    for i, c in enumerate(coils):
        aprime = aprimes[i]
        bprime = bprimes[i]
        forces[i, :-1, :] = coil_force(c, all_coils, regularization_rect(aprime, bprime))
        torques[i, :-1, :] = coil_torque(c, all_coils, regularization_rect(aprime, bprime))
    
    forces[:, -1, :] = forces[:, 0, :]
    torques[:, -1, :] = torques[:, 0, :]
    forces = forces.reshape(-1, 3)
    torques = torques.reshape(-1, 3)
    point_data = {"Pointwise_Forces": (contig(forces[:, 0]), contig(forces[:, 1]), contig(forces[:, 2])), 
                  "Pointwise_Torques": (contig(torques[:, 0]), contig(torques[:, 1]), contig(torques[:, 2]))}
    return point_data

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

curves_to_vtk(
    curves_TF, 
    OUT_DIR + "curves_TF_0", 
    close=True,
    extra_point_data=pointData_forces_torques(coils_TF, coils + coils_TF, np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b),
    I=currents_TF,
    NetForces=coil_net_forces(coils_TF, coils + coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b)),
    NetTorques=coil_net_torques(coils_TF, coils + coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b))
)
curves_to_vtk(
    curves, 
    OUT_DIR + "curves_0", 
    close=True, 
    extra_point_data=pointData_forces_torques(coils, coils + coils_TF, np.ones(len(coils)) * aa, np.ones(len(coils)) * bb),
    I=currents,
    NetForces=coil_net_forces(coils, coils + coils_TF, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb)),
    NetTorques=coil_net_torques(coils, coils + coils_TF, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb))
)
pointData = {"B_N": np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init_DA", extra_data=pointData)

btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_full_init_DA", extra_data=pointData)
btot.set_points(s.gamma().reshape((-1, 3)))

# Repeat for whole B field
pointData = {"B_N": np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_full_init", extra_data=pointData)
btot.set_points(s.gamma().reshape((-1, 3)))

LENGTH_WEIGHT = Weight(0.01)
LENGTH_TARGET = 70
# CURRENTS_WEIGHT = 10
LINK_WEIGHT = 100
LINK_WEIGHT2 = 1e-3
CC_THRESHOLD = 1.0
CC_WEIGHT = 10
CS_THRESHOLD = 1.3
CS_WEIGHT = 1
# CURVATURE_THRESHOLD = 1.
# CURVATURE_WEIGHT = 1e-12
# MSC_THRESHOLD = 1
# MSC_WEIGHT = 1e-12

# Weight for the Coil Coil forces term
FORCE_WEIGHT = Weight(5e-18) # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
# And this term weights the NetForce^2 ~ 10^10-10^12 

# TORQUE_WEIGHT = Weight(1e-18)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons

# TVE_WEIGHT = 1e-19

SF_WEIGHT = 0.0

# Define the individual terms objective function:
Jf = SquaredFlux(s, btot)
# Separate length penalties on the dipole coils and the TF coils
# since they have very different sizes
Jls = [CurveLength(c) for c in base_curves]
Jls_TF = [CurveLength(c) for c in base_curves_TF]
Jlength = QuadraticPenalty(sum(Jls_TF), LENGTH_TARGET, "max")

# coil-coil and coil-plasma distances should be between all coils
Jccdist = CurveCurveDistance(curves_TF, CC_THRESHOLD, num_basecurves=len(coils_TF))
# Jccdist = CurveCurveDistance(curves + curves_TF, CC_THRESHOLD, num_basecurves=ncoils + len(coils_TF))
# Jcsdist = CurveSurfaceDistance(curves + curves_TF, s, CS_THRESHOLD)
Jcsdist = CurveSurfaceDistance(curves + curves_TF, s, CS_THRESHOLD)

# Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
# Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

# While the coil array is not moving around, they cannot
# interlink. 
linkNum = LinkingNumber(curves_TF)
linkNum2 = LinkingNumber(curves)
##### Note need coils_TF + coils below!!!!
Jforce = [SquaredMeanForce(c, coils + coils_TF, regularization_rect(base_a_list[i], base_b_list[i])) for i, c in enumerate(base_coils + base_coils_TF)]


print('Timing calls: ')
t1 = time.time()
Jf.J()
t2 = time.time()
print('Jf time = ', t2 - t1, ' s')
t1 = time.time()
Jf.dJ()
t2 = time.time()
print('dJf time = ', t2 - t1, ' s')
t1 = time.time()
Jccdist.J()
Jccdist.dJ()
t2 = time.time()
print('Jcc time = ', t2 - t1, ' s')
t1 = time.time()
Jcsdist.J()
Jcsdist.dJ()
t2 = time.time()
print('Jcs time = ', t2 - t1, ' s')
t1 = time.time()
linkNum.J()
linkNum.dJ()
t2 = time.time()
print('linkNum time = ', t2 - t1, ' s')
t1 = time.time()
linkNum2.J()
linkNum2.dJ()
t2 = time.time()
print('linkNum2 time = ', t2 - t1, ' s')
t1 = time.time()
Jlength.J()
Jlength.dJ()
t2 = time.time()
print('sum(Jls_TF) time = ', t2 - t1, ' s')
t1 = time.time()
sum(Jforce).J()
t2 = time.time()
print('Jforces time = ', t2 - t1, ' s')
t1 = time.time()
sum(Jforce).dJ()
t2 = time.time()
print('dJforces time = ', t2 - t1, ' s')

# Jforces = CoilCoilNetForces(bs) + CoilCoilNetForces12(bs, bs_TF) + CoilCoilNetForces(bs_TF)
# Jtorque = CoilCoilNetTorques(bs) + CoilCoilNetTorques12(bs, bs_TF) + CoilCoilNetTorques(bs_TF)
# Jtve = TotalVacuumEnergy(bs, a=a, b=b)
# Jsf = CoilSelfNetForces(bs, a=a, b=b)

# Jccdist_TF = CurveCurveDistance(curves_TF, CC_THRESHOLD, num_basecurves=ncoils)
# Jcsdist_TF = CurveSurfaceDistance(curves_TF, s, CS_THRESHOLD)
# Jcs_TF = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves_TF]
# Jmscs_TF = [MeanSquaredCurvature(c) for c in base_curves_TF]

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:

# Coil shapes and center positions are fixed right now so not including this one below
# + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), 2.6*ncoils) \

# class currents_obj(Optimizable):
#     def __init__(self, currents):
#         self.currents = currents
#         Optimizable.__init__(self, depends_on=[currents])

#     def J(self):
#         return np.sum((np.array(self.currents) - 1.0e6) ** 2)

#     def dJ(self):
#         return 2.0 * (np.array(self.currents) - 1.0e6)

# DipoleCurrentsObj = currents_obj(base_currents)
# DipoleCurrentsObj = QuadraticPenalty(base_currents, 1e6, "max")
JF = Jf \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + LINK_WEIGHT * linkNum \
    + LINK_WEIGHT2 * linkNum2 \
    + LENGTH_WEIGHT * Jlength  \
    + FORCE_WEIGHT * sum(Jforce)  # \
    # + TORQUE_WEIGHT * sum(Jtorque) 
    # + TVE_WEIGHT * Jtve
    # + SF_WEIGHT * Jsf
    # + CURRENTS_WEIGHT * DipoleCurrentsObj
    # + CURVATURE_WEIGHT * sum(Jcs_TF) \
    # + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs_TF) \
#    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs) \
    # + CURVATURE_WEIGHT * sum(Jcs) \

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    length_val = LENGTH_WEIGHT.value * Jlength.J()
    cc_val = CC_WEIGHT * Jccdist.J()
    cs_val = CS_WEIGHT * Jcsdist.J()
    link_val1 = LINK_WEIGHT * linkNum.J()
    link_val2 = LINK_WEIGHT2 * linkNum2.J()
    forces_val = FORCE_WEIGHT.value * sum(J.J() for J in Jforce)
    # torques_val = TORQUE_WEIGHT.value * sum(J.J() for J in Jtorque)
    # tve_val = TVE_WEIGHT * Jtve.J()
    # sf_val = SF_WEIGHT * Jsf.J()
    BdotN = np.mean(np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    BdotN_over_B = np.mean(np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
        ) / np.mean(btot.AbsB())
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}, ⟨B·n⟩/⟨B⟩={BdotN_over_B:.1e}"
    valuestr = f"J={J:.2e}, Jf={jf:.2e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls_TF])
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls_TF):.2f}" 
    valuestr += f", LenObj={length_val:.2e}" 
    valuestr += f", ccObj={cc_val:.2e}" 
    valuestr += f", csObj={cs_val:.2e}" 
    valuestr += f", Lk1Obj={link_val1:.2e}" 
    valuestr += f", Lk2Obj={link_val2:.2e}" 
    valuestr += f", forceObj={forces_val:.2e}" 
    # valuestr += f", torqueObj={torques_val:.2e}" 
    # valuestr += f", tveObj={tve_val:.2e}" 
    # valuestr += f", sfObj={sf_val:.2e}" 
    # valuestr += f", currObj={curr_val:.2e}" 
    #, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    # outstr += f", avg(L)={np.mean(np.array([J.J() for J in Jls])):.2f}"
    # outstr += f", Lengths=" + cl_string
    # outstr += f", avg(kappa)={np.mean(np.array([c.kappa() for c in base_curves])):.2f}"
    # outstr += f", var(kappa)={np.mean(np.array([c.kappa() for c in base_curves])):.2f}"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f", Link Number = {linkNum.J()}"
    outstr += f", Link Number 2 = {linkNum2.J()}"
    outstr += f", F={sum(J.J() for J in Jforce):.2e}"
    # outstr += f", T={sum(J.J() for J in Jtorque):.2e}"
    # outstr += f", TVE={Jtve.J():.1e}"
    # outstr += f", TotalSelfForces={Jsf.J():.1e}"
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
    print("err", (J1-J2)/(2*eps) - dJh)  #(J1-J2)/(2*eps), dJh, (J1-J2)/(2*eps) - dJh)

print('Timing calls: ')
t1 = time.time()
Jf.J()
t2 = time.time()
print('Jf time = ', t2 - t1, ' s')
t1 = time.time()
Jf.dJ()
t2 = time.time()
print('dJf time = ', t2 - t1, ' s')
t1 = time.time()
Jccdist.J()
Jccdist.dJ()
t2 = time.time()
print('Jcc time = ', t2 - t1, ' s')
t1 = time.time()
Jcsdist.J()
Jcsdist.dJ()
t2 = time.time()
print('Jcs time = ', t2 - t1, ' s')
t1 = time.time()
linkNum.J()
linkNum.dJ()
t2 = time.time()
print('linkNum time = ', t2 - t1, ' s')
t1 = time.time()
linkNum2.J()
linkNum2.dJ()
t2 = time.time()
print('linkNum2 time = ', t2 - t1, ' s')
t1 = time.time()
sum(Jls_TF).J()
sum(Jls_TF).dJ()
t2 = time.time()
print('sum(Jls_TF) time = ', t2 - t1, ' s')
t1 = time.time()
sum(Jforce).J()
t2 = time.time()
print('Jforces time = ', t2 - t1, ' s')
t1 = time.time()
sum(Jforce).dJ()
t2 = time.time()
print('dJforces time = ', t2 - t1, ' s')

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")

n_saves = 1
MAXITER = 400
for i in range(1, n_saves + 1):
    print('Iteration ' + str(i) + ' / ' + str(n_saves))
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', 
        options={'maxiter': MAXITER, 'maxcor': 1000}, tol=1e-15)
    dofs = res.x

    dipole_currents = [c.current.get_value() for c in bs.coils]
    curves_to_vtk(
        [c.curve for c in bs.coils], 
        OUT_DIR + "curves_{0:d}".format(i), 
        close=True,
        extra_point_data=pointData_forces_torques(coils, coils + coils_TF, np.ones(len(coils)) * aa, np.ones(len(coils)) * bb),
        I=dipole_currents,
        NetForces=coil_net_forces(coils, coils + coils_TF, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb)),
        NetTorques=coil_net_torques(coils, coils + coils_TF, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb)),
    )
    curves_to_vtk(
        [c.curve for c in bs_TF.coils], 
        OUT_DIR + "curves_TF_{0:d}".format(i), 
        close=True, 
        extra_point_data=pointData_forces_torques(coils_TF, coils + coils_TF, np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b),
        I=[c.current.get_value() for c in bs_TF.coils],
        NetForces=coil_net_forces(coils_TF, coils + coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b)),
        NetTorques=coil_net_torques(coils_TF, coils + coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b)),
    )

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
btot.save("biot_savart_optimized_QA.json")

