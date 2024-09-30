#!/usr/bin/env python
r"""
"""

import os
import shutil
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from simsopt.field import JaxBiotSavart, JaxCurrent, coils_via_symmetries
from simsopt.field import CoilCoilNetForces, CoilCoilNetTorques, \
    TotalVacuumEnergy, CoilSelfNetForces, CoilCoilNetForces12, CoilCoilNetTorques12
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
# input_name = 'wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc'
# filename = TEST_DIR / input_name

# Directory for output
OUT_DIR = "./dipole_array_optimization_QA_reactorScale_jax/"
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################

# Initialize the boundary magnetic surface:
range_param = "half period"
nphi = 32
ntheta = 32
poff = 1.5
coff = 2.5
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
    from simsopt.field import JaxCurrent, Coil, coils_via_symmetries
    from simsopt.geo import curves_to_vtk

    # generate planar TF coils
    ncoils = 1
    R0 = s.get_rc(0, 0)
    R1 = s.get_rc(1, 0) * 5
    order = 4

    from simsopt.mhd.vmec import Vmec
    vmec_file = 'wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc'
    total_current = Vmec(TEST_DIR / vmec_file).external_current() / (2 * s.nfp) / 1.105
    print('Total current = ', total_current)
    base_curves = create_equally_spaced_curves(
        ncoils, s.nfp, stellsym=True, 
        R0=R0, R1=R1, order=order, numquadpoints=256,
        jax_flag=True,
    )

    # base_currents = [(JaxCurrent(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils - 1)]
    base_currents = [(JaxCurrent(total_current / ncoils * 1e-7) * 1e7) for _ in range(ncoils)]
    base_currents[0].fix_all()

    # total_current = JaxCurrent(total_current)
    # total_current.fix_all()
    # base_currents += [total_current - sum(base_currents)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    # for c in coils:
    #     c.current.fix_all()
    #     c.curve.fix_all()

    # Initialize the coil curves and save the data to vtk
    curves = [c.curve for c in coils]
    currents = [c.current.get_value() for c in coils]
    curves_to_vtk(curves, OUT_DIR + "curves_TF_0", I=currents,
            # NetForces=np.array(bs.coil_coil_forces()),
            # NetTorques=bs.coil_coil_torques(),
            # NetSelfForces=bs.coil_self_forces(a, b)
    )
    return base_curves, curves, coils, base_currents

# initialize the coils
base_curves_TF, curves_TF, coils_TF, currents_TF = initialize_coils_QA(TEST_DIR, s)
# currents_TF = np.array([coil.current.get_value() for coil in coils_TF])

# Set up JaxBiotSavart fields
bs_TF = JaxBiotSavart(coils_TF)

# Calculate average, approximate on-axis B field strength
calculate_on_axis_B(bs_TF, s)

# wire cross section for the TF coils is a square 10 cm x 10 cm
# Only need this if make self forces and TVE nonzero in the objective! 
# a = 0.1
# b = 0.1

Nx = 5
Ny = Nx
Nz = Nx
# Create the initial coils:
base_curves, all_curves = create_planar_curves_between_two_toroidal_surfaces(
    s, s_inner, s_outer, Nx, Ny, Nz, order=order, coil_coil_flag=True, jax_flag=True,
    numquadpoints=20  # Defaults is (order + 1) * 40 so this halves it
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
base_currents = [JaxCurrent(1e-1) * 2e7 for i in range(ncoils)]
# Fix currents in each coil
# for i in range(ncoils):
#     base_currents[i].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

bs = JaxBiotSavart(coils)  # Include TF coils here
btot = bs + bs_TF
btot.set_points(s.gamma().reshape((-1, 3)))
bs.set_points(s.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
currents = [c.current.get_value() for c in coils]
# print(len(curves))
# print(CoilCoilNetForces12(bs, bs_TF).coil_coil_forces12())
# print(np.sum(CoilCoilNetForces12(bs, bs_TF).coil_coil_forces12() ** 2, axis=-1))

# exit()
curves_to_vtk(curves, OUT_DIR + "curves_0", close=True, I=currents,
            NetForces=bs.coil_coil_forces(),
            NetTorques=bs.coil_coil_torques(),
            MixedCoilForces=CoilCoilNetForces12(bs, bs_TF).coil_coil_forces12()[:len(curves), :],
            MixedCoilTorques=CoilCoilNetTorques12(bs, bs_TF).coil_coil_torques12()[:len(curves), :],
            # NetSelfForces=bs.coil_self_forces(a, b)
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

LENGTH_WEIGHT = Weight(0.005)
# CURRENTS_WEIGHT = 10
LINK_WEIGHT = 100
LINK_WEIGHT2 = 1e-5
CC_THRESHOLD = 0.7
CC_WEIGHT = 400
CS_THRESHOLD = 1.3
CS_WEIGHT = 1e1
# CURVATURE_THRESHOLD = 1.
# CURVATURE_WEIGHT = 1e-12
# MSC_THRESHOLD = 1
# MSC_WEIGHT = 1e-12

# Weight for the Coil Coil forces term
FORCES_WEIGHT = 0.0  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
# And this term weights the NetForce^2 ~ 10^10-10^12 

TORQUES_WEIGHT = 0.0  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons

# TVE_WEIGHT = 1e-19

SF_WEIGHT = 0.0

# Define the individual terms objective function:
Jf = SquaredFlux(s, btot)
# Separate length penalties on the dipole coils and the TF coils
# since they have very different sizes
Jls = [CurveLength(c) for c in base_curves]
Jls_TF = [CurveLength(c) for c in base_curves_TF]

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
# Jforces = CoilCoilNetForces(bs) + CoilCoilNetForces12(bs, bs_TF) + CoilCoilNetForces(bs_TF)
# Jtorques = CoilCoilNetTorques(bs) + CoilCoilNetTorques12(bs, bs_TF) + CoilCoilNetTorques(bs_TF)
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

# DipoleJaxCurrentsObj = currents_obj(base_currents)
# DipoleJaxCurrentsObj = QuadraticPenalty(base_currents, 1e6, "max")
JF = Jf \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + LINK_WEIGHT * linkNum \
    + LINK_WEIGHT2 * linkNum2 \
    + LENGTH_WEIGHT * sum(Jls_TF) 
    # + FORCES_WEIGHT * Jforces \
    # + TORQUES_WEIGHT * Jtorques # \
    # + TVE_WEIGHT * Jtve
    # + SF_WEIGHT * Jsf
    # + CURRENTS_WEIGHT * DipoleJaxCurrentsObj
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
    length_val = LENGTH_WEIGHT.value * sum(J.J() for J in Jls_TF)
    cc_val = CC_WEIGHT * Jccdist.J()
    cs_val = CS_WEIGHT * Jcsdist.J()
    link_val1 = LINK_WEIGHT * linkNum.J()
    link_val2 = LINK_WEIGHT2 * linkNum2.J()
    # forces_val = FORCES_WEIGHT * Jforces.J()
    # torques_val = TORQUES_WEIGHT * Jtorques.J()
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
    # valuestr += f", forceObj={forces_val:.2e}" 
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
    # outstr += f", C-C-Forces={Jforces.J():.1e}"
    # outstr += f", C-C-Torques={Jtorques.J():.1e}"
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
        options={'maxiter': MAXITER, 'maxcor': 500}, tol=1e-10)
    dofs = res.x

    dipole_currents = [c.current.get_value() for c in bs.coils]
    curves_to_vtk([c.curve for c in bs.coils], OUT_DIR + "curves_{0:d}".format(i), 
        I=dipole_currents,
        NetForces=np.array(bs.coil_coil_forces()),
        NetTorques=bs.coil_coil_torques(),
        MixedCoilForces=CoilCoilNetForces12(bs, bs_TF).coil_coil_forces12()[:len(curves), :],
        MixedCoilTorques=CoilCoilNetTorques12(bs, bs_TF).coil_coil_torques12()[:len(curves), :],
        # NetSelfForces=bs.coil_self_forces(a, b)
        )
    curves_to_vtk([c.curve for c in bs_TF.coils], OUT_DIR + "curves_TF_{0:d}".format(i), 
        I=[c.current.get_value() for c in bs_TF.coils],
        NetForces=np.array(bs_TF.coil_coil_forces()),
        NetTorques=bs_TF.coil_coil_torques(),
        MixedCoilForces=CoilCoilNetForces12(bs, bs_TF).coil_coil_forces12()[len(curves):, :],
        MixedCoilTorques=CoilCoilNetTorques12(bs, bs_TF).coil_coil_torques12()[len(curves):, :],
        # NetSelfForces=bs_TF.coil_self_forces(a, b)
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
    # LENGTH_WEIGHT *= 0.01
    # JF = Jf \
    #     + CC_WEIGHT * Jccdist \
    #     + CS_WEIGHT * Jcsdist \
    #     + LINK_WEIGHT * linkNum \
    #     + LINK_WEIGHT2 * linkNum2 \
    #     + LENGTH_WEIGHT * sum(Jls_TF) 


t2 = time.time()
print('Total time = ', t2 - t1)
# btot.save("biot_savart_optimized_QA.json")

