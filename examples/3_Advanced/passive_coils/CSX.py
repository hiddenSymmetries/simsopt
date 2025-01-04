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
from simsopt.util import calculate_on_axis_B, make_Bnormal_plots
from simsopt.geo import (
    CurveLength, CurveCurveDistance, create_equally_spaced_curves,
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier, curves_to_vtk, create_planar_curves_between_two_toroidal_surfaces
)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt import load

t1 = time.time()

# Number of Fourier modes describing each Cartesian component of each coil:
order = 0

# Directory for output
OUT_DIR = ("./CSX/")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
input_name = 'wout_csx_wps_5.0.nc'
filename = TEST_DIR / input_name

# Initialize the boundary magnetic surface:
range_param = "half period"
nphi = 32
ntheta = 32
poff = 0.1
coff = 0.3
s = SurfaceRZFourier.from_wout(filename, range=range_param, nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_wout(filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4)
s_outer = SurfaceRZFourier.from_wout(filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4)

# Make the inner and outer surfaces by extending the plasma surface
s_inner.extend_via_projected_normal(poff)
s_outer.extend_via_projected_normal(poff + coff)

qphi = nphi * 2
qtheta = ntheta * 2
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)

# Make high resolution, full torus version of the plasma boundary for plotting
s_plot = SurfaceRZFourier.from_wout(
    filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

# initialize the TF coils
# configuration_name = "CSX_5.0_WPs"
bsurf = load(os.path.join(TEST_DIR / "boozer_surface_CSX_5.0_wps.json"))
# bsurf = load(os.path.join(TEST_DIR / "boozer_surface_CSX_4.5.json"))
coils_TF = bsurf.biotsavart._coils
print(len(coils_TF), coils_TF)
# Plot original coils
# iota = -0.3
current_sum = sum(abs(c.current.get_value()) for c in bsurf.biotsavart.coils[0:2])
G0 = -2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
# res = bsurf.run_code(iota, G0)
volume = bsurf.surface.volume()
aspect = bsurf.surface.aspect_ratio()
rmaj = bsurf.surface.major_radius()
rmin = bsurf.surface.minor_radius()
L = float(CurveLength(bsurf.biotsavart.coils[0].curve).J())
# QS = float(NonQuasiSymmetricRatio(bsurf, bsurf.biotsavart).J())
il_current = bsurf.biotsavart.coils[0].current.get_value()
pf_current = bsurf.biotsavart.coils[2].current.get_value()
# iota = res['iota']

from matplotlib import pyplot as plt
ax = plt.figure().add_subplot(projection='3d')
s.plot(ax=ax, show=False, close=True)
for c in bsurf.biotsavart.coils:
    c.curve.plot(ax=ax, show=False)
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_aspect('equal')

# Plot Bn / B errors
theta = s.quadpoints_theta
phi = s.quadpoints_phi
ntheta = theta.size
nphi = phi.size
bsurf.biotsavart.set_points(s.gamma().reshape((-1, 3)))
Bdotn = np.sum(bsurf.biotsavart.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
modB = bsurf.biotsavart.AbsB().reshape((nphi, ntheta))
fig, ax = plt.subplots()
c = ax.contourf(theta, phi, Bdotn / modB)
plt.colorbar(c)
ax.set_title(r'$\mathbf{B}\cdot\hat{n} / |B|$ ')
ax.set_ylabel(r'$\phi$')
ax.set_xlabel(r'$\theta$')
plt.tight_layout()
# plt.show()

# Plot the original coils that Antoine used
make_Bnormal_plots(bsurf.biotsavart, s_plot, OUT_DIR, "biot_savart_with_original_window_panes")
curves_to_vtk([c.curve for c in coils_TF], OUT_DIR + "curves_with_original_window_panes")

# Subtract out the window-pane coils used in Antoines paper
# coils_TF = coils_TF[:4]

# Need to get all the coils with same number of quadpoints
from simsopt.geo import CurveXYZFourier, RotatedCurve
from simsopt.field import Coil
curves_TF = [coil.curve for coil in coils_TF]
print(curves_TF)
print(coils_TF[0].curve.order)
print(curves_TF[0].dof_names)
c1 = CurveXYZFourier(200, 15)
for name in c1.local_dof_names:
    if name in curves_TF[0].local_dof_names:
        continue
    else:
        c1.fix(name)
print('c1 names = ', c1.dof_names)
c1.x = coils_TF[0].curve.x
# c1.set_dofs(coils_TF[0].curve.x)
c2 = RotatedCurve(c1, np.pi, True)
coil1 = Coil(c1, coils_TF[0].current)
coil2 = Coil(c2, coils_TF[1].current)
coils_TF = [coil1, coil2, coils_TF[2], coils_TF[3]]
currents_TF = np.array([coil.current.get_value() for coil in coils_TF])
curves_TF = [coil.curve for coil in coils_TF]
base_curves_TF = [curves_TF[0]] + [curves_TF[2]]
num_TF_unique_coils = len(base_curves_TF)
base_coils_TF = [coils_TF[0]] + [coils_TF[2]]
currents_TF = np.array([coil.current.get_value() for coil in coils_TF])
print(currents_TF)
print([len(coil.curve.quadpoints) for coil in coils_TF])
curves_to_vtk([c.curve for c in coils_TF], OUT_DIR + "curves_fixed_quadpoints")

# wire cross section for the TF coils is a square 20 cm x 20 cm
# Only need this if make self forces and TVE nonzero in the objective!
a = 0.2
b = 0.2
nturns_TF = 200

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

biot_savart_TF = BiotSavart(coils_TF)
biot_savart_TF.set_points(s.gamma().reshape((-1, 3)))
# Calculate average, approximate on-axis B field strength
calculate_on_axis_B(biot_savart_TF, s)
biot_savart_TF.set_points(s.gamma().reshape((-1, 3)))
btot = biot_savart_TF
calculate_on_axis_B(btot, s)
btot.set_points(s.gamma().reshape((-1, 3)))

LENGTH_WEIGHT = Weight(0.01)
LENGTH_TARGET = 5
LINK_WEIGHT = 1e4
CC_THRESHOLD = 0.05
CC_WEIGHT = 1
CS_THRESHOLD = 0.05
CS_WEIGHT = 1
curves_to_vtk(
    [c.curve for c in btot.coils],
    OUT_DIR + "curves_TF_0",
    close=True,
    extra_point_data=pointData_forces_torques(coils_TF, coils_TF, np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b, np.ones(len(coils_TF)) * nturns_TF),
    I=[c.current.get_value() for c in btot.coils],
    NetForces=coil_net_forces(coils_TF, coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF),
    NetTorques=coil_net_torques(coils_TF, coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF)
)
# Force and Torque calculations spawn a bunch of spurious BiotSavart child objects -- erase them!
for c in (coils_TF):
    c._children = set()

btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
    "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                    ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_full_init", extra_data=pointData)
btot.set_points(s.gamma().reshape((-1, 3)))

# Define the individual terms objective function:
Jf = SquaredFlux(s, btot)
# Separate length penalties on the dipole coils and the TF coils
# since they have very different sizes
Jls_TF = [CurveLength(base_curves_TF[0])]
Jlength = QuadraticPenalty(sum(Jls_TF), LENGTH_TARGET, "max")

# coil-coil and coil-plasma distances should be between all coils
Jccdist = CurveCurveDistance(curves_TF, CC_THRESHOLD / 2.0, num_basecurves=len(coils_TF))
Jccdist2 = CurveCurveDistance(curves_TF, CC_THRESHOLD, num_basecurves=len(coils_TF))
Jcsdist = CurveSurfaceDistance(curves_TF, s, CS_THRESHOLD)

# While the coil array is not moving around, they cannot
# interlink.
linkNum = LinkingNumber(curves_TF, downsample=2)

# Currently, all force terms involve all the coils
all_coils = coils_TF
all_base_coils = base_coils_TF

CURVATURE_THRESHOLD = 5
MSC_THRESHOLD = 0.5
CURVATURE_WEIGHT = 1e-7
MSC_WEIGHT = 1e-8
Jcs = [LpCurveCurvature(c.curve, 2, CURVATURE_THRESHOLD) for c in base_coils_TF]
Jmscs = [MeanSquaredCurvature(c.curve) for c in base_coils_TF]

JF = Jf \
    + CS_WEIGHT * Jcsdist \
    + CC_WEIGHT * Jccdist \
    + CC_WEIGHT * Jccdist2 \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + LINK_WEIGHT * linkNum  \
    + LENGTH_WEIGHT * Jlength 

print(JF.dof_names)
# for i in range(len(JF.dof_names) - len(opt_bounds)):
#     opt_bounds.append((None, None))
# print(opt_bounds)
# print(opt_bounds, np.shape(opt_bounds), np.shape(JF.dof_names))
# exit()

def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ() 
    jf = Jf.J()
    length_val = LENGTH_WEIGHT.value * Jlength.J()
    cc_val = CC_WEIGHT * (Jccdist.J() + Jccdist2.J())
    cs_val = CS_WEIGHT * Jcsdist.J()
    link_val = LINK_WEIGHT * linkNum.J()
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
MAXITER = 1000
for i in range(1, n_saves + 1):
    print('Iteration ' + str(i) + ' / ' + str(n_saves))
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B',   # bounds=opt_bounds,
                   options={'maxiter': MAXITER, 'maxcor': 500}, tol=1e-15)
    # dofs = res.x
    curves_to_vtk(
        [c.curve for c in btot.coils],
        OUT_DIR + "curves_TF_{0:d}".format(i),
        close=True,
        extra_point_data=pointData_forces_torques(coils_TF, coils_TF, np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b, np.ones(len(coils_TF)) * nturns_TF),
        I=[c.current.get_value() for c in btot.coils],
        NetForces=coil_net_forces(coils_TF, coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF),
        NetTorques=coil_net_torques(coils_TF, coils_TF, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF),
    )

    btot.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
        "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                    ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "surf_full_final", extra_data=pointData)

    btot.set_points(s.gamma().reshape((-1, 3)))
    calculate_on_axis_B(btot, s)

t2 = time.time()
print('Total time = ', t2 - t1)
print(OUT_DIR)
