#!/usr/bin/env python
r"""
This script demonstrates the use of the simsopt package to design passive
dipole coils jointly with TF coils, for a given plasma boundary. 

This work is based on:
A. A. Kaptanoglu, M. Landreman, and M. C. Zarnstorff, "Optimization of passive 
superconductors for shaping stellarator magnetic fields," Phys. Rev. E 111, 065202 (2025).
https://journals.aps.org/pre/abstract/10.1103/PhysRevE.111.065202

The script uses the Schuett-Henneberg QA equilibrium from the VMEC test-suite.
This equilibrium has nontrivial plasma current, so VirtualCasing is used. Note 
that B_plasma is evaluated only on the plasma surfaec, so poincare plots and 
other post-processing diagnostics cannot be used. 

The script also allows one to continue from a previous run (continuation_run = True).
"""

import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from simsopt.field import regularization_rect, PSCArray
from simsopt.field.force import LpCurveForce, \
    SquaredMeanForce, \
    LpCurveTorque, \
    SquaredMeanTorque
from simsopt.util import calculate_modB_on_major_radius, initialize_coils, remove_inboard_dipoles, \
    align_dipoles_with_plasma, save_coil_sets, in_github_actions
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier, create_planar_curves_between_two_toroidal_surfaces
)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt import load
from simsopt.mhd import VirtualCasing
import json

t1 = time.time()

# Number of Fourier modes describing each Cartesian component of each coil:
order = 2

# Continue from a previous file
continuation_run = False

if continuation_run:
    MAXITER = 1000
    file_suffix = "_continuation"
else:
    if in_github_actions:
        MAXITER = 10
        file_suffix = "_ci"
    else:
        MAXITER = 500
        file_suffix = ""

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
input_name = 'wout_schuett_henneberg_nfp2_QA.nc'
filename = TEST_DIR / input_name

# Directory for output
OUT_DIR = ("./passive_coils_QASH/")
os.makedirs(OUT_DIR, exist_ok=True)

# Initialize the boundary magnetic surface:
range_param = "half period"

# Virtual casing must not have been run yet.
if in_github_actions:
    print('Skipping virtual casing calculation (CI mode)')
    # Use lower resolution for CI
    vc_src_nphi = 32
    nphi = 32
    ntheta = 32
    # Create a dummy VirtualCasing object with zeros
    dummy_surf = SurfaceRZFourier.from_wout(filename, range=range_param, nphi=nphi, ntheta=ntheta)
    vc = VirtualCasing()
    vc.src_nphi = vc_src_nphi
    vc.src_ntheta = vc_src_nphi
    vc.trgt_nphi = nphi
    vc.trgt_ntheta = ntheta
    vc.nfp = dummy_surf.nfp
    vc.B_external_normal = np.zeros((nphi, ntheta))
    vc.trgt_surf = dummy_surf
    vc.trgt_surf_full = dummy_surf
else:
    print('Running the virtual casing calculation')
    # Resolution for the virtual casing calculation:
    vc_src_nphi = 128
    nphi = 64
    ntheta = 64
    vc = VirtualCasing.from_vmec(
        filename,
        src_nphi=vc_src_nphi, src_ntheta=vc_src_nphi,
        trgt_nphi=nphi, trgt_ntheta=ntheta,
    )

# Add these lines to save B_external_normal to JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

with open(os.path.join(OUT_DIR, 'B_external_normal.json'), 'w') as f:
    json.dump({'B_external_normal': vc.B_external_normal}, f, cls=NumpyEncoder)

# Initialize the boundary magnetic surface:
range_param = "half period"
poff = 1.5
coff = 3.0

s = SurfaceRZFourier.from_wout(filename, range=range_param, nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_wout(filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4)
s_outer = SurfaceRZFourier.from_wout(filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4)

# Make the inner and outer surfaces by extending the plasma surface
s_inner.extend_via_normal(poff)
s_outer.extend_via_normal(poff + coff)

# Make a high-res surface for plotting
qphi = nphi * 4
qtheta = ntheta * 4
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)
s_plot = SurfaceRZFourier.from_wout(
    filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

# Make another VirtualCasing object for the high-res plotting surface
if in_github_actions:
    # Create a dummy VirtualCasing object with zeros
    trgt_nphi = qphi // 4
    trgt_ntheta = qtheta
    dummy_surf2_trgt = SurfaceRZFourier.from_wout(filename, range=range_param, nphi=trgt_nphi, ntheta=trgt_ntheta)
    dummy_surf2_full = SurfaceRZFourier.from_wout(filename, range='full torus', nphi=trgt_nphi * dummy_surf2_trgt.nfp * 2, ntheta=trgt_ntheta)
    vc2 = VirtualCasing()
    vc2.src_nphi = vc_src_nphi
    vc2.src_ntheta = vc_src_nphi
    vc2.trgt_nphi = trgt_nphi
    vc2.trgt_ntheta = trgt_ntheta
    vc2.nfp = dummy_surf2_trgt.nfp
    # B_external_normal_extended shape: (trgt_nphi * nfp * 2, trgt_ntheta)
    vc2.B_external_normal_extended = np.zeros((vc2.trgt_nphi * vc2.nfp * 2, vc2.trgt_ntheta))
    vc2.trgt_surf_full = dummy_surf2_full
else:
    vc2 = VirtualCasing.from_vmec(
        filename, src_nphi=vc_src_nphi, src_ntheta=vc_src_nphi,
        trgt_nphi=qphi // 4, trgt_ntheta=qtheta)

with open(os.path.join(OUT_DIR, 'B_external_normal_extended.json'), 'w') as f:
    json.dump({'B_external_normal_extended': vc2.B_external_normal_extended}, f, cls=NumpyEncoder)

s_plot = vc2.trgt_surf_full

# initialize the coils
base_curves_TF, curves_TF, coils_TF, currents_TF = initialize_coils(s, TEST_DIR, 'SchuettHennebergQAnfp2')
num_TF_unique_coils = len(base_curves_TF)
base_coils_TF = coils_TF[:num_TF_unique_coils]
currents_TF = np.array([coil.current.get_value() for coil in coils_TF])

# wire cross section for the TF coils is a square 20 cm x 20 cm
# Only need this if make self forces and B2Energy nonzero in the objective!
a = 0.2
b = 0.2
nturns = 100
nturns_TF = 200

# wire cross section for the dipole coils should be more like 6 cm x 6 cm
aa = 0.06
bb = 0.06

if continuation_run:
    coils = load(OUT_DIR + "psc_coils.json")
    coils_TF = load(OUT_DIR + "TF_coils.json")
    curves = [c.curve for c in coils]
    base_curves = curves[:len(curves) // 4]
    base_coils = coils[:len(coils) // 4]
    curves_TF = [c.curve for c in coils_TF]
    base_curves_TF = curves_TF[:len(curves_TF) // 4]
    base_coils_TF = coils_TF[:len(coils_TF) // 4]
else:
    Nx = 4
    Ny = Nx
    Nz = Nx
    # Create the initial coils:
    base_curves, all_curves = create_planar_curves_between_two_toroidal_surfaces(
        s, s_inner, s_outer, Nx, Ny, Nz, order=order, use_jax_curve=False,
    )
    base_curves = remove_inboard_dipoles(s, base_curves, eps=0.2)
    alphas, deltas = align_dipoles_with_plasma(s, base_curves)

    for i in range(len(base_curves)):
        alpha2 = alphas[i] / 2.0
        delta2 = deltas[i] / 2.0
        calpha2 = np.cos(alpha2)
        salpha2 = np.sin(alpha2)
        cdelta2 = np.cos(delta2)
        sdelta2 = np.sin(delta2)
        # Set quaternion DOFs: q0, qi, qj, qk
        base_curves[i].set('q0', calpha2 * cdelta2)
        base_curves[i].set('qi', salpha2 * cdelta2)
        base_curves[i].set('qj', calpha2 * sdelta2)
        base_curves[i].set('qk', -salpha2 * sdelta2)
        # Fix orientations of each coil
        # base_curves[i].fix('q0')
        # base_curves[i].fix('qi')
        # base_curves[i].fix('qj')
        # base_curves[i].fix('qk')

        # Fix shape of each coil (Fourier coefficients)
        # for j in range(order + 1):
        #     base_curves[i].fix(f'rc({j})')
        # for j in range(1, order + 1):
        #     base_curves[i].fix(f'rs({j})')
        # Fix center points of each coil
        # base_curves[i].fix('X')
        # base_curves[i].fix('Y')
        # base_curves[i].fix('Z')

ncoils = len(base_curves)
a_list = np.ones(len(base_curves)) * aa
b_list = np.ones(len(base_curves)) * aa
print('Num dipole coils = ', ncoils)
print('R0 = ', base_curves[0].x[0])

# Initialize the PSCArray object
eval_points = s.gamma().reshape(-1, 3)
psc_array = PSCArray(
    base_curves, 
    coils_TF, 
    eval_points,
    a_list, 
    b_list, 
    nfp=s.nfp, 
    stellsym=s.stellsym
)

# Calculate average, approximate on-axis B field strength
calculate_modB_on_major_radius(psc_array.biot_savart_TF, s)
psc_array.biot_savart_TF.set_points(eval_points)
btot = psc_array.biot_savart_total
calculate_modB_on_major_radius(btot, s)
btot.set_points(s.gamma().reshape((-1, 3)))
coils = psc_array.coils
base_coils = coils[:ncoils]
curves = [c.curve for c in coils]
currents = [c.current.get_value() for c in coils]
a_list = np.hstack((np.ones(len(coils)) * aa, np.ones(len(coils_TF)) * a))
b_list = np.hstack((np.ones(len(coils)) * bb, np.ones(len(coils_TF)) * b))

# Set weights and thresholds for the optimization
if continuation_run:
    LENGTH_TARGET = 80
else:
    LENGTH_TARGET = 90
LENGTH_TARGET2 = 80
LENGTH_WEIGHT = Weight(0.01)
LINK_WEIGHT = 1e4
CC_THRESHOLD = 0.8
CC_WEIGHT = 1e2
CS_THRESHOLD = 1.3
CS_WEIGHT = 1e2
# Weight for the Coil Coil forces term
FORCE_WEIGHT = Weight(0.0)  # 1e-34 Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
FORCE_WEIGHT2 = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT2 = Weight(0.0)  # 1e-22 Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons

# Save the initial coils and the initial B_external_normal
save_coil_sets(btot, OUT_DIR, "_initial" + file_suffix)
btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {
    "B_N1": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2))[:, :, None],
    "B_N2": (vc2.B_external_normal_extended)[:, :, None],
    "B_N": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2) - vc2.B_external_normal_extended)[:, :, None],
    "B_N / B": ((np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                        ) - vc2.B_external_normal_extended) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_initial" + file_suffix, extra_data=pointData)
btot.set_points(s.gamma().reshape((-1, 3)))
bpsc = btot.Bfields[0]
bpsc.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bpsc.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(bpsc.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(bpsc.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_PSC" + file_suffix, extra_data=pointData)
bpsc.set_points(s.gamma().reshape((-1, 3)))

# Define the individual terms objective function:
Jf = SquaredFlux(s, btot, target=vc.B_external_normal)

# Separate length penalties on the dipole coils and the TF coils
# since they have very different sizes
Jls = [CurveLength(c) for c in base_curves]
Jls_TF = [CurveLength(c) for c in base_curves_TF]
Jlength = QuadraticPenalty(sum(Jls_TF), LENGTH_TARGET, "max")
Jlength2 = QuadraticPenalty(sum(Jls), LENGTH_TARGET2, "max")

# coil-coil and coil-plasma distances should be between all coils
Jccdist = CurveCurveDistance(curves + curves_TF, CC_THRESHOLD / 2.0, num_basecurves=len(coils + coils_TF))
Jccdist2 = CurveCurveDistance(curves_TF, CC_THRESHOLD, num_basecurves=len(coils_TF))
Jcsdist = CurveSurfaceDistance(curves + curves_TF, s, CS_THRESHOLD)

# While the coil array is not moving around, they cannot interlink each other
linkNum = LinkingNumber(curves + curves_TF, downsample=2)

# Passive MUST be passed in the psc_array argument to the force and
# torque terms for the Jacobian to be correct!
all_base_coils = base_coils + base_coils_TF
other_coils = [c for c in coils + coils_TF if c not in all_base_coils]
regularization_list = [regularization_rect(aa, bb) for _ in base_coils] + \
    [regularization_rect(a, b) for _ in base_coils_TF]
Jforce = LpCurveForce(all_base_coils, other_coils,
                      regularization_list,
                      p=4, downsample=2,
                      psc_array=psc_array
                      )
Jforce2 = SquaredMeanForce(all_base_coils, other_coils,
                           psc_array=psc_array,
                           downsample=2
                           )
Jtorque = LpCurveTorque(all_base_coils, other_coils,
                        regularization_list,
                        p=2, downsample=2,
                        psc_array=psc_array
                        )
Jtorque2 = SquaredMeanTorque(all_base_coils, other_coils,
                             psc_array=psc_array,
                             downsample=2
                             )

if continuation_run:
    CURVATURE_WEIGHT = 1e-1
    MSC_WEIGHT = 1e-5
else:
    CURVATURE_WEIGHT = 1e-2
    MSC_WEIGHT = 1e-4
CURVATURE_THRESHOLD = 0.5
MSC_THRESHOLD = 0.05
Jcs = [LpCurveCurvature(c.curve, 2, CURVATURE_THRESHOLD) for c in base_coils_TF]
Jmscs = [MeanSquaredCurvature(c.curve) for c in base_coils_TF]

# Note that only Jf and the forces/torques depend on the PSC currents,
# which is the tricky part of the Jacobian
JF = Jf \
    + CS_WEIGHT * Jcsdist \
    + CC_WEIGHT * Jccdist \
    + CC_WEIGHT * Jccdist2 \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + LINK_WEIGHT * linkNum \
    + LENGTH_WEIGHT * Jlength \
    + LENGTH_WEIGHT * Jlength2

if FORCE_WEIGHT.value > 0.0:
    JF += FORCE_WEIGHT.value * Jforce 

if FORCE_WEIGHT2.value > 0.0:
    JF += FORCE_WEIGHT2.value * Jforce2 

if TORQUE_WEIGHT.value > 0.0:
    JF += TORQUE_WEIGHT * Jtorque

if TORQUE_WEIGHT2.value > 0.0:
    JF += TORQUE_WEIGHT2 * Jtorque2

# print(JF.dof_names)


def fun(dofs):
    JF.x = dofs
    # absolutely essential line that updates the PSC currents 
    # even though they are not being directly optimized.
    psc_array.recompute_currents()

    # absolutely essential line if the PSCs do not have any dofs
    btot.Bfields[0].invalidate_cache()

    # Begin normal calculations and print output
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    length_val = LENGTH_WEIGHT.value * Jlength.J()
    length_val2 = LENGTH_WEIGHT.value * Jlength2.J()
    cc_val = CC_WEIGHT * (Jccdist.J() + Jccdist2.J())
    cs_val = CS_WEIGHT * Jcsdist.J()
    link_val = LINK_WEIGHT * linkNum.J()
    forces_val = FORCE_WEIGHT.value * Jforce.J()
    forces_val2 = FORCE_WEIGHT2.value * Jforce2.J()
    torques_val = TORQUE_WEIGHT.value * Jtorque.J()
    torques_val2 = TORQUE_WEIGHT2.value * Jtorque2.J()
    absBn = np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2) - vc.B_external_normal)
    BdotN = np.mean(absBn)
    BdotN_over_B = np.mean(absBn) / np.mean(btot.AbsB())
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}, ⟨B·n⟩/⟨B⟩={BdotN_over_B:.1e}"
    valuestr = f"J={J:.2e}, Jf={jf:.2e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls_TF])
    cl_string2 = ", ".join([f"{J.J():.2f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.2f}" for c in base_curves_TF)
    msc_string = ", ".join(f"{J.J():.2f}" for J in Jmscs)
    outstr += f", ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls_TF):.2f}"
    outstr += f", Len=sum([{cl_string2}])={sum(J.J() for J in Jls):.2f}"
    valuestr += f", LenObj={length_val:.2e}"
    valuestr += f", LenObj2={length_val2:.2e}"
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
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    print("err", (J1-J2)/(2*eps) - dJh)
    print((J1-J2)/(2*eps), dJh)

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")

# Perform the minimization 
res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
               options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)

# Save the optimized coils and the optimized B_external_normal
bpsc = btot.Bfields[0]
bpsc.set_points(s_plot.gamma().reshape((-1, 3)))
dipole_currents = [c.current.get_value() for c in bpsc.coils]
psc_array.recompute_currents()
save_coil_sets(btot, OUT_DIR, "_optimized" + file_suffix)
btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {
    "B_N1": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2))[:, :, None],
    "B_N2": (vc2.B_external_normal_extended)[:, :, None],
    "B_N": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2) - vc2.B_external_normal_extended)[:, :, None],
    "B_N / B": ((np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                        ) - vc2.B_external_normal_extended) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_final" + file_suffix, extra_data=pointData)
btf = btot.Bfields[1]
btf.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btf.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(btf.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btf.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_TF" + file_suffix, extra_data=pointData)

bpsc.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bpsc.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(bpsc.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(bpsc.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_PSC" + file_suffix, extra_data=pointData)

# Print the maximum and minimum dipole currents
btot.set_points(s.gamma().reshape((-1, 3)))
print('Max I = ', np.max(dipole_currents))
print('Min I = ', np.min(dipole_currents))
calculate_modB_on_major_radius(btot, s)

# Print the total time taken
t2 = time.time()
print('Total time = ', t2 - t1)

# Save the optimized coils
from simsopt import save
save(btot.Bfields[0].coils, OUT_DIR + 'psc_coils' + file_suffix + '.json')
save(btot.Bfields[1].coils, OUT_DIR + 'TF_coils' + file_suffix + '.json')
print(OUT_DIR)
