#!/usr/bin/env python
r"""
"""

import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from simsopt.field import Current, coils_via_symmetries
from simsopt.field import regularization_rect, BiotSavart
from simsopt.field.force import LpCurveForce, \
    SquaredMeanForce, \
    SquaredMeanTorque, LpCurveTorque
from simsopt.util import calculate_on_axis_B, align_dipoles_with_plasma, \
    remove_interlinking_dipoles_and_TFs, initialize_coils, save_coil_sets
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier, create_planar_curves_between_two_toroidal_surfaces
)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty

t1 = time.time()

continuation_run = False
if continuation_run:
    MAXITER = 1000
    file_suffix = '_continuation'
else:
    MAXITER = 600
    file_suffix = ''

# Number of Fourier modes describing each Cartesian component of each coil:
order = 2

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
input_name = 'input.LandremanPaul2021_QA_reactorScale_lowres'
filename = TEST_DIR / input_name

# Initialize the boundary magnetic surface:
range_param = "half period"
nphi = 32
ntheta = 32
poff = 1.5
coff = 2.0
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
# Only need this if make self forces and TVE nonzero in the objective!
a = 0.2
b = 0.2
nturns = 100
nturns_TF = 200

# wire cross section for the dipole coils should be more like 5 cm x 5 cm
aa = 0.06
bb = 0.06

# Directory for output
OUT_DIR = ("./dipole_coils_QA/")
os.makedirs(OUT_DIR, exist_ok=True)

if not continuation_run:
    # initialize the TF coils
    base_curves_TF, curves_TF, coils_TF, currents_TF = initialize_coils(s, TEST_DIR, 'LandremanPaulQA')
    num_TF_unique_coils = len(base_curves_TF)
    base_coils_TF = coils_TF[:num_TF_unique_coils]
    currents_TF = np.array([coil.current.get_value() for coil in coils_TF])
    bs_TF = BiotSavart(coils_TF)

    Nx = 5
    Ny = Nx
    Nz = Nx
    # Create the initial coils:
    base_curves, all_curves = create_planar_curves_between_two_toroidal_surfaces(
        s, s_inner, s_outer, Nx, Ny, Nz, order=order,
    )
    # base_curves = remove_inboard_dipoles(s, base_curves, eps=0.05)
    base_curves = remove_interlinking_dipoles_and_TFs(base_curves, base_curves_TF, eps=0.05)
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
        base_curves[i].fix('q0')
        base_curves[i].fix('qi')
        base_curves[i].fix('qj')
        base_curves[i].fix('qk')

        # Fix shape of each coil (Fourier coefficients)
        for j in range(order + 1):
            base_curves[i].fix(f'rc({j})')
        for j in range(1, order + 1):
            base_curves[i].fix(f'rs({j})')
        # Fix center points of each coil
        # base_curves[i].fix('X')
        # base_curves[i].fix('Y')
        # base_curves[i].fix('Z')
    # psc_array = PSCArray(base_curves, coils_TF, eval_points, a_list, b_list, nfp=s.nfp, stellsym=s.stellsym)
    ncoils = len(base_curves)
    base_currents = [Current(1.0) * 2e7 for i in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    base_coils = coils[:ncoils]
    bs = BiotSavart(coils)
else:
    from simsopt import load
    coils = load(OUT_DIR + "psc_coils_continuation.json")
    # coils = load(OUT_DIR + "psc_coils.json")
    print('R = ', coils[0].curve.x[0])
    # coils_TF = load(OUT_DIR + "TF_coils.json")
    coils_TF = load(OUT_DIR + "TF_coils_continuation.json")
    inds = np.array([1, 2, 3, 5, 6], dtype=int)
    # inds = np.array([1, 2, 3, 4, 7, 8, 9], dtype=int)
    curves = [c.curve for c in coils]
    base_curves = curves[:len(curves) // 4]
    base_coils = coils[:len(coils) // 4]
    base_curves = [base_curves[i] for i in inds]
    curves_TF = [c.curve for c in coils_TF]
    base_curves_TF = curves_TF[:len(curves_TF) // 4]
    base_coils_TF = coils_TF[:len(coils_TF) // 4]
    # Give coils more dofs now that we have a good initial guess
    for i in range(len(base_curves)):
        # unfix orientations of each coil
        base_curves[i].unfix('q0')
        base_curves[i].unfix('qi')
        base_curves[i].unfix('qj')
        base_curves[i].unfix('qk')

        # unfix shape of each coil (Fourier coefficients)
        for j in range(order + 1):
            base_curves[i].unfix(f'rc({j})')
        for j in range(1, order + 1):
            base_curves[i].unfix(f'rs({j})')

    base_coils = [coils[i] for i in inds]
    base_currents = [c.current for c in base_coils]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    bs = BiotSavart(coils)
    bs_TF = BiotSavart(coils_TF)

ncoils = len(base_curves)
a_list = np.ones(len(base_curves)) * aa
b_list = np.ones(len(base_curves)) * aa
print('Num dipole coils = ', ncoils)
eval_points = s.gamma().reshape(-1, 3)

# Calculate average, approximate on-axis B field strength
# calculate_on_axis_B(psc_array.biot_savart_TF, s)
# psc_array.biot_savart_TF.set_points(eval_points)
# btot = psc_array.biot_savart_total
# calculate_on_axis_B(btot, s)
btot = bs + bs_TF
btot.set_points(s.gamma().reshape((-1, 3)))

# bs.set_points(s.gamma().reshape((-1, 3)))
base_coils = coils[:ncoils]
curves = [c.curve for c in coils]
currents = [c.current.get_value() for c in coils]
a_list = np.hstack((np.ones(len(coils)) * aa, np.ones(len(coils_TF)) * a))
b_list = np.hstack((np.ones(len(coils)) * bb, np.ones(len(coils_TF)) * b))

LENGTH_TARGET2 = 100
if continuation_run:
    LENGTH_TARGET = 135
    CC_THRESHOLD = 1.0
else:
    LENGTH_TARGET = 150
    CC_THRESHOLD = 0.8
LENGTH_WEIGHT = Weight(0.01)
LINK_WEIGHT = 1e4
CC_WEIGHT = 1
CS_THRESHOLD = 1.5
CS_WEIGHT = 1
# Weight for the Coil Coil forces term
FORCE_WEIGHT = Weight(1e-34)  # 1e-34 Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
FORCE_WEIGHT2 = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT2 = Weight(0.0)  # 1e-22 Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
save_coil_sets(btot, OUT_DIR, "_initial" + file_suffix, a, b, nturns_TF, aa, bb, nturns)

# Force and Torque calculations spawn a bunch of spurious BiotSavart child objects -- erase them!
for c in (coils + coils_TF):
    c._children = set()

btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_init" + file_suffix, extra_data=pointData)
btot.set_points(s.gamma().reshape((-1, 3)))

bpsc = btot.Bfields[0]
bpsc.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bpsc.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(bpsc.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(bpsc.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_PSC" + file_suffix, extra_data=pointData)
bpsc.set_points(s.gamma().reshape((-1, 3)))

# Define the individual terms objective function:
Jf = SquaredFlux(s, btot)
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
MSC_THRESHOLD = 0.06
if continuation_run:
    CURVATURE_WEIGHT = 1e-3
    MSC_WEIGHT = 1e-8
else:
    CURVATURE_WEIGHT = 1e-4
    MSC_WEIGHT = 1e-6
Jcs = [LpCurveCurvature(c.curve, 2, CURVATURE_THRESHOLD) for c in base_coils_TF]
Jmscs = [MeanSquaredCurvature(c.curve) for c in base_coils_TF]

JF = Jf \
    + CS_WEIGHT * Jcsdist \
    + CC_WEIGHT * Jccdist \
    + CC_WEIGHT * Jccdist2 \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + LINK_WEIGHT * linkNum \
    + LENGTH_WEIGHT * Jlength \
    + LENGTH_WEIGHT * Jlength2

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
    # absolutely essential line that updates the PSC currents even though they are not
    # being directly optimized.
    # psc_array.recompute_currents()
    # absolutely essential line if the PSCs do not have any dofs
    # btot.Bfields[0].invalidate_cache()
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
    BdotN = np.mean(np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    BdotN_over_B = BdotN / np.mean(btot.AbsB())
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}, ⟨B·n⟩/⟨B⟩={BdotN_over_B:.1e}"
    valuestr = f"J={J:.2e}, Jf={jf:.2e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls_TF])
    cl_string2 = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.2f}" for c in base_curves_TF)
    msc_string = ", ".join(f"{J.J():.2f}" for J in Jmscs)
    outstr += f", ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls_TF):.2f}"
    outstr += f", Len2=sum([{cl_string2}])={sum(J.J() for J in Jls):.2f}"
    valuestr += f", LenObj={length_val:.2e}, LenObj2={length_val2:.2e}"
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

res = minimize(fun, dofs, jac=True, method='L-BFGS-B',   # bounds=opt_bounds,
               options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)

bpsc = btot.Bfields[0]
bpsc.set_points(s_plot.gamma().reshape((-1, 3)))
dipole_currents = [c.current.get_value() for c in bpsc.coils]
save_coil_sets(btot, OUT_DIR, "_optimized" + file_suffix, a, b, nturns_TF, aa, bb, nturns)

btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_optimized" + file_suffix, extra_data=pointData)

btf = btot.Bfields[1]
btf.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btf.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(btf.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_TF_optimized" + file_suffix, extra_data=pointData)

bpsc.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bpsc.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(bpsc.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_PSC_optimized" + file_suffix, extra_data=pointData)

btot.set_points(s.gamma().reshape((-1, 3)))
print('Max I = ', np.max(dipole_currents))
print('Min I = ', np.min(dipole_currents))
calculate_on_axis_B(btot, s)

t2 = time.time()
print('Total time = ', t2 - t1)
from simsopt import save
save(btot.Bfields[0].coils, OUT_DIR + 'psc_coils' + file_suffix + '.json')
save(btot.Bfields[1].coils, OUT_DIR + 'TF_coils' + file_suffix + '.json')
print(OUT_DIR)
