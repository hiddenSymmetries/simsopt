"""
This script illustrates Jake Halpbern's method to initialize a dipole array
conformal to a winding surface. It allows one to specify lots of geometrical
parameters. 
"""
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.field.force import LpCurveForce, \
    SquaredMeanForce, \
    LpCurveTorque, \
    SquaredMeanTorque, \
    regularization_rect
from simsopt.util import calculate_modB_on_major_radius, save_coil_sets, \
    generate_curves
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier
)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
import os
import time

outdir = './dipole_array_tutorial/'
os.makedirs(outdir, exist_ok=True)
nphi = 32
ntheta = 32
qphi = 2 * nphi
qtheta = 2 * ntheta
# load in a sample hybrid torus equilibria
filename = 'input.LandremanPaul2021_QA_reactorScale_lowres'
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
s = SurfaceRZFourier.from_vmec_input(TEST_DIR / filename, range='half period', nphi=nphi, ntheta=ntheta)
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)
s_plot = SurfaceRZFourier.from_vmec_input(
    TEST_DIR / filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)
# create a vacuum vessel to place coils on with sufficient plasma-coil distance
VV = SurfaceRZFourier.from_vmec_input(
    TEST_DIR / filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)
VV.extend_via_projected_normal(1.75)
base_wp_curves, base_tf_curves = generate_curves(s, VV, outdir=outdir)

# wire cross section for the TF coils is a square 25 cm x 25 cm
# Only need this if make self forces and B2Energy nonzero in the objective!
a = 0.25
b = 0.25
nturns = 100
nturns_TF = 200

# wire cross section for the dipole coils should be at least 10 cm x 10 cm
aa = 0.1
bb = 0.1
total_current = 70000000
print('Total current = ', total_current)

ncoils_TF = len(base_tf_curves)
base_currents_TF = [(Current(total_current / ncoils_TF * 1e-7) * 1e7) for _ in range(ncoils_TF - 1)]
total_current = Current(total_current)
total_current.fix_all()
base_currents_TF += [total_current - sum(base_currents_TF)]
# Use rectangular regularization for force/torque calculations
regularization_TF = regularization_rect(a, b)
regularizations_TF = [regularization_TF for _ in range(ncoils_TF)]
coils_TF = coils_via_symmetries(base_tf_curves, base_currents_TF, s.nfp, True, regularizations=regularizations_TF)
base_coils_TF = coils_TF[:ncoils_TF]
curves_TF = [c.curve for c in coils_TF]
bs_TF = BiotSavart(coils_TF)
# Finished initializing the TF coils

ncoils = len(base_wp_curves)

# Fix the window pane curve dofs
[c.fix_all() for c in base_wp_curves]
base_wp_currents = [Current(1.0) * 1e6 for i in range(ncoils)]
# Fix currents in each coil
# for i in range(ncoils):
#     base_currents[i].fix_all()

# Use rectangular regularization for force/torque calculations
regularization_wp = regularization_rect(aa, bb)
regularizations_wp = [regularization_wp for _ in range(ncoils)]
coils = coils_via_symmetries(base_wp_curves, base_wp_currents, s.nfp, True, regularizations=regularizations_wp)
base_coils = coils[:ncoils]

bs = BiotSavart(coils)
btot = bs + bs_TF

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
LENGTH_TARGET = 120
LINK_WEIGHT = 1e3

# Weight for the Coil Coil forces term
FORCE_WEIGHT = Weight(0.0)  # 1e-34 Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
FORCE_WEIGHT2 = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT2 = Weight(0.0)  # 1e-22 Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons

save_coil_sets(btot, outdir, "_initial")
pointData = {
    "B_N_dipoles": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
    "B_N_TF": np.sum(bs_TF.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
    "B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
    "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                       ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(outdir + "surf_initial", extra_data=pointData)
btot.set_points(s.gamma().reshape((-1, 3)))

# Define the individual terms objective function:
Jf = SquaredFlux(s, btot)
# Separate length penalties on the dipole coils and the TF coils
# since they have very different sizes
# Jls = [CurveLength(c) for c in base_curves]
Jls_TF = [CurveLength(c) for c in base_tf_curves]
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
Jforce = LpCurveForce(all_base_coils, all_coils, regularization_list, p=4, threshold=1e3, downsample=2)
Jforce2 = SquaredMeanForce(all_base_coils, all_coils, downsample=2)
Jtorque = LpCurveTorque(all_base_coils, all_coils, regularization_list, p=2, threshold=1e3, downsample=2)
Jtorque2 = SquaredMeanTorque(all_base_coils, all_coils, downsample=2)

CURVATURE_THRESHOLD = 0.5
MSC_THRESHOLD = 0.05
CURVATURE_WEIGHT = 1e-4
MSC_WEIGHT = 1e-5
Jcs = [LpCurveCurvature(c.curve, 2, CURVATURE_THRESHOLD) for c in base_coils_TF]
Jmscs = [MeanSquaredCurvature(c.curve) for c in base_coils_TF]

JF = Jf \
    + CC_WEIGHT * Jccdist \
    + CC_WEIGHT * Jccdist2 \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + LENGTH_WEIGHT * Jlength
# + LINK_WEIGHT * linkNum \

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
    kap_string = ", ".join(f"{np.max(c.kappa()):.2f}" for c in base_tf_curves)
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


print(JF.dof_names)

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

MAXITER = 1000
res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
               options={'maxiter': MAXITER, 'maxcor': 500}, tol=1e-10)
save_coil_sets(btot, outdir, "_optimized")
pointData = {
    "B_N_dipoles": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
    "B_N_TF": np.sum(bs_TF.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
    "B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
    "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                       ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(outdir + "surf_optimized", extra_data=pointData)

btot.set_points(s.gamma().reshape((-1, 3)))
calculate_modB_on_major_radius(btot, s)

t2 = time.time()
print('Total time = ', t2 - t1)
btot.save(outdir + "biot_savart_optimized" + ".json")
print(outdir)
