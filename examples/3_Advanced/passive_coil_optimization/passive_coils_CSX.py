#!/usr/bin/env python
r"""
This script demonstrates the use of the simsopt package to design passive coils
for a given plasma boundary. 

The script uses the CSX plasma boundary obtained using the window panes (with best quasisymmetry reported in the paper)
from Baillod et al. 2025 (https://iopscience.iop.org/article/10.1088/1741-4326/ada6dd/meta).

Then it removes those window panes coils, and jointly optimizes the interlinking coils + 
a single passive coil per half field period. 

You will find that with the initial settings, a few of the passive coils will shrink to zero. They
are allowed to change their shapes but are restricted to circular shape, so they can only change their radii.

Aftering running the first optimization,
rerunning and setting continuation_run = True will allow one to throw out these small and irrelevant coils from the 
first optimization problem, and continue optimizing the remaining coils.
"""

import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from simsopt.field import regularization_rect, PSCArray
from simsopt.field.force import LpCurveForce, SquaredMeanForce, \
    SquaredMeanTorque, LpCurveTorque
from simsopt.util import calculate_modB_on_major_radius, make_Bnormal_plots, in_github_actions
from simsopt.geo import (
    CurveLength, CurveCurveDistance, MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier, curves_to_vtk, create_planar_curves_between_two_toroidal_surfaces
)
from simsopt.util import remove_inboard_dipoles, align_dipoles_with_plasma, save_coil_sets
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt import load

t1 = time.time()

continuation_run = False
nphi = 32
ntheta = 32
if continuation_run:
    file_suffix = "_continuation"
    MAXITER = 4000
else:
    file_suffix = ""
    MAXITER = 200

# Set some parameters -- if doing CI, lower the resolution
# but resolution must be large enough so that at least one dipole coil is initialized
if in_github_actions:
    MAXITER = 10
    nphi = 8
    ntheta = 8

# Directory for output
OUT_DIR = ("./passive_coils_CSX/")
os.makedirs(OUT_DIR, exist_ok=True)

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()

# Get the plasma surface corresponding to the CSX plasma obtained using the
# window panes (with best quasisymmetry reported in the paper)
input_name = 'wout_csx_wps_5.0.nc'
filename = TEST_DIR / input_name

# Initialize the boundary magnetic surface:
range_param = "half period"
poff = 0.1
coff = 0.05
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

# initialize the TF coils from json file -- going to reoptimize the interlinking coils anyways
bsurf = load(os.path.join(TEST_DIR / "boozer_surface_CSX_4.5.json"))
coils_TF = bsurf.biotsavart._coils

# Plot original coils
plot_coils = False
if plot_coils:
    current_sum = sum(abs(c.current.get_value()) for c in bsurf.biotsavart.coils[0:2])
    G0 = -2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
    volume = bsurf.surface.volume()
    aspect = bsurf.surface.aspect_ratio()
    rmaj = bsurf.surface.major_radius()
    rmin = bsurf.surface.minor_radius()
    L = float(CurveLength(bsurf.biotsavart.coils[0].curve).J())
    from matplotlib import pyplot as plt
    ax = plt.figure().add_subplot(projection='3d')
    s.plot(ax=ax, show=False, close=True)
    for c in bsurf.biotsavart.coils:
        c.curve.plot(ax=ax, show=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
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
    plt.show()

# Plot the original coils that Antoine Baillod used
make_Bnormal_plots(bsurf.biotsavart, s_plot, OUT_DIR, "biot_savart_with_original_window_panes")
curves_to_vtk([c.curve for c in coils_TF], OUT_DIR + "curves_with_original_window_panes")

# Subtract out the window-pane coils used in Antoines paper
# Need to reinitialize all the coils with same number of quadpoints
from simsopt.geo import CurveXYZFourier, RotatedCurve
from simsopt.field import Coil
curves_TF = [coil.curve for coil in coils_TF]
c1 = CurveXYZFourier(200, 15)
for name in c1.local_dof_names:
    if name in curves_TF[0].local_dof_names:
        continue
    else:
        c1.fix(name)
c1.x = coils_TF[0].curve.x
c2 = RotatedCurve(c1, np.pi, True)
coil1 = Coil(c1, coils_TF[0].current)
coil2 = Coil(c2, coils_TF[1].current)
coils_TF = [coil1, coil2, coils_TF[2], coils_TF[3]]
currents_TF = np.array([coil.current.get_value() for coil in coils_TF])
curves_TF = [coil.curve for coil in coils_TF]
base_curves_TF = [curves_TF[0]] + [curves_TF[2]]  # remove the window panes
num_TF_unique_coils = len(base_curves_TF)
base_coils_TF = [coils_TF[0]] + [coils_TF[2]]
currents_TF = np.array([coil.current.get_value() for coil in coils_TF])
curves_to_vtk([c.curve for c in coils_TF], OUT_DIR + "curves_without_original_window_panes")

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
    base_curves_TF = [curves_TF[0]] + [curves_TF[2]]
    base_coils_TF = [coils_TF[0]] + [coils_TF[2]]
    ncoils = len(base_curves)
    a_list = np.ones(len(base_curves)) * aa
    b_list = np.ones(len(base_curves)) * aa
    print(base_curves[0].order)

    # Initialize the PSCArray object
    eval_points = s.gamma().reshape(-1, 3)
    psc_array = PSCArray(base_curves, coils_TF, eval_points, a_list, b_list, nfp=s.nfp, stellsym=s.stellsym)
    coils = psc_array.coils
    coils_TF = psc_array.coils_TF
    print([c.current.get_value() for c in coils])

    # Remove the coils that are not contributing much
    coils = [c for c in coils if np.abs(c.curve.x[0]) > 0.065 * 3]
    curves = [c.curve for c in coils]
    base_curves = curves[:len(curves) // 4]
    base_coils = coils[:len(coils) // 4]
    curves_TF = [c.curve for c in coils_TF]

    # need to give the passive coil the same # of quadrature point as the interlinking coil
    # for various calculations (like coil-coil forces) to work easily
    # Also make it higher order so we can do shape optimization.
    from simsopt.geo import CurvePlanarFourier
    curves_TF = [coil.curve for coil in coils_TF]
    order = 20
    c_new = CurvePlanarFourier(300, order, 2, True)
    dofs = np.zeros(2 * order + 8)
    dofs[0] = base_curves[0].x[0]
    dofs[-7:] = base_curves[0].x[1:]
    c_new.set_dofs(dofs)
    base_curves = [c_new]
    base_curves_TF = [curves_TF[0]] + [curves_TF[2]]
    base_coils_TF = [coils_TF[0]] + [coils_TF[2]]
    [c.current.unfix_all() for c in base_coils_TF]

    # Reinitialize the PSCArray object with one less passive coil after the truncation above
    psc_array = PSCArray(base_curves, coils_TF, eval_points, a_list, b_list, nfp=s.nfp, stellsym=s.stellsym)
    coils = psc_array.coils
    coils_TF = psc_array.coils_TF
    curves = [c.curve for c in coils]
    curves_TF = [coil.curve for coil in coils_TF]
    base_coils_TF = [coils_TF[0]] + [coils_TF[2]]
    base_curves_TF = [curves_TF[0]] + [curves_TF[2]]
    base_curves = curves[:len(curves) // 4]
else:
    Nx = 4
    Ny = Nx
    Nz = Nx
    order = 0
    # Create the initial coils:
    base_curves, all_curves = create_planar_curves_between_two_toroidal_surfaces(
        s, s_inner, s_outer, Nx, Ny, Nz, order=order, use_jax_curve=False,
    )

    # Remove all the coils on the inboard side!
    base_curves = remove_inboard_dipoles(s, base_curves, eps=0.2)

    # Initialize so coils are oriented so that their normal points
    # at the nearest quadrature point on the plasma surface
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

# Initialize the PSCArray object
eval_points = s.gamma().reshape(-1, 3)
psc_array = PSCArray(base_curves, coils_TF, eval_points, a_list, b_list, nfp=s.nfp, stellsym=s.stellsym)

# Calculate average, approximate on-axis B field strength
calculate_modB_on_major_radius(psc_array.biot_savart_TF, s)
psc_array.biot_savart_TF.set_points(eval_points)
btot = psc_array.biot_savart_total
calculate_modB_on_major_radius(btot, s)
btot.set_points(s.gamma().reshape((-1, 3)))

# bs.set_points(s.gamma().reshape((-1, 3)))
coils = psc_array.coils
base_coils = coils[:ncoils]
curves = [c.curve for c in coils]
base_curves = curves[:ncoils]
currents = [c.current.get_value() for c in coils]
a_list = np.hstack((np.ones(len(coils)) * aa, np.ones(len(coils_TF)) * a))
b_list = np.hstack((np.ones(len(coils)) * bb, np.ones(len(coils_TF)) * b))

LENGTH_WEIGHT = Weight(0.01)
CURVATURE_THRESHOLD = 5
MSC_THRESHOLD = 0.5
CURVATURE_WEIGHT = 1e-6
MSC_WEIGHT = 1e-7
if continuation_run:
    LENGTH_TARGET = 5.0
    LENGTH_TARGET2 = 3.0
    LINK_WEIGHT = 1e4
    CC_THRESHOLD = 0.05
    CC_WEIGHT = 1e2
    CS_THRESHOLD = 0.05
    CS_WEIGHT = 1e1
else:
    LENGTH_TARGET = 4.5
    LENGTH_TARGET2 = 4.0
    LINK_WEIGHT = 1e4
    CC_THRESHOLD = 0.1
    CC_WEIGHT = 1
    CS_THRESHOLD = 0.1
    CS_WEIGHT = 1

# Weight for the Coil Coil forces term
FORCE_WEIGHT = Weight(0.0)  # 1e-34 Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
FORCE_WEIGHT2 = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT = Weight(0.0)  # Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
TORQUE_WEIGHT2 = Weight(0.0)  # 1e-22 Forces are in Newtons, and typical values are ~10^5, 10^6 Newtons
save_coil_sets(btot, OUT_DIR, "_initial" + file_suffix)
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
Jls_TF = [CurveLength(base_curves_TF[0])]
Jlength = QuadraticPenalty(sum(Jls_TF), LENGTH_TARGET, "max")
Jlength2 = QuadraticPenalty(sum(Jls), LENGTH_TARGET2, "max")

# coil-coil and coil-plasma distances should be between all coils
Jccdist = CurveCurveDistance(curves + curves_TF, CC_THRESHOLD / 2.0, num_basecurves=len(coils + coils_TF))
Jccdist2 = CurveCurveDistance(curves_TF, CC_THRESHOLD, num_basecurves=len(coils_TF))
Jcsdist = CurveSurfaceDistance(curves + curves_TF, s, CS_THRESHOLD)

# While the coil array is not moving around, they cannot
# interlink.
linkNum = LinkingNumber(curves + curves_TF, downsample=2)

# Passive MUST be passed in the psc_array argument for the Jacobian to be correct!
all_base_coils = base_coils + base_coils_TF
other_coils = [c for c in coils + coils_TF if c not in all_base_coils]  # all other coils
regularization_list = [regularization_rect(aa, bb) for _ in base_coils] + [regularization_rect(a, b) for _ in base_coils_TF]
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
    Jcs = [LpCurveCurvature(c.curve, 2, CURVATURE_THRESHOLD) for c in all_base_coils]
    Jmscs = [MeanSquaredCurvature(c.curve) for c in all_base_coils]
else:
    Jcs = [LpCurveCurvature(c.curve, 2, CURVATURE_THRESHOLD) for c in base_coils_TF]
    Jmscs = [MeanSquaredCurvature(c.curve) for c in base_coils_TF]

# Note that only Jf and the forces/torques depend on the PSC currents,
# which is the tricky part of the Jacobian
JF = Jf \
    + CS_WEIGHT * Jcsdist \
    + CC_WEIGHT * Jccdist \
    + CC_WEIGHT * Jccdist2 \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + LINK_WEIGHT * linkNum  \
    + LENGTH_WEIGHT * Jlength  \
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
    psc_array.recompute_currents()
    # absolutely essential line if the PSCs do not have any dofs
    btot.Bfields[0].invalidate_cache()
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
    cl_string2 = ", ".join([f"{J.J():.2f}" for J in Jls])
    cl_string = ", ".join([f"{J.J():.2f}" for J in Jls_TF])
    kap_string = ", ".join(f"{np.max(c.kappa()):.2f}" for c in base_curves_TF)
    msc_string = ", ".join(f"{J.J():.2f}" for J in Jmscs)
    outstr += f", ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", Len=sum([{cl_string2}])={sum(J.J() for J in Jls):.2f}"
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls_TF):.2f}"
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
               options={'maxiter': MAXITER, 'maxcor': 1000}, tol=1e-15)

bpsc = btot.Bfields[0]
bpsc.set_points(s_plot.gamma().reshape((-1, 3)))
dipole_currents = [c.current.get_value() for c in bpsc.coils]
psc_array.recompute_currents()
save_coil_sets(btot, OUT_DIR, "_optimized" + file_suffix)
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
             "B_N / B": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
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

btot.set_points(s.gamma().reshape((-1, 3)))
print('Max I = ', np.max(dipole_currents))
print('Min I = ', np.min(dipole_currents))
calculate_modB_on_major_radius(btot, s)

t2 = time.time()
print('Total time = ', t2 - t1)
from simsopt import save
save(btot.Bfields[0].coils, OUT_DIR + 'psc_coils' + file_suffix + '.json')
save(btot.Bfields[1].coils, OUT_DIR + 'TF_coils' + file_suffix + '.json')
print(OUT_DIR)
