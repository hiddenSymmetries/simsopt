#!/usr/bin/env python
r"""
This coil optimization script is similar to stage_two_optimization.py. However
in this version, the coils are constrained to be planar, by using the curve type
CurvePlanarFourier. Also the LinkingNumber objective is used to prevent coils
from becoming topologically linked with each other.

In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

The objective is given by

    J = (1/2) \int |B dot n|^2 ds
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)
        + LinkingNumber

if any of the weights are increased, or the thresholds are tightened, the coils
are more regular and better separated, but the target normal field may not be
achieved as well. This example demonstrates the adjustment of weights and
penalties via the use of the `Weight` class.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.util import calculate_on_axis_B
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier, curves_to_vtk, create_equally_spaced_planar_curves,
    create_planar_curves_between_two_toroidal_surfaces
)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions

# Number of Fourier modes describing each Cartesian component of each coil:
order = 1

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
input_name = 'input.LandremanPaul2021_QH_reactorScale_lowres'
filename = TEST_DIR / input_name
# input_name = 'wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc'
# filename = TEST_DIR / input_name

# Directory for output
OUT_DIR = "./dipole_array_optimization/"
os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################

# Initialize the boundary magnetic surface:
range_param = "half period"
nphi = 32
ntheta = 32
poff = 1.5
coff = 1.5
s = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi, ntheta=ntheta)
s_outer = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi, ntheta=ntheta)

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
def initialize_coils_QH(TEST_DIR, s):
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
    ncoils = 2
    R0 = s.get_rc(0, 0)
    R1 = s.get_rc(1, 0) * 5
    order = 1

    # qh needs to be scaled to 0.1 T on-axis magnetic field strength
    from simsopt.mhd.vmec import Vmec
    vmec_file = 'wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc'
    total_current = Vmec(TEST_DIR / vmec_file).external_current() / (2 * s.nfp) / 1.311753 / 1.999
    print('Total current = ', total_current)
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, 
                                               R0=R0, R1=R1, order=order, numquadpoints=256)
    base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils - 1)]
    # base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils)]
    # base_currents[0].fix_all()

    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

    # Initialize the coil curves and save the data to vtk
    curves = [c.curve for c in coils]
    currents = [c.current.get_value() for c in coils]
    curves_to_vtk(curves, OUT_DIR + "curves_TF_init", scalar_data=currents)
    return base_curves, curves, coils

# initialize the coils
base_curves_TF, curves_TF, coils_TF = initialize_coils_QH(TEST_DIR, s)
# currents_TF = np.array([coil.current.get_value() for coil in coils_TF])

# Set up BiotSavart fields
bs_TF = BiotSavart(coils_TF)

# Calculate average, approximate on-axis B field strength
calculate_on_axis_B(bs_TF, s)

Nx = 6
Ny = Nx
Nz = Nx
# Create the initial coils:
base_curves, all_curves = create_planar_curves_between_two_toroidal_surfaces(
    s, s_inner, s_outer, Nx, Ny, Nz, order=order
)
ncoils = len(base_curves)
print('Ncoils = ', ncoils)
for i in range(len(base_curves)):
    base_curves[i].set('x' + str(2 * order + 1), np.random.rand(1) - 0.5)
    base_curves[i].set('x' + str(2 * order + 2), np.random.rand(1) - 0.5)
    base_curves[i].set('x' + str(2 * order + 3), np.random.rand(1) - 0.5)
    base_curves[i].set('x' + str(2 * order + 4), np.random.rand(1) - 0.5)
    for j in range(2 * order + 1):
        base_curves[i].fix('x' + str(j))
    base_curves[i].fix('x' + str(2 * order + 5))
    base_curves[i].fix('x' + str(2 * order + 6))
    base_curves[i].fix('x' + str(2 * order + 7))
base_currents = [Current(1.0) * 1e6 for i in range(ncoils)]
# for i in range(ncoils):
#     base_currents[i].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
btot = bs + bs_TF
bs.set_points(s.gamma().reshape((-1, 3)))
btot.set_points(s.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
currents = [c.current.get_value() for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init", close=True, scalar_data=currents)
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init_DA", extra_data=pointData)

bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_full_init_DA", extra_data=pointData)
bs.set_points(s.gamma().reshape((-1, 3)))

# Repeat for whole B field
pointData = {"B_N": np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_full_init", extra_data=pointData)
btot.set_points(s.gamma().reshape((-1, 3)))

LENGTH_WEIGHT = Weight(0.01)
LINK_WEIGHT = 10
CC_THRESHOLD = 0.2
CC_WEIGHT = 1000
CS_THRESHOLD = 1.3
CS_WEIGHT = 1e2
CURVATURE_THRESHOLD = 1.
CURVATURE_WEIGHT = 1e-8
MSC_THRESHOLD = 1
MSC_WEIGHT = 1e-8
MAXITER = 5000

# Define the individual terms objective function:
Jf = SquaredFlux(s, btot)
# Separate length penalties on the dipole coils and the TF coils
# since they have very different sizes
Jls = [CurveLength(c) for c in base_curves]
Jls_TF = [CurveLength(c) for c in base_curves_TF]

# coil-coil and coil-plasma distances should be between all coils
Jccdist = CurveCurveDistance(curves + curves_TF, CC_THRESHOLD, num_basecurves=ncoils + len(coils_TF))
Jcsdist = CurveSurfaceDistance(curves + curves_TF, s, CS_THRESHOLD)

Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
linkNum = LinkingNumber(curves + curves_TF)

# Jccdist_TF = CurveCurveDistance(curves_TF, CC_THRESHOLD, num_basecurves=ncoils)
# Jcsdist_TF = CurveSurfaceDistance(curves_TF, s, CS_THRESHOLD)
Jcs_TF = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves_TF]
Jmscs_TF = [MeanSquaredCurvature(c) for c in base_curves_TF]

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:

# Coil shapes and center positions are fixed right now so not including this one below
# + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), 2.6*ncoils) \
JF = Jf \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs) \
    + LINK_WEIGHT * linkNum \
    + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls_TF), 2.6*5) \
    + CURVATURE_WEIGHT * sum(Jcs_TF) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs_TF) \

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    BdotN2 = 0.5 * np.mean((np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2
        )) ** 2 * np.linalg.norm(s.normal(), axis=-1))
    BdotN2_normalized = 0.5 * np.mean((np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2
        ).reshape(-1) / np.linalg.norm(btot.B(), axis=-1)) ** 2 * np.linalg.norm(s.normal(), axis=-1).reshape(-1))
    Bn_over_B = 0.5 * np.mean((np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2
        )) ** 2 * np.linalg.norm(s.normal(), axis=-1)) / np.mean(
            (np.linalg.norm(btot.B().reshape(-1, 3), axis=-1)
            ) ** 2 * np.linalg.norm(s.normal(), axis=-1).reshape(-1))
    MaxBdotN = np.max(np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    mean_AbsB = np.mean(btot.AbsB())
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    outstr += f", 0.5⟨|B·n|^2⟩={BdotN2:.1e}, 0.5⟨(|B·n|/|B|)^2⟩={BdotN2_normalized:.1e}, 0.5⟨|B·n|^2⟩/⟨|B|^2⟩={Bn_over_B:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls_TF])
    # kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    # msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    # outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", avg(L)={np.mean(np.array([J.J() for J in Jls])):.2f}"
    outstr += f", Lengths=" + cl_string
    outstr += f", avg(kappa)={np.mean(np.array([c.kappa() for c in base_curves])):.2f}"
    outstr += f", var(kappa)={np.mean(np.array([c.kappa() for c in base_curves])):.2f}"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    outstr += f", ⟨B·n⟩/|B|={BdotN/mean_AbsB:.1e}"
    outstr += f", (Max B·n)/|B|={MaxBdotN/mean_AbsB:.1e}"
    outstr += f", Link Number = {linkNum.J()}"
    print(outstr)
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
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-20)
print([c.current.get_value() for c in bs.coils], [c.current.get_value() for c in bs_TF.coils])
curves_to_vtk([c.curve for c in bs.coils], OUT_DIR + "curves_opt", scalar_data=[c.current.get_value() for c in bs.coils])
curves_to_vtk([c.curve for c in bs_TF.coils], OUT_DIR + "curves_TF_opt", 
    scalar_data=[c.current.get_value() for c in bs_TF.coils])
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt_DA", extra_data=pointData)

bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_full_opt_DA", extra_data=pointData)

# Repeat with total
pointData = {"B_N": np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData)

btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_full_opt", extra_data=pointData)

btot.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N / |B|": (np.sum(btot.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
    ) / np.linalg.norm(btot.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
s_plot.to_vtk(OUT_DIR + "surf_full_opt_normalizedBn", extra_data=pointData)