#!/usr/bin/env python

import os
import numpy as np
from simsopt.util import MpiPartition
from simsopt.mhd import Vmec
from simsopt.mhd import QuasisymmetryRatioResidual
from scipy.optimize import minimize
from simsopt.objectives import Weight
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import curves_to_vtk, create_equally_spaced_curves
from simsopt.field import BiotSavart
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import CurveLength, CurveCurveDistance, \
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance
from simsopt._core.finite_difference import MPIFiniteDifference
"""
Optimize a VMEC equilibrium for quasisymmetry good coils
"""

print("Running 2_Intermediate/single_stage.py")
print("=============================================")

mpi = MpiPartition()

# For forming filenames for vmec, pathlib sometimes does not work, so use os.path.join instead.
filename = os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp4_QH_warm_start')
vmec = Vmec(filename, mpi=mpi, verbose=True)

# Define parameter space:
surf = vmec.boundary
surf.fix_all()
max_mode = 4
surf.fixed_range(mmin=0, mmax=max_mode,
                 nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")  # Major radius

# print('Parameter space:', surf.dof_names)

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(vmec,
                                [0.1, 0.5], #np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 4

# Major radius for the initial circular coils:
R0 = 1.0

# Minor radius for the initial circular coils:
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5


# Weight on the curve lengths in the objective function. We use the `Weight`
# class here to later easily adjust the scalar value and rerun the optimization
# without having to rebuild the objective.
LENGTH_WEIGHT = Weight(1e-6)

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.1
CC_WEIGHT = 1000

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.3
CS_WEIGHT = 10

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 5.
CURVATURE_WEIGHT = 1e-6

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 5
MSC_WEIGHT = 1e-6

# Number of iterations to perform:
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
MAXITER = 50 if ci else 400

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

# Initialize the boundary magnetic surface:
nphi = 50
ntheta = 50
s = surf

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, vmec.indata.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1e5) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, vmec.indata.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = Jf \
    + LENGTH_WEIGHT * sum(Jls) \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize

number_vmec_dofs = int(len(surf.x))
def fun(dofs):
    ## Order of dofs: (coils dofs, surface dofs)
    JF.x = dofs[:-number_vmec_dofs]
    vmec.x = dofs[-number_vmec_dofs:]
    # surf.x = dofs[-number_vmec_dofs:] # This one should be changed automatically

    # J = np.concatenate(([vmec.aspect()-7],qs.residuals(),[JF.J()]))
    J = vmec.aspect()-7+np.sum(qs.residuals())+JF.J()

    ## Finite differences for the aspect ratio
    aspect_ratio_jacobian = MPIFiniteDifference(vmec.aspect, mpi)
    aspect_ratio_jacobian.mpi_apart()
    aspect_ratio_jacobian.init_log()
    if mpi.proc0_world:
        aspect_ratio_dJ = aspect_ratio_jacobian.jac()
    mpi.together()
    if mpi.proc0_world:
        aspect_ratio_jacobian.log_file.close()

    ## Finite differences for the quasisymmetry residuals
    qs_residuals_jacobian = MPIFiniteDifference(qs.residuals, mpi)
    qs_residuals_jacobian.mpi_apart()
    qs_residuals_jacobian.init_log()
    if mpi.proc0_world:
        qs_residuals_dJ = qs_residuals_jacobian.jac()
    mpi.together()
    if mpi.proc0_world:
        qs_residuals_jacobian.log_file.close()

    ## Finite differences for the coils objective function
    coils_dJ = JF.dJ()

    ## Derivative matrix: columns = (coils dofs, surface dofs), rows = (aspect_ratio, qs_residuals, coils.J)
    # grad_with_respect_to_surface = np.vstack((aspect_ratio_dJ, qs_residuals_dJ, np.zeros((1,aspect_ratio_dJ.shape[1]))))
    # grad_with_respect_to_coils = np.vstack((np.zeros((qs_residuals_dJ.shape[0]+1,coils_dJ.shape[0])), [coils_dJ]))
    grad_with_respect_to_coils = coils_dJ
    grad_with_respect_to_surface = np.sum(qs_residuals_dJ,axis=0) + aspect_ratio_dJ[0]
    grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))

    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    # outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    outstr = f"J={np.sum(J):.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    outstr += f", Quasisymmetry objective={qs.total()}"
    outstr += f", aspect={vmec.aspect()}"
    print(outstr)
    return J, grad


print("Quasisymmetry objective before optimization:", qs.total())
# print("Total objective before optimization:", prob.objective())

# Define objective function
initial_dofs = np.concatenate((JF.x, surf.x))
# res = minimize(fun, initial_dofs)
res = minimize(fun, initial_dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
# prob = LeastSquaresProblem.from_tuples([(vmec.aspect, 7, 1),
#                                         (qs.residuals, 0, 1),
#                                         (Jf.J, 0, 1)])
# least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-8, max_nfev=2)

print("Final aspect ratio:", vmec.aspect())
print("Quasisymmetry objective after optimization:", qs.total())
# print("Total objective after optimization:", prob.objective())

print("End of 2_Intermediate/QH_fixed_resolution.py")
print("============================================")
