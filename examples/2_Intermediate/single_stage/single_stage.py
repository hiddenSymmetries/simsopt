#!/usr/bin/env python

import os
import numpy as np
from simsopt.objectives import LeastSquaresProblem
from simsopt.util import MpiPartition
from simsopt.mhd import Vmec
from simsopt.mhd import QuasisymmetryRatioResidual
from scipy.optimize import minimize, fmin
from simsopt.objectives import Weight
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import curves_to_vtk, create_equally_spaced_curves
from simsopt.field import BiotSavart
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import CurveLength, CurveCurveDistance, \
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.util.mpi import log
from simsopt._core.derivative import derivative_dec, Derivative
"""
Optimize a VMEC equilibrium for quasisymmetry and coils
"""

log()

print("Running single_stage.py")
print("=============================================")

mpi = MpiPartition()

# For forming filenames for vmec, pathlib sometimes does not work, so use os.path.join instead.
# filename = os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp4_QH_warm_start')
filename = os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp2_QA')
vmec = Vmec(filename, mpi=mpi, verbose=True)

# Define parameter space:
surf = vmec.boundary
surf.fix_all()
max_mode = 1
surf.fixed_range(mmin=0, mmax=max_mode,
                 nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")  # Major radius

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(vmec,
                                np.arange(0, 1.01, 0.1),  # np.arange(0, 1.01, 0.1),  # Radii to target
                                # helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|
                                helicity_m=1, helicity_n=0)  # (M, N) you want in |B|
# aspect_target = 7
aspect_target = 6

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 4

# Major radius for the initial circular coils:
R0 = 1.0

# Minor radius for the initial circular coils:
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = 4

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
MAXITER = 50

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, vmec.indata.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1e5) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, vmec.indata.nfp, True)
bs = BiotSavart(coils)
bs.set_points(surf.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
nphi = 50
ntheta = 50
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
surf.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the individual terms objective function:
Jf = SquaredFlux(surf, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, surf, CS_THRESHOLD)
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
flux_weight = 1
inner_coil_iterations = 30

## First stage objective function
prob = LeastSquaresProblem.from_tuples([(vmec.aspect, aspect_target, 1),
                                        (qs.residuals, 0, 1),
                                        (vmec.mean_iota, 0.42, 1)])

def jac_fun(dofs, prob_jacobian=None):
    ## Order of dofs: (coils dofs, surface dofs)
    JF.x = dofs[:-number_vmec_dofs]
    prob.x = dofs[-number_vmec_dofs:]
    bs.set_points(surf.gamma().reshape((-1, 3)))

    ## Finite differences for the first-stage objective function
    prob_dJ = prob_jacobian.jac()

    ## Finite differences for the second-stage objective function
    coils_dJ = JF.dJ()

    ## Mixed term - derivative of squared flux with respect to the surface shape
    n = surf.normal()
    absn = np.linalg.norm(n, axis=2)
    B = bs.B().reshape((nphi, ntheta, 3))
    dB_by_dX = bs.dB_by_dX().reshape((nphi, ntheta, 3, 3))
    Bcoil = bs.B().reshape(n.shape)
    B_N = np.sum(Bcoil * n, axis=2)
    dJdx = (B_N/absn)[:, :, None] * (np.sum(dB_by_dX*n[:, :, None, :], axis=3))
    dJdN = (B_N/absn)[:, :, None] * B - 0.5 * (B_N**2/absn**3)[:, :, None] * n

    deriv = surf.dnormal_by_dcoeff_vjp(dJdN/(nphi*ntheta)) + surf.dgamma_by_dcoeff_vjp(dJdx/(nphi*ntheta))
    mixed_dJ = Derivative({surf: deriv})(surf)

    ## Put both gradients together
    grad_with_respect_to_coils = flux_weight * coils_dJ
    grad_with_respect_to_surface = np.sum(prob_dJ, axis=0) + flux_weight * mixed_dJ
    grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))

    return grad

def fun_coils(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)))
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad

def fun(dofs, prob_jacobian=None):
    ## Order of dofs: (coils dofs, surface dofs)
    JF.x = dofs[:-number_vmec_dofs]
    prob.x = dofs[-number_vmec_dofs:]
    bs.set_points(surf.gamma().reshape((-1, 3)))

    # Do two coil optimization loops, the latter with slightly longer coils
    curves_to_vtk(curves, OUT_DIR + f"curves_before_inner_loop")
    res = minimize(fun_coils, dofs[:-number_vmec_dofs], jac=True, method='L-BFGS-B', options={'maxiter': inner_coil_iterations, 'maxcor': 300}, tol=1e-15)
    dofs[:-number_vmec_dofs] = res.x
    JF.x = dofs[:-number_vmec_dofs]
    curves_to_vtk(curves, OUT_DIR + f"curves_after_inner_loop")

    ## Objective function
    try:
        J = np.sum(prob.objective()) + flux_weight * JF.J()
    except:
        print("Exception caught during function evaluation. Returing J=1e12")
        J = 1e12

    # Print some results
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)))
    outstr = f"Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    try:
        outstr += f", Quasisymmetry objective={qs.total()}"
        outstr += f", aspect={vmec.aspect()}"
        outstr += f", mean iota={vmec.mean_iota()}"
    except Exception as e:
        print(e)
    print(outstr)
    return J


print("Quasisymmetry objective before optimization:", qs.total())

x0 = np.copy(np.concatenate((JF.x, vmec.x)))
## Optimize using finite differences
with MPIFiniteDifference(prob.objective, mpi, abs_step=1e-5) as prob_jacobian:
    if mpi.proc0_world:
        res = minimize(fun, np.concatenate((JF.x, vmec.x)), args=(prob_jacobian,), jac=jac_fun, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300, 'iprint': 101}, tol=1e-15)
## Optimize without using finite differences
# res = minimize(fun, x0)

print("Final aspect ratio:", vmec.aspect())
print("Quasisymmetry objective after optimization:", qs.total())

# Output the result
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
surf.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData)
curves_to_vtk(curves, OUT_DIR + f"curves_opt")

# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
bs.save(OUT_DIR + "biot_savart_opt.json")

print("End of single_stage.py")
print("============================================")
