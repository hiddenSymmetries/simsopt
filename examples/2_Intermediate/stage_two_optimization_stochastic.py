#!/usr/bin/env python

r"""
In this example we solve a stochastic version of the FOCUS like Stage II coil
optimisation problem: the goal is to find coils that generate a specific target
normal field on a given surface.  In this particular case we consider a vacuum
field, so the target is just zero.

The objective is given by

    J = (1/2) Mean(\int |B dot n|^2 ds)
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)
        + ARCLENGTH_WEIGHT * ArclengthVariation

where the Mean is approximated by a sample average over perturbed coils.

The target equilibrium is the QA configuration of arXiv:2108.03711.

The coil perturbations for each coil are the sum of a 'systematic error' and a
'statistical error'.  The former satisfies rotational and stellarator symmetry,
the latter is independent for each coil.
"""

import os
from pathlib import Path
import numpy as np
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.geo import curves_to_vtk, create_equally_spaced_curves
from simsopt.field import BiotSavart
from simsopt.field import Current, Coil, coils_via_symmetries
from simsopt.geo import CurveLength, CurveCurveDistance, \
    MeanSquaredCurvature, LpCurveCurvature, ArclengthVariation
from simsopt.geo.curveperturbed import GaussianSampler, CurvePerturbed, PerturbationSample
from simsopt.objectives import QuadraticPenalty, MPIObjective
from randomgen import PCG64

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    def pprint(*args, **kwargs):
        if comm.rank == 0:  # only print on rank 0
            print(*args, **kwargs)
except ImportError:
    comm = None
    pprint = print

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 4

# Major radius for the initial circular coils:
R0 = 1.0

# Minor radius for the initial circular coils:
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = 24

# Weight on the curve lengths in the objective function:
LENGTH_WEIGHT = 1e-6

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
DISTANCE_THRESHOLD = 0.1
DISTANCE_WEIGHT = 10

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 5.
CURVATURE_WEIGHT = 1e-6

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 5
MSC_WEIGHT = 1e-6

# Weight for the arclength variation penalty in the objective function:
ARCLENGTH_WEIGHT = 1e-2

# Standard deviation for the coil errors
SIGMA = 1e-3

# Length scale for the coil errors
L = 0.5

# Number of samples to approximate the mean
N_SAMPLES = 16

# Number of samples for out-of-sample evaluation
N_OOS = 256

# Number of iterations to perform:
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
MAXITER = 50 if ci else 400

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################

# Initialize the boundary magnetic surface; errors break symmetries, so consider the full torus
nphi = 64
ntheta = 16
s = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi, ntheta=ntheta)

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1e5) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)


Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jdist = CurveCurveDistance(curves, DISTANCE_THRESHOLD, num_basecurves=ncoils)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]

seed = 0
rg = np.random.Generator(PCG64(seed, inc=0))

sampler = GaussianSampler(curves[0].quadpoints, SIGMA, L, n_derivs=1)
Jfs = []
curves_pert = []
for i in range(N_SAMPLES):
    # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
    base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
    coils = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
    # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
    coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
    curves_pert.append([c.curve for c in coils_pert])
    bs_pert = BiotSavart(coils_pert)
    Jfs.append(SquaredFlux(s, bs_pert))
Jmpi = MPIObjective(Jfs, comm, needs_splitting=True)

for i in range(len(curves_pert)):
    curves_to_vtk(curves_pert[i], OUT_DIR + f"curves_init_{i}")

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = Jmpi \
    + LENGTH_WEIGHT * sum(Jls) \
    + DISTANCE_WEIGHT * Jdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs) \
    + ARCLENGTH_WEIGHT * sum(Jals)

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    cl_string = ", ".join([f"{J.J():.3f}" for J in Jls])
    mean_AbsB = np.mean(bs.AbsB())
    jf = Jmpi.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J:.1e}, ⟨Jf⟩={jf:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L>=[{msc_string}], C-C-Sep={Jdist.shortest_distance():.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    pprint(outstr, flush=True)
    return J, grad


pprint("""
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
    pprint("err", (J1-J2)/(2*eps) - dJh)

pprint("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
from scipy.optimize import minimize
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 400}, tol=1e-15)
alen_string = ", ".join([f"{np.max(c.incremental_arclength())/np.min(c.incremental_arclength())-1:.2e}" for c in base_curves])
pprint(f"Final arclength variation max(|ℓ|)/min(|ℓ|) - 1=[{alen_string}]")

pprint("""
################################################################################
### Evaluate the obtained coils ################################################
################################################################################
""")
curves_to_vtk(curves, OUT_DIR + "curves_opt")
for i in range(len(curves_pert)):
    curves_to_vtk(curves_pert[i], OUT_DIR + f"curves_opt_{i}")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData)
Jf.x = res.x
pprint(f"Mean Flux Objective across perturbed coils: {Jmpi.J():.3e}")
pprint(f"Flux Objective for exact coils coils      : {Jf.J():.3e}")

# now draw some fresh samples to evaluate the out-of-sample error
rg = np.random.Generator(PCG64(seed+1, inc=0))
val = 0
for i in range(N_OOS):
    # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
    base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
    coils = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
    # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
    coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
    curves_pert.append([c.curve for c in coils_pert])
    bs_pert = BiotSavart(coils_pert)
    val += SquaredFlux(s, bs_pert).J()

val *= 1./N_OOS
pprint(f"Out-of-sample flux value                  : {val:.3e}")
