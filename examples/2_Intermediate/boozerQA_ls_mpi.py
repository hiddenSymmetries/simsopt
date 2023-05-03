#!/usr/bin/env python3
from simsopt.geo import SurfaceXYZTensorFourier, SurfaceRZFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual, \
    ToroidalFlux, Volume, MajorRadius, CurveLength, CurveCurveDistance, NonQuasiSymmetricRatio, Iotas, BoozerResidual, \
    LpCurveCurvature, MeanSquaredCurvature, ArclengthVariation
from simsopt._core import load
from simsopt.objectives import MPIObjective
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.configs import get_ncsx_data
from simsopt.objectives import QuadraticPenalty
from scipy.optimize import minimize
from qsc import Qsc as Qsc
import numpy as np
import os
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
except ImportError:
    comm = None
    rank = 0
    size = 1

"""
"""

# Directory for output
IN_DIR = "./inputs/input_ncsx/"
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

print("Running 2_Intermediate/boozerQA_ls.py")
print("================================")

base_curves, base_currents, ma = get_ncsx_data()
coils = coils_via_symmetries(base_curves, base_currents, 3, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
bs_tf = BiotSavart(coils)
current_sum = sum(abs(c.current.get_value()) for c in coils)
ncoils = len(base_curves)

## COMPUTE THE INITIAL SURFACE ON WHICH WE WANT TO OPTIMIZE FOR QA##
# Resolution details of surface on which we optimize for qa
mpol = 6
ntor = 6
stellsym = True
nfp = 3

phis = np.linspace(0, 1/nfp, 2*ntor+1+5, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1+5, endpoint=False)
s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)

boozer_surface, res = load(IN_DIR + f"surface_{rank}.json")
res = boozer_surface.run_code('ls', res['iota'], res['G'])

## SET UP THE OPTIMIZATION PROBLEM AS A SUM OF OPTIMIZABLES ##
bs_nonQS = BiotSavart(coils)
mr = MajorRadius(boozer_surface)
ls = [CurveLength(c) for c in base_curves]
iotas = Iotas(boozer_surface)
mean_iota = MPIObjective([iotas], comm)


CC_THRESHOLD = 0.15
KAPPA_THRESHOLD = 15.
MSC_THRESHOLD = 15.
IOTAS_TARGET = -0.4

RES_WEIGHT = 1e3
LENGTH_WEIGHT = 1.
MR_WEIGHT = 1.
CC_WEIGHT = 1e2
KAPPA_WEIGHT = 1.
MSC_WEIGHT = 1.
IOTAS_WEIGHT = 1.
ARCLENGTH_WEIGHT = 1e-2

Jnqs = NonQuasiSymmetricRatio(boozer_surface, bs_nonQS)
Jbr = BoozerResidual(boozer_surface, BiotSavart(coils))
J_nonQSRatio = MPIObjective([Jnqs], comm)
JBoozerResidual = MPIObjective([Jbr], comm)
Jiotas = IOTAS_WEIGHT * QuadraticPenalty(mean_iota, IOTAS_TARGET, 'identity')
if rank == 0:
    J_major_radius = QuadraticPenalty(mr, mr.J(), 'identity')  # target major radius only on innermost surface
else:
    J_major_radius = 0 * QuadraticPenalty(mr, mr.J(), 'identity')  # target major radius only on innermost surface

Jls = LENGTH_WEIGHT * QuadraticPenalty(sum(ls), float(sum(ls).J()), 'max')
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcs = KAPPA_WEIGHT * sum([LpCurveCurvature(c, 2, KAPPA_THRESHOLD) for c in base_curves])
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jmr = MR_WEIGHT * MPIObjective([J_major_radius], comm)
Jals = [ArclengthVariation(c) for c in base_curves]

JF = J_nonQSRatio + RES_WEIGHT * JBoozerResidual + Jiotas + Jmr \
    + Jls + CC_WEIGHT * Jccdist + Jcs\
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
    + ARCLENGTH_WEIGHT * sum(Jals)

boozer_surface.surface.to_vtk(OUT_DIR + f"surf_init_{rank}")
if comm is None or comm.rank == 0:
    curves_to_vtk(curves, OUT_DIR + f"curves_init")

# let's fix the coil current
base_currents[0].fix_all()

def fun(dofs):
    # save these as a backup in case the boozer surface Newton solve fails
    sdofs_prev = boozer_surface.surface.x
    iota_prev = boozer_surface.res['iota']
    G_prev = boozer_surface.res['G']

    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    
    success = boozer_surface.res['success'] if comm is not None else MPI.COMM_WORLD.allreduce(boozer_surface.res['success'], op=MPI.LAND)
    if not success:
        # failed, so reset back to previous surface and return a large value
        # of the objective.  The purpose is to trigger the line search to reduce
        # the step size.
        J = 1e3
        boozer_surface.surface.x = sdofs_prev
        boozer_surface.res['iota'] = iota_prev
        boozer_surface.res['G'] = G_prev
    
    return J, grad

def callback(x):
    J = JF.J()
    grad = JF.dJ()
    
    cl_string = ", ".join([f"{J.J():.3f}" for J in ls])
    kappa_string = ", ".join([f"{np.max(c.kappa()):.3f}" for c in base_curves])
    msc_string = ", ".join([f"{Jmsc.J():.3f}" for Jmsc in Jmscs])
    iotas_string = ", ".join([f"{val:.3f}" for val in (comm.allgather(iotas.J()) if comm is not None else iotas.J()) ])
    mr_string = ", ".join([f"{val:.3f}" for val in (comm.allgather(boozer_surface.surface.major_radius()) if comm is not None else boozer_surface.surface.major_radius() ) ])
    ar_string = ", ".join([f"{val:.3f}" for val in (comm.allgather(boozer_surface.surface.aspect_ratio()) if comm is not None else boozer_surface.surface.aspect_ratio() ) ])
    br_string = ", ".join([f"{val:.3e}" for val in (comm.allgather(Jbr.J()) if comm is not None else Jbr.J() ) ])
    nqs_ratio_string = ", ".join([f"{val:.3e}" for val in (comm.allgather(Jnqs.J()) if comm is not None else Jnqs.J() ) ])
    outstr = f"\nJ={J:.3e}\n"
    outstr += f"nqas_ratio_string=[{nqs_ratio_string}], \n"
    outstr += f"br=[{br_string}], \n"
    outstr += f"mean_iota={mean_iota.J():.3f}, \n"
    outstr += f"iotas=[{iotas_string}], \n"
    outstr += f"mr=[{mr_string}], \n"
    outstr += f"AR=[{ar_string}], \n"
    outstr += f"Len=sum([{cl_string}])={sum(J.J() for J in ls):.3f},\n"
    outstr += f"kappa=[{kappa_string}], \n"
    outstr += f"msc=[{msc_string}], \n"
    outstr += f"cc_dist={Jccdist.shortest_distance():.3f}, \n"
    outstr += f"║∇J║={np.linalg.norm(grad):.3e}\n\n"
    if comm is None or comm.rank == 0:
        print(outstr, flush=True)


#print("""
#################################################################################
#### Perform a Taylor test ######################################################
#################################################################################
#""")
#f = fun
#dofs = JF.x
#np.random.seed(1)
#h = np.random.uniform(size=dofs.shape)
#J0, dJ0 = f(dofs)
#dJh = sum(dJ0 * h)
#for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
#    J1, _ = f(dofs + 2*eps*h)
#    J2, _ = f(dofs + eps*h)
#    J3, _ = f(dofs - eps*h)
#    J4, _ = f(dofs - 2*eps*h)
#    print("err", ((J1*(-1/12) + J2*(8/12) + J3*(-8/12) + J4*(1/12))/eps - dJh)/np.linalg.norm(dJh))

dofs = JF.x
callback(dofs)
#quit()

print("""
################################################################################
### Run the optimization #######################################################
################################################################################
""")
# Number of iterations to perform:
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
MAXITER = 50 if ci else 1e3

res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER}, tol=1e-15, callback=callback)
curves_to_vtk(curves, OUT_DIR + f"curves_opt")
boozer_surface.surface.to_vtk(OUT_DIR + "surf_opt")

print("End of 2_Intermediate/boozerQA_ls.py")
print("================================")
