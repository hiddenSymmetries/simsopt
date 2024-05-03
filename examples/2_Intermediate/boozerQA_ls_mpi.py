#!/usr/bin/env python3
from simsopt.geo import SurfaceXYZTensorFourier, SurfaceRZFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual, \
    ToroidalFlux, Volume, MajorRadius, CurveLength, CurveCurveDistance, NonQuasiSymmetricRatio, Iotas, BoozerResidual, \
    LpCurveCurvature, MeanSquaredCurvature, ArclengthVariation
from simsopt._core import load
from simsopt.objectives import MPIObjective, MPIOptimizable
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.configs import get_ncsx_data
from simsopt.objectives import QuadraticPenalty
from scipy.optimize import minimize
import numpy as np
import os
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    def pprint(*args, **kwargs):
        if comm.rank == 0:  # only print on rank 0
            print(*args, **kwargs)

except ImportError:
    comm = None
    size = 1
    pprint = print
    rank = 0

"""
This example optimizes the NCSX coils and currents for QA on potentially multiple surfaces using the BoozerLS approach.  
For a single surface, the objective is:

    J = ( \int_S B_nonQA**2 dS )/(\int_S B_QA dS)
        + 0.5*(iota - iota_0)**2
        + 0.5*(major_radius - target_major_radius)**2
        + 0.5*max(\sum_{coils} CurveLength - CurveLengthTarget, 0)**2

We first load a surface close to the magnetic axis, then optimize for QA on that surface.  
The objective also includes penalty terms on the rotational transform, major radius,
and total coil length.  The rotational transform and major radius penalty ensures that the surface's
rotational transform and aspect ratio do not stray too far from the value in the initial configuration.
There is also a penalty on the total coil length as a regularizer to prevent the coils from becoming
too complex.  The BFGS optimizer is used, and quasisymmetry is improved substantially on the surface.
Surface solves using the BoozerLS approach can be costly, so this script supports distributing the solves 
across multiple MPI ranks.

More details on this work can be found at or doi:10.1063/5.0129716 arxiv:2210.03248.
"""

# Directory for output
IN_DIR = "./inputs/input_ncsx/"
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

pprint("Running 2_Intermediate/boozerQA_ls_mpi.py")
pprint("================================")

base_curves, base_currents, coils, curves, surfaces, boozer_surfaces, ress = load(IN_DIR + f"ncsx_init.json")
nsurfaces = 2
surfaces = surfaces[:nsurfaces]
boozer_surfaces = boozer_surfaces[:nsurfaces]
ress = ress[:nsurfaces]
for boozer_surface, res in zip(boozer_surfaces, ress):
    boozer_surface.run_code('ls', res['iota'], res['G'], verbose=False)

mpi_surfaces = MPIOptimizable(surfaces, ["x"], comm)
mpi_boozer_surfaces = MPIOptimizable(boozer_surfaces, ["res", "need_to_run_code"], comm)

mrs = [MajorRadius(boozer_surface) for boozer_surface in boozer_surfaces]
iotas = [Iotas(boozer_surface) for boozer_surface in boozer_surfaces]
nonQSs = [NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils)) for boozer_surface in boozer_surfaces]
brs = [BoozerResidual(boozer_surface, BiotSavart(coils)) for boozer_surface in boozer_surfaces]

MIN_DIST_THRESHOLD = 0.15
KAPPA_THRESHOLD = 15.
MSC_THRESHOLD = 15.
IOTAS_TARGET = -0.4

RES_WEIGHT = 1e4
LENGTH_WEIGHT = 1.
MR_WEIGHT = 1.
MIN_DIST_WEIGHT = 1e2
KAPPA_WEIGHT = 1.
MSC_WEIGHT = 1.
IOTAS_WEIGHT = 1.
ARCLENGTH_WEIGHT = 1e-2

mean_iota = MPIObjective(iotas, comm, needs_splitting=True)
Jiotas = QuadraticPenalty(mean_iota, IOTAS_TARGET, 'identity')
JnonQSRatio = MPIObjective(nonQSs, comm, needs_splitting=True)
JBoozerResidual = MPIObjective(brs, comm, needs_splitting=True)
Jmajor_radius = MPIObjective([len(mrs)*QuadraticPenalty(mr, mr.J(), 'identity') if idx == 0 else 0*QuadraticPenalty(mr, mr.J(), 'identity') for idx, mr in enumerate(mrs)], comm, needs_splitting=True)

ls = [CurveLength(c) for c in base_curves]
Jls = QuadraticPenalty(sum(ls), float(sum(ls).J()), 'max')
Jccdist = CurveCurveDistance(curves, MIN_DIST_THRESHOLD, num_basecurves=len(base_curves))
Jcs = sum([LpCurveCurvature(c, 2, KAPPA_THRESHOLD) for c in base_curves])
msc_list = [MeanSquaredCurvature(c) for c in base_curves]
Jmsc = sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in msc_list)
Jals = sum([ArclengthVariation(c) for c in base_curves])

JF = JnonQSRatio + RES_WEIGHT * JBoozerResidual + IOTAS_WEIGHT * Jiotas + MR_WEIGHT * Jmajor_radius \
    + LENGTH_WEIGHT * Jls + MIN_DIST_WEIGHT * Jccdist + KAPPA_WEIGHT * Jcs\
    + MSC_WEIGHT * Jmsc \
    + ARCLENGTH_WEIGHT * Jals

# let's fix the coil current
base_currents[0].fix_all()

boozer_surface.surface.to_vtk(OUT_DIR + f"surf_init_{rank}")
if comm is None or comm.rank == 0:
    curves_to_vtk(curves, OUT_DIR + f"curves_init")

# dictionary used to save the last accepted surface dofs in the line search, in case Newton's method fails
prevs = {'sdofs': [surface.x.copy() for surface in mpi_surfaces], 'iota': [boozer_surface.res['iota'] for boozer_surface in mpi_boozer_surfaces],
         'G': [boozer_surface.res['G'] for boozer_surface in mpi_boozer_surfaces], 'J': JF.J(), 'dJ': JF.dJ().copy(), 'it': 0}

def fun(dofs):
    # initialize to last accepted surface values
    for idx, surface in enumerate(mpi_surfaces):
        surface.x = prevs['sdofs'][idx]
    for idx, boozer_surface in enumerate(mpi_boozer_surfaces):
        boozer_surface.res['iota'] = prevs['iota'][idx]
        boozer_surface.res['G'] = prevs['G'][idx]
    
    #alldofs = MPI.COMM_WORLD.allgather(dofs)
    #assert np.all(np.norm(alldofs[0]-d) == 0 for d in alldofs)
 
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    
    success = np.all([boozer_surface.res['success'] for boozer_surface in mpi_boozer_surfaces])
    if not success:
        J = prevs['J']
        grad = -prevs['dJ']
        for idx, boozer_surface in enumerate(mpi_boozer_surfaces):
            boozer_surface.surface.x = prevs['sdofs'][idx]
            boozer_surface.res['iota'] = prevs['iota'][idx]
            boozer_surface.res['G'] = prevs['G'][idx]
    return J, grad


def callback(x):
    # since this set of coil dofs was accepted, set the backup surface dofs
    # to accepted ones in case future Newton solves fail.
    for idx, surface in enumerate(mpi_surfaces):
        prevs['sdofs'][idx] = surface.x.copy()
    for idx, boozer_surface in enumerate(mpi_boozer_surfaces):
        prevs['iota'][idx] = boozer_surface.res['iota']
        prevs['G'][idx] = boozer_surface.res['G']
    prevs['J'] = JF.J()
    prevs['dJ'] = JF.dJ().copy()
    
    width = 35
    outstr = f"\nIteration {prevs['it']}\n"
    outstr += f"{'J':{width}} {JF.J():.6e} \n"
    outstr += f"{'║∇J║':{width}} {np.linalg.norm(JF.dJ()):.6e} \n\n"
    outstr += f"{'nonQS ratio':{width}}" + ", ".join([f'{np.sqrt(nonqs.J()):.6e}' for nonqs in nonQSs]) + "\n"
    outstr += f"{'Boozer Residual':{width}}" + ", ".join([f'{br.J():.6e}' for br in brs]) + "\n"
    outstr += f"{'<ι>':{width}} {mean_iota.J():.6f} \n"
    outstr += f"{'ι on surfaces':{width}}" + ", ".join([f"{boozer_surface.res['iota']:.6f}" for boozer_surface in mpi_boozer_surfaces]) + "\n"
    outstr += f"{'major radius on surfaces':{width}}" + ', '.join([f'{surface.major_radius():.6f}' for surface in mpi_surfaces]) + "\n"
    outstr += f"{'minor radius on surfaces':{width}}" + ', '.join([f'{surface.minor_radius():.6f}' for surface in mpi_surfaces]) + "\n"
    outstr += f"{'aspect ratio radius on surfaces':{width}}" + ', '.join([f'{surface.aspect_ratio():.6f}' for surface in mpi_surfaces]) + "\n"
    outstr += f"{'volume':{width}}" + ', '.join([f'{surface.volume():.6f}' for surface in mpi_surfaces]) + "\n"
    outstr += f"{'surfaces are self-intersecting':{width}}" + ', '.join([f'{surface.is_self_intersecting()}' for surface in mpi_surfaces]) + "\n"
    outstr += f"{'shortest coil to coil distance':{width}} {Jccdist.shortest_distance():.3f} \n"
    outstr += f"{'coil lengths':{width}}" + ', '.join([f'{J.J():5.6f}' for J in ls]) + "\n"
    outstr += f"{'coil length sum':{width}} {sum(J.J() for J in ls):.3f} \n"
    outstr += f"{'max κ':{width}}" + ', '.join([f'{np.max(c.kappa()):.6f}' for c in base_curves]) + "\n"
    outstr += f"{'∫ κ^2 dl / ∫ dl':{width}}" + ', '.join([f'{Jmsc.J():.6f}' for Jmsc in msc_list]) + "\n"
    outstr += "\n\n"

    pprint(outstr)
    prevs['it'] += 1


dofs = JF.x
callback(dofs)

pprint("""
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

pprint("End of 2_Intermediate/boozerQA_ls.py")
pprint("================================")
