#!/usr/bin/env python

r"""
This example optimizes the NCSX coils and currents for QA on potentially multiple surfaces using the BoozerLS approach.  
For a single surface, the objective is:

    J = ( \int_S B_nonQA**2 dS )/(\int_S B_QA dS)
        + \int \int BoozerResidual(varphi, theta)^2 d\varphi d\theta
        + 0.5*(iota - iota_0)**2
        + 0.5*(major_radius - target_major_radius)**2
        + 0.5*max(\sum_{coils} CurveLength - CurveLengthTarget, 0)**2
        + other coil regularization penalties on curvature, mean squared curvature,
          coil-to-coil distance, and coil arclength.

We first load surfaces in the NCSX equilibrium, then optimize for QA on them. 
The first term the objective minimizes the deviation from quasi-axisymmetry on the loaded surfaces.
The second term minimizes the Boozer residual on the surfaces, which pushes the optimizer to heal
islands and generalized chaos.  It is typically a good idea to weight this penalty term to be ~ the
same order of magnitude of the first term in the objective.

The objective also includes penalty terms on the rotational transform, major radius, total coil length, curvature,
mean squared curvature, and coil arclength.  The rotational transform and major radius penalty ensures 
that the surfaces' rotational transform and aspect ratio do not stray too far from the value in the initial configuration.
There are also a few coil regularization penalties to prevent the coils from becoming too complex.  The BFGS 
optimizer is used, and quasisymmetry is improved substantially on the surface.  Surface solves using the BoozerLS 
approach can be costly, so this script supports distributing the solves across multiple MPI ranks.

You can change the value of the variable `nsurfaces` below to optimize for nested surfaces and QS on up to 10 surfaces.  
The BoozerSurface solves can be distributed to Nranks ranks using:
    mpirun -np Nranks ./boozerQA_ls_mpi.py
where `nsurfaces` is the number of surfaces your optimizing on.  For example, if you want one surface solve per rank,
and nsurfaces=Nranks=2, then the proper call is:
    mpirun -np 2 ./boozerQA_ls_mpi.py

More details on this work can be found at
A Giuliani et al, "Direct stellarator coil optimization for nested magnetic surfaces with precise
quasi-symmetry", Physics of Plasmas 30, 042511 (2023) doi:10.1063/5.0129716
or arxiv:2210.03248.
"""

import os
import numpy as np
from scipy.optimize import minimize
from simsopt.geo import curves_to_vtk, MajorRadius, CurveLength, CurveCurveDistance, NonQuasiSymmetricRatio, Iotas, \
    BoozerResidual, LpCurveCurvature, MeanSquaredCurvature, ArclengthVariation
from simsopt._core import load
from simsopt.objectives import MPIObjective, MPIOptimizable
from simsopt.field import BiotSavart
from simsopt.objectives import QuadraticPenalty
from simsopt.util import proc0_print, in_github_actions
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

except ImportError:
    comm = None
    size = 1
    rank = 0

# Directory for output
IN_DIR = "./inputs/input_ncsx/"
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

proc0_print("Running 2_Intermediate/boozerQA_ls_mpi.py")
proc0_print("=========================================")

base_curves, base_currents, coils, curves, surfaces, boozer_surfaces, ress = load(IN_DIR + "ncsx_init.json")

# you can optimize for QA on up to 10 surfaces, by changing nsurfaces below.
nsurfaces = 2
assert nsurfaces <= 10

surfaces = surfaces[:nsurfaces]
boozer_surfaces = boozer_surfaces[:nsurfaces]
ress = ress[:nsurfaces]
for boozer_surface, res in zip(boozer_surfaces, ress):
    boozer_surface.run_code(res['iota'], res['G'])

mpi_surfaces = MPIOptimizable(surfaces, ["x"], comm)
mpi_boozer_surfaces = MPIOptimizable(boozer_surfaces, ["res", "need_to_run_code"], comm)

mrs = [MajorRadius(boozer_surface) for boozer_surface in boozer_surfaces]
mrs_equality = [len(mrs)*QuadraticPenalty(mr, mr.J(), 'identity') if idx == len(mrs)-1 else 0*QuadraticPenalty(mr, mr.J(), 'identity') for idx, mr in enumerate(mrs)]
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
Jmajor_radius = MPIObjective(mrs_equality, comm, needs_splitting=True)

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

if comm is None or rank == 0:
    curves_to_vtk(curves, OUT_DIR + "curves_init")
for idx, surface in enumerate(mpi_surfaces):
    if comm is None or rank == 0:
        surface.to_vtk(OUT_DIR + f"surf_init_{idx}")

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

    # this check makes sure that all ranks have exactly the same dofs
    if comm is not None:
        alldofs = comm.allgather(dofs)
        assert np.all(np.all(alldofs[0]-d == 0) for d in alldofs)

    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()

    # check to make sure that all the surface solves succeeded
    success1 = np.all([boozer_surface.res['success'] for boozer_surface in mpi_boozer_surfaces])
    # check to make sure that the surfaces are not self-intersecting
    success2 = np.all([not surface.is_self_intersecting() for surface in mpi_surfaces])
    if not (success1 and success2):
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

    proc0_print(outstr)
    prevs['it'] += 1


dofs = JF.x
callback(dofs)

proc0_print("""
################################################################################
### Run the optimization #######################################################
################################################################################
""")
# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 1e3

res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER}, tol=1e-15, callback=callback)
if comm is None or rank == 0:
    curves_to_vtk(curves, OUT_DIR + "curves_opt")
for idx, surface in enumerate(mpi_surfaces):
    if comm is None or rank == 0:
        surface.to_vtk(OUT_DIR + f"surf_opt_{idx}")

proc0_print("End of 2_Intermediate/boozerQA_ls_mpi.py")
proc0_print("========================================")
