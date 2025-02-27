#!/usr/bin/env python3
from randomgen import PCG64
import os
import numpy as np
from scipy.optimize import minimize
import sys

from simsopt._core.derivative import derivative_dec
from simsopt._core import save, load, Optimizable
from simsopt.objectives import MPIObjective, MPIOptimizable
from simsopt.configs import get_ncsx_data
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual, \
    Volume, MajorRadius, CurveLength, NonQuasiSymmetricRatio, Iotas
from simsopt.objectives import QuadraticPenalty
from simsopt.util import in_github_actions
from simsopt.geo import (CurveLength, CurveCurveDistance, curves_to_vtk, create_equally_spaced_curves, SurfaceRZFourier,
                         MeanSquaredCurvature, LpCurveCurvature, ArclengthVariation, GaussianSampler, CurvePerturbed, PerturbationSample)
from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
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




"""
This example optimizes the NCSX coils and currents for QA on a single surface.  The objective is

    J = ( \int_S B_nonQA**2 dS )/(\int_S B_QA dS)
        + 0.5*(iota - iota_0)**2
        + 0.5*(major_radius - target_major_radius)**2
        + 0.5*max(\sum_{coils} CurveLength - CurveLengthTarget, 0)**2

We first compute a surface close to the magnetic axis, then optimize for QA on that surface.  
The objective also includes penalty terms on the rotational transform, major radius,
and total coil length.  The rotational transform and major radius penalty ensures that the surface's
rotational transform and aspect ratio do not stray too far from the value in the initial configuration.
There is also a penalty on the total coil length as a regularizer to prevent the coils from becoming
too complex.  The BFGS optimizer is used, and quasisymmetry is improved substantially on the surface.

More details on this work can be found at doi:10.1017/S0022377822000563 or arxiv:2203.03753.
"""

# Directory for output
OUT_DIR      = f"./Nsamples={comm.size}/"
os.makedirs(OUT_DIR, exist_ok=True)

proc0_print("Running 2_Intermediate/boozerQA.py", flush=True)
proc0_print("================================", flush=True)

perf = False
seed = rank
base_curves, base_currents, ma = get_ncsx_data()
coils = coils_via_symmetries(base_curves, base_currents, 3, True)
curves = [c.curve for c in coils]
bs_orig = BiotSavart(coils)
current_sum = sum(abs(c.current.get_value()) for c in coils)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
coils_orig = coils
# let's fix the coil current
base_currents[0].fix_all()

#bs_orig.fix_all()
#base_currents[0].unfix_all()

## COMPUTE THE INITIAL SURFACE ON WHICH WE WANT TO OPTIMIZE FOR QA##
# Resolution details of surface on which we optimize for qa

N=6
nfp_eff = 3
mpol = N
ntor = nfp_eff*N
stellsym = False
nfp = 1

phis = np.linspace(0, 1, 2*ntor+1, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
s_orig = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)

# To generate an initial guess for the surface computation, start with the magnetic axis and extrude outward
s_orig.fit_to_curve(ma, 0.1, flip_theta=True)

# Use a volume surface label
vol = Volume(s_orig)
vol_target = vol.J()

iota = -0.406
G = G0

## compute the surface
boozer_surface = BoozerSurface(bs_orig, s_orig, vol, vol_target, options={'newton_tol':1e-12, 'newton_maxiter':10, 'verbose':rank==0})

# compute surface first using LBFGS, this will just be a rough initial guess
res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-10, maxiter=300, constraint_weight=100., iota=iota, G=G)
proc0_print(f"After LBFGS:   iota={res['iota']:.3f}, area={s_orig.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s_orig, res['iota'], res['G'], bs_orig, derivatives=0)):.3e}", flush=True)


boozer_surface.need_to_run_code = True
# now drive the residual down using a specialised least squares algorithm
res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=100, constraint_weight=100., iota=res['iota'], G=res['G'], method='manual')
proc0_print(f"After Lev-Mar: iota={res['iota']:.3f}, area={s_orig.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s_orig, res['iota'], res['G'], bs_orig, derivatives=0)):.3e}", flush=True)
boozer_surface.need_to_run_code = True

res_orig = boozer_surface.solve_residual_equation_exactly_newton2(tol=1e-12, maxiter=20, iota=res['iota'], G=G0, verbose=rank==0)
xorig = s_orig.x.copy()
boozer_surface_orig = boozer_surface

SIGMA = 1.
rg = np.random.Generator(PCG64(seed, inc=0))
L = 0.4*np.pi
sqrtC = GaussianSampler(base_curves[0].quadpoints, SIGMA, L, n_derivs=1).L

surfaces = [s_orig]
boozer_surfaces = [boozer_surface_orig]
J_nonQSRatio = NonQuasiSymmetricRatio(boozer_surface_orig, BiotSavart(coils), sqrtC=sqrtC)

#x = J_nonQSRatio.x.copy()
#x[10] += 1e-200*1j
#J_nonQSRatio.x = x
#print(J_nonQSRatio.J())
#tr = J_nonQSRatio.GN_trace()
#print(tr.real, tr.imag/1e-200)

## SET UP THE OPTIMIZATION PROBLEM AS A SUM OF OPTIMIZABLES ##
MIN_DIST_THRESHOLD = 0.15
KAPPA_THRESHOLD = 15.
MSC_THRESHOLD = 15.
IOTAS_TARGET = -0.415
MR_TARGET = 1.467434

LENGTH_WEIGHT = 1.
MR_WEIGHT = 1.
MIN_DIST_WEIGHT = 1e2
KAPPA_WEIGHT = 1.
MSC_WEIGHT = 1.
IOTAS_WEIGHT = 1e2
ARCLENGTH_WEIGHT = 1e-2

ls = [CurveLength(c) for c in base_curves]
mrs = [MajorRadius(boozer_surface) for boozer_surface in boozer_surfaces]
iotas = [Iotas(boozer_surface) for boozer_surface in boozer_surfaces]

J_major_radius = QuadraticPenalty(mrs[0], MR_TARGET, 'identity')
J_iotas = QuadraticPenalty(iotas[0], IOTAS_TARGET, 'identity')
Jls = QuadraticPenalty(sum(ls), sum(ls).J(), 'max')
Jccdist = CurveCurveDistance(curves, MIN_DIST_THRESHOLD, num_basecurves=len(base_curves))
Jcs = sum([LpCurveCurvature(c, 2, KAPPA_THRESHOLD) for c in base_curves])
msc_list = [MeanSquaredCurvature(c) for c in base_curves]
Jmsc = sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in msc_list)
Jals = sum([ArclengthVariation(c) for c in base_curves])

# sum the objectives together
#JF = J_nonQSRatio + IOTAS_WEIGHT*J_iotas + MR_WEIGHT*J_major_radius + LENGTH_WEIGHT*Jls\
#        + MIN_DIST_WEIGHT * Jccdist + KAPPA_WEIGHT*Jcs + MSC_WEIGHT*Jmsc + ARCLENGTH_WEIGHT*Jals
JF = IOTAS_WEIGHT*J_iotas + MR_WEIGHT*J_major_radius + LENGTH_WEIGHT*Jls + ARCLENGTH_WEIGHT*Jals


#curves_to_vtk(curves, OUT_DIR + "curves_init")
#boozer_surface.surface.to_vtk(OUT_DIR + "surf_init")


eps = 1e-200
# dictionary used to save the last accepted surface dofs in the line search, in case Newton's method fails
prevs = {'sdofs': [surface.x.copy() for surface in surfaces], 'iota': [boozer_surface.res['iota'] for boozer_surface in boozer_surfaces],
         'G': [boozer_surface.res['G'] for boozer_surface in boozer_surfaces], 'J': JF.J(), 'it': 0}
def fun(dofs):
    # initialize to last accepted surface values
    for idx, surface in enumerate(surfaces):
        surface.x = prevs['sdofs'][idx]
    for idx, boozer_surface in enumerate(boozer_surfaces):
        boozer_surface.res['iota'] = prevs['iota'][idx]
        boozer_surface.res['G'] = prevs['G'][idx]
    
    # this check makes sure that all ranks have exactly the same dofs
    if comm is not None:
        alldofs = comm.allgather(dofs)
        assert np.all(np.all(alldofs[0]-d == 0) for d in alldofs)
    
    dofs_complex = dofs.copy().astype(complex)
    dofs_complex[rank] = dofs[rank] + eps*1j
    JF.x = dofs_complex
    J_rank = JF.J() + J_nonQSRatio.J()
    # check to make sure that all the surface solves succeeded
    success1 = np.all([boozer_surface.res['success'] for boozer_surface in boozer_surfaces])
    # check to make sure that the surfaces are not self-intersecting
    #success2 = np.all([not boozer_surface.res['is_self_intersecting'] for boozer_surface in boozer_surfaces])
    success2 = True
    
    if (success1 and success2):
        J_rank += J_nonQSRatio.GN_trace() * 0.01**2

        J = np.array(comm.allgather(J_rank)).real[0]
        grad = np.array(comm.allgather(J_rank)).imag/eps
        
        prevs['Jtemp'] = J
        prevs['dJtemp'] = grad.copy()
    else:
        J = prevs['J']
        grad = -prevs['dJ']
        for idx, boozer_surface in enumerate(boozer_surfaces):
            boozer_surface.surface.x = prevs['sdofs'][idx]
            boozer_surface.res['iota'] = prevs['iota'][idx]
            boozer_surface.res['G'] = prevs['G'][idx]
    return J, grad

def callback(x):
    x = x.astype(complex)
    # since this set of coil dofs was accepted, set the backup surface dofs
    # to accepted ones in case future Newton solves fail.
    for idx, surface in enumerate(surfaces):
        prevs['sdofs'][idx] = surface.x.copy()
    for idx, boozer_surface in enumerate(boozer_surfaces):
        prevs['iota'][idx] = boozer_surface.res['iota']
        prevs['G'][idx] = boozer_surface.res['G']

    
    mr_list    = [boozer_surface.res["minor_radius"] for boozer_surface in boozer_surfaces]
    Mr_list    = [boozer_surface.res["major_radius"] for boozer_surface in boozer_surfaces]
    AR_list    = [boozer_surface.res["aspect_ratio"] for boozer_surface in boozer_surfaces]
    iota_list  = [boozer_surface.res['iota'] for boozer_surface in boozer_surfaces]
    width = 35

    prevs['J'] = prevs['Jtemp']
    prevs['dJ'] = prevs['dJtemp']


    outstr = f"\nIteration {prevs['it']}\n"
    outstr += f"{'J':{width}} {prevs['J'].real:.6e} \n"
    outstr += f"{'dJ':{width}} {np.linalg.norm(prevs['dJ']):.6e} \n"
    outstr += f"{'ι on surfaces':{width}} ({IOTAS_WEIGHT*J_iotas.J().real:.6e}):  " + ", ".join([f"{iot.real:.6f}" for iot in [np.min(iota_list), np.mean(iota_list), np.max(iota_list)]]) + "\n"
    outstr += f"{'major radius on surfaces':{width}} ({MR_WEIGHT*J_major_radius.J().real:.6e}):  " + ', '.join([f'{Mr.real:.6f}' for Mr in [np.min(Mr_list), np.mean(Mr_list), np.max(Mr_list)]]) + "\n"
    #outstr += f"{'shortest coil to coil distance':{width}} ({MIN_DIST_WEIGHT * Jccdist.J():.6e}):  {Jccdist.shortest_distance():.3f} \n"
    outstr += f"{'coil length sum':{width}} ({LENGTH_WEIGHT*Jls.J().real:.6e}):  {sum(J.J().real for J in ls):.3f} \n"
    outstr += f"{'max κ':{width}} ({KAPPA_WEIGHT*Jcs.J().real:.6e}):  " + ', '.join([f'{np.max(c.kappa().real):.6f}' for c in base_curves]) + "\n"
    outstr += f"{'∫ κ^2 dl / ∫ dl':{width}} ({MSC_WEIGHT*Jmsc.J().real:.6e}):  " + ', '.join([f'{Jmsc.J().real:.6f}' for Jmsc in msc_list]) + "\n"
    outstr += f"{'Arclength':{width}} ({ARCLENGTH_WEIGHT*Jals.J().real:.6e}):  " + ', '.join([f'{np.var(np.linalg.norm(c.gammadash().real, axis=-1)):.6f}' for c in base_curves]) + "\n"
    outstr += f"{'aspect ratio radius on surfaces':{width}}" + ', '.join([f'{AR.real:.6f}' for AR in [np.min(AR_list), np.mean(AR_list), np.max(AR_list)]]) + "\n"
    outstr += f"{'minor radius on surfaces':{width}}" + ', '.join([f'{mr.real:.6f}' for mr in [np.min(mr_list), np.mean(mr_list), np.max(mr_list)]]) + "\n"
    outstr += f"{'coil lengths':{width}}" + ', '.join([f'{J.J().real:5.6f}' for J in ls]) + "\n"
    outstr += "\n\n"
    
    proc0_print(outstr, flush=True)
    prevs['it'] += 1


f = fun
dofs = JF.x.copy().real
out = fun(dofs)
callback(dofs)
#fval = [prevs['J']]
#epsval = np.linspace(0, 1e-2, 100)[:15]
#h = JF.dJ()
#for eps in epsval[1:]:
#    ff, _ = f(dofs - eps*h)
#    if ff == prevs['J']:
#        break
#    fval.append(ff)
#    proc0_print(ff)
#
#if rank == 0:
#    fval = np.array(fval)
#    np.savetxt(f'fval{comm.size}.txt', np.concatenate((epsval[:fval.size, None], fval[:, None]), axis=1))
#quit()


#print("""
#################################################################################
#### Perform a Taylor test ######################################################
#################################################################################
#""")
#np.random.seed(1)
#h = np.random.uniform(size=dofs.shape)
#J0, dJ0 = f(dofs)
#dJh = sum(dJ0 * h)
#for eps in [1e-6, 1e-7, 1e-8, 1e-9]:
#    J1, _ = f(dofs + 2*eps*h)
#    J2, _ = f(dofs + eps*h)
#    J3, _ = f(dofs - eps*h)
#    J4, _ = f(dofs - 2*eps*h)
#    proc0_print("err", ((J1*(-1/12) + J2*(8/12) + J3*(-8/12) + J4*(1/12))/eps - dJh)/np.linalg.norm(dJh))
#
#quit()

proc0_print("""
################################################################################
### Run the optimization #######################################################
################################################################################
""", flush=True)
# Number of iterations to perform:
MAXITER = 5e3
restart = 1

for j in range(restart):
    res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER}, tol=1e-15, callback=callback)
    proc0_print("RESTARTING ", j)
    
    iota_list = [boozer_surface.res['iota'] for boozer_surface in boozer_surfaces]
    G_list = [boozer_surface.res['G'] for boozer_surface in boozer_surfaces]
    if rank == 0:
        save([coils, surfaces, iota_list, G_list], OUT_DIR + f'stoch.json')
    #curves_to_vtk(curves, OUT_DIR + "curves_opt")
    #boozer_surface_orig.surface.to_vtk(OUT_DIR + "surf_opt")
    dofs = res['x'].copy()

proc0_print("End of 2_Intermediate/boozerQA.py", flush=True)
proc0_print("================================", flush=True)
