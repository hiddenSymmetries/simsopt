#!/usr/bin/env python3
from randomgen import PCG64
import os
import numpy as np
from scipy.optimize import minimize

from simsopt._core import save, load
from simsopt.objectives import MPIObjective, MPIOptimizable
from simsopt.configs import get_ncsx_data
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual, \
    Volume, MajorRadius, CurveLength, NonQuasiSymmetricRatio, Iotas
from simsopt.objectives import QuadraticPenalty, BoozerSquaredFlux
from simsopt.util import in_github_actions
from simsopt.geo import (CurveLength, CurveCurveDistance, curves_to_vtk, create_equally_spaced_curves, SurfaceRZFourier,
                         MeanSquaredCurvature, LpCurveCurvature, ArclengthVariation, GaussianSampler, CurvePerturbed, PerturbationSample)
from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.util import proc0_print, in_github_actions
import sys
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

ID = int(sys.argv[1])
# Directory for output
OUT_DIR = f"./out_stoch_{ID}/"
os.makedirs(OUT_DIR, exist_ok=True)

proc0_print(f"Running 2_Intermediate/boozerQA.py seed = {ID}", flush=True)
proc0_print("================================", flush=True)

base_curves, base_currents, ma = get_ncsx_data()
coils = coils_via_symmetries(base_curves, base_currents, 3, True)
curves = [c.curve for c in coils]
bs_orig = BiotSavart(coils)
current_sum = sum(abs(c.current.get_value()) for c in coils)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
coils_orig = coils
# let's fix the coil current
base_currents[0].fix_all()

np.random.seed(ID)
for curve in base_curves:
    for i in range(6):
        rnd = np.random.normal(0., 0.01) 
        curve.set(f'xc({i})', curve.get(f'xc({i})') + rnd)
        rnd = np.random.normal(0., 0.01) 
        curve.set(f'yc({i})', curve.get(f'yc({i})') + rnd)
        rnd = np.random.normal(0., 0.01) 
        curve.set(f'zc({i})', curve.get(f'zc({i})') + rnd)
        
        if i > 0:
            rnd = np.random.normal(0., 0.01) 
            curve.set(f'xs({i})', curve.get(f'xs({i})') + rnd)
            rnd = np.random.normal(0., 0.01) 
            curve.set(f'ys({i})', curve.get(f'ys({i})') + rnd)
            rnd = np.random.normal(0., 0.01) 
            curve.set(f'zs({i})', curve.get(f'zs({i})') + rnd)





## COMPUTE THE INITIAL SURFACE ON WHICH WE WANT TO OPTIMIZE FOR QA##
# Resolution details of surface on which we optimize for qa
mpol = 6
ntor = 6
stellsym = True
nfp = 3

phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
s_orig = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)

# To generate an initial guess for the surface computation, start with the magnetic axis and extrude outward
s_orig.fit_to_curve(ma, 0.1, flip_theta=True)

# Use a volume surface label
vol = Volume(s_orig)
vol_target = vol.J()

iota = -0.406
G = G0

#[in_coils, in_surfaces, in_iota_list, in_G_list]=load(f'determ_0.json')
#bs_in = BiotSavart(in_coils)
#bs_orig.x = bs_in.x
#s_orig.x = in_surfaces[0].x
#iota = in_iota_list[0]
#G = in_G_list[0]


## compute the surface
boozer_surface = BoozerSurface(bs_orig, s_orig, vol, vol_target, options={'newton_tol':1e-12, 'newton_maxiter':10})

# compute surface first using LBFGS, this will just be a rough initial guess
res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-10, maxiter=300, constraint_weight=100., iota=iota, G=G)
proc0_print(f"After LBFGS:   iota={res['iota']:.3f}, area={s_orig.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s_orig, res['iota'], res['G'], bs_orig, derivatives=0)):.3e}", flush=True)


boozer_surface.need_to_run_code = True
# now drive the residual down using a specialised least squares algorithm
res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=100, constraint_weight=100., iota=res['iota'], G=res['G'], method='manual')
proc0_print(f"After Lev-Mar: iota={res['iota']:.3f}, area={s_orig.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s_orig, res['iota'], res['G'], bs_orig, derivatives=0)):.3e}", flush=True)
boozer_surface.need_to_run_code = True

res = boozer_surface.run_code(res['iota'], G=res['G'])

surfaces = [s_orig]
boozer_surfaces = [boozer_surface]



Jsf = []
#if rank == 0:
#    Jsf +=[BoozerSquaredFlux(boozer_surface, BiotSavart(coils), definition='local')]


N_SAMPLES = 4
proc0_print(f"There are {N_SAMPLES * comm.size} samples")
for ii in range(N_SAMPLES):
    rg = np.random.Generator(PCG64(inc=0))

    L = 0.4*np.pi
    SIGMA = 1.e-2
    sampler = GaussianSampler(base_curves[0].quadpoints, SIGMA, L, n_derivs=1)
    curves_pert = []
    # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
    base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
    coils = coils_via_symmetries(base_curves_perturbed, base_currents, ma.nfp, True)
    # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
    coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
    curves_pert.append([c.curve for c in coils_pert])
    bs_pert = BiotSavart(coils_pert)
    Jsf += [BoozerSquaredFlux(boozer_surface, BiotSavart(coils_pert), definition='local')]
J_nonQSRatio = NonQuasiSymmetricRatio(boozer_surface, bs_orig)

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
IOTAS_WEIGHT = 1e1
ARCLENGTH_WEIGHT = 1e-2

ls = [CurveLength(c) for c in base_curves]
mrs = MajorRadius(boozer_surface)
iotas = Iotas(boozer_surface)

J_squared_flux = MPIObjective(Jsf, comm, needs_splitting=False)
J_major_radius = QuadraticPenalty(mrs, MR_TARGET, 'identity')
J_iotas = QuadraticPenalty(iotas, IOTAS_TARGET, 'identity')
Jls = QuadraticPenalty(sum(ls), float(sum(ls).J()), 'max')
Jccdist = CurveCurveDistance(curves, MIN_DIST_THRESHOLD, num_basecurves=len(base_curves))
Jcs = sum([LpCurveCurvature(c, 2, KAPPA_THRESHOLD) for c in base_curves])
msc_list = [MeanSquaredCurvature(c) for c in base_curves]
Jmsc = sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in msc_list)
Jals = sum([ArclengthVariation(c) for c in base_curves])

# sum the objectives together
JF = J_nonQSRatio + J_squared_flux + IOTAS_WEIGHT*J_iotas + MR_WEIGHT*J_major_radius + LENGTH_WEIGHT*Jls\
        + MIN_DIST_WEIGHT * Jccdist + KAPPA_WEIGHT*Jcs + MSC_WEIGHT*Jmsc + ARCLENGTH_WEIGHT*Jals

curves_to_vtk(curves, OUT_DIR + "curves_init")
boozer_surface.surface.to_vtk(OUT_DIR + "surf_init")


# dictionary used to save the last accepted surface dofs in the line search, in case Newton's method fails
prevs = {'sdofs': [surface.x.copy() for surface in surfaces], 'iota': [boozer_surface.res['iota'] for boozer_surface in boozer_surfaces],
         'G': [boozer_surface.res['G'] for boozer_surface in boozer_surfaces], 'J': JF.J(), 'dJ': JF.dJ().copy(), 'it': 0}
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

    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    
    # check to make sure that all the surface solves succeeded
    success1 = np.all([i for o in comm.allgather([boozer_surface.res['success'] for boozer_surface in boozer_surfaces]) for i in o])
    # check to make sure that the surfaces are not self-intersecting
    success2 = np.all([i for o in comm.allgather([not boozer_surface.res['is_self_intersecting'] for boozer_surface in boozer_surfaces]) for i in o])
    if not (success1 and success2):
        J = prevs['J']
        grad = -prevs['dJ']
        for idx, boozer_surface in enumerate(boozer_surfaces):
            boozer_surface.surface.x = prevs['sdofs'][idx]
            boozer_surface.res['iota'] = prevs['iota'][idx]
            boozer_surface.res['G'] = prevs['G'][idx]
    return J, grad

def callback(x):
    # since this set of coil dofs was accepted, set the backup surface dofs
    # to accepted ones in case future Newton solves fail.
    for idx, surface in enumerate(surfaces):
        prevs['sdofs'][idx] = surface.x.copy()
    for idx, boozer_surface in enumerate(boozer_surfaces):
        prevs['iota'][idx] = boozer_surface.res['iota']
        prevs['G'][idx] = boozer_surface.res['G']
    prevs['J'] = JF.J()
    prevs['dJ'] = JF.dJ().copy()
    
    sqr_flux_list = [i for o in comm.allgather([nqs.J() for nqs in Jsf]) for i in o]
    width = 35

#JF = J_nonQSRatio + IOTAS_WEIGHT*J_iotas + MR_WEIGHT*J_major_radius + LENGTH_WEIGHT*Jls\
#        + MIN_DIST_WEIGHT * Jccdist + KAPPA_WEIGHT*Jcs + MSC_WEIGHT*Jmsc + ARCLENGTH_WEIGHT*Jals


    outstr = f"\nIteration {prevs['it']}\n"
    outstr += f"{'J':{width}} {JF.J():.6e} \n"
    outstr += f"{'║∇J║':{width}} {np.linalg.norm(JF.dJ()):.6e} \n\n"
    outstr += f"{'nonQS ratio':{width}} ({J_nonQSRatio.J():.6e}) " + ", ".join([f'{np.sqrt(nonqs.J()):.6e}' for nonqs in [J_nonQSRatio]]) + "\n"
    outstr += f"{'squared flux':{width}} ({J_squared_flux.J():.6e}):  " + ", ".join([f'{np.sqrt(nonqs):.6e}' for nonqs in [np.min(sqr_flux_list), np.mean(sqr_flux_list), np.max(sqr_flux_list)]]) + "\n"
    outstr += f"{'ι on surfaces':{width}}" + ", ".join([f"{boozer_surface.res['iota']:.6f}" for boozer_surface in boozer_surfaces]) + "\n"
    outstr += f"{'major radius on surfaces':{width}}" + ', '.join([f'{surface.major_radius():.6f}' for surface in surfaces]) + "\n"
    outstr += f"{'minor radius on surfaces':{width}}" + ', '.join([f'{surface.minor_radius():.6f}' for surface in surfaces]) + "\n"
    outstr += f"{'aspect ratio radius on surfaces':{width}}" + ', '.join([f'{surface.aspect_ratio():.6f}' for surface in surfaces]) + "\n"
    outstr += f"{'shortest coil to coil distance':{width}} ({MIN_DIST_WEIGHT * Jccdist.J():.6e}):  {Jccdist.shortest_distance():.3f} \n"
    outstr += f"{'coil length sum':{width}} ({LENGTH_WEIGHT*Jls.J():.6e}):  {sum(J.J() for J in ls):.3f} \n"
    outstr += f"{'max κ':{width}} ({KAPPA_WEIGHT*Jcs.J():.6e}):  " + ', '.join([f'{np.max(c.kappa()):.6f}' for c in base_curves]) + "\n"
    outstr += f"{'∫ κ^2 dl / ∫ dl':{width}} ({MSC_WEIGHT*Jmsc.J():.6e}):  " + ', '.join([f'{Jmsc.J():.6f}' for Jmsc in msc_list]) + "\n"
    outstr += f"{'Arclength':{width}} ({ARCLENGTH_WEIGHT*Jals.J():.6e}):  " + ', '.join([f'{np.var(np.linalg.norm(c.gammadash(), axis=-1)):.6f}' for c in base_curves]) + "\n"
    outstr += f"{'coil lengths':{width}}" + ', '.join([f'{J.J():5.6f}' for J in ls]) + "\n"
    outstr += "\n\n"
    
    proc0_print(outstr, flush=True)
    prevs['it'] += 1

f = fun
dofs = JF.x.copy()
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
MAXITER = 5e3

res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER}, tol=1e-15, callback=callback)

sqr_flux_list = [i for o in comm.allgather([nqs.J() for nqs in Jsf]) for i in o]
iota_list = [boozer_surface.res['iota'] for boozer_surface in boozer_surfaces]
G_list = [boozer_surface.res['G'] for boozer_surface in boozer_surfaces]
if rank == 0:
    save([coils_orig, s_orig, iota_list, G_list], OUT_DIR + f'boozer_surface_{rank}.json')
    curves_to_vtk(curves, OUT_DIR + "curves_opt")
    boozer_surface.surface.to_vtk(OUT_DIR + "surf_opt")
    
    np.savetxt(OUT_DIR + 'sqr_flux.txt', sqr_flux_list)

proc0_print("End of 2_Intermediate/boozerQA.py", flush=True)
print("================================", flush=True)
