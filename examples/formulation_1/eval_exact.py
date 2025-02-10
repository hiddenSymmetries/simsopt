#!/usr/bin/env python3
from randomgen import PCG64
import os
import numpy as np
from scipy.optimize import minimize
import sys

from simsopt._core import save, load
from simsopt.objectives import MPIObjective, MPIOptimizable
from simsopt.configs import get_ncsx_data
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual, \
    Volume, MajorRadius, CurveLength, NonQuasiSymmetricRatio, Iotas
from simsopt.objectives import QuadraticPenalty
from simsopt.util import in_github_actions
from simsopt.geo import (CurveLength, CurveCurveDistance, curves_to_vtk, create_equally_spaced_curves, SurfaceRZFourier,
                         MeanSquaredCurvature, LpCurveCurvature, ArclengthVariation, GaussianSampler, CurvePerturbed, PerturbationSample, BoozerResidual)
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

sigma      = float(sys.argv[1])
N_s = int(sys.argv[2])
OUT_DIR = f'sigma={sigma}_Nsamples={N_s}/'
[coils_opt, s_opt, iota_opt, G_opt] = load(OUT_DIR + f'stoch.json')

s_opt = s_opt[0]

# Directory for output
os.makedirs(OUT_DIR, exist_ok=True)

proc0_print("Running 2_Intermediate/boozerQA.py", flush=True)
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

bs_opt = BiotSavart(coils_opt)
bs_orig.x = bs_opt.x

N = 6
nfp_eff = 3
## COMPUTE THE INITIAL SURFACE ON WHICH WE WANT TO OPTIMIZE FOR QA##
# Resolution details of surface on which we optimize for qa
mpol = N
ntor = N*nfp_eff
stellsym = False
nfp = 1

phis = np.linspace(0, 1, 2*ntor+1+10, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1+10, endpoint=False)
stemp = SurfaceXYZTensorFourier(mpol=s_opt.mpol, ntor=s_opt.ntor, stellsym=s_opt.stellsym, nfp=s_opt.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
stemp.x = s_opt.x

phis = np.linspace(0, 1, 2*ntor+1+10, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1+10, endpoint=False)
s_orig = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
s_orig.least_squares_fit(stemp.gamma())

# Use a volume surface label
vol = Volume(s_orig)
vol_target = Volume(s_opt).J()

iota = iota_opt[0]
G = G_opt[0]

## compute the surface
boozer_surface = BoozerSurface(bs_orig, s_orig, vol, vol_target, constraint_weight=100., \
        options={'newton_tol':1e-10, 'newton_maxiter':15, 'limited_memory':True, 
            'bfgs_tol':1e-16, 'bfgs_maxiter':1000})

# compute surface first using LBFGS, this will just be a rough initial guess
res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-13, maxiter=400, constraint_weight=100., iota=iota, G=G, weight_inv_modB=True)
proc0_print(f"After LBFGS:   iota={res['iota']:.3f}, area={s_orig.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s_orig, res['iota'], res['G'], bs_orig, derivatives=0)):.3e}", flush=True)

iota = res['iota']
G = res['G']
boozer_surface.need_to_run_code = True
res = boozer_surface.minimize_boozer_penalty_constraints_newton(constraint_weight=100., iota=iota, G=G, \
        verbose=True, tol=1e-10, maxiter=15,\
        weight_inv_modB=True)
iota = res['iota']
G = res['G']

res_orig = res
xorig = s_orig.x.copy()
boozer_surface_orig = boozer_surface

surfaces = []
boozer_surfaces = []
nonQSs = []

ii = 0
N_SAMPLES = 1
proc0_print(f"There are {N_SAMPLES * comm.size} samples for sigma={sigma:.2e}")

rg = np.random.Generator(PCG64(inc=0))
while len(surfaces) != N_SAMPLES:
    L = 0.4*np.pi
    SIGMA = sigma
    sampler = GaussianSampler(base_curves[0].quadpoints, SIGMA, L, n_derivs=1)
    curves_pert = []
    # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
    base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
    coils = coils_via_symmetries(base_curves_perturbed, base_currents, ma.nfp, True)
    # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
    coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
    curves_pert.append([c.curve for c in coils_pert])
    bs_pert = BiotSavart(coils_pert)
    current_sum = sum(abs(c.current.get_value()) for c in coils_pert)

    stellsym = False
    nfp = 1
    phis = np.linspace(0, 1., 2*ntor+1+10, endpoint=False)
    thetas = np.linspace(0, 1, 2*mpol+1+10, endpoint=False)
    s = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    s.x = xorig.copy()

    iota = res_orig['iota']
    G = res_orig['G']
    
    # Use a volume surface label
    vol = Volume(s)
    #vol_target = vol.J()

    ## compute the surface
    boozer_surface_pert = BoozerSurface(bs_pert, s, vol, vol_target, constraint_weight=100., options={'newton_tol':1e-10, 'newton_maxiter':15, 'limited_memory':True,\
            'bfgs_tol':1e-16, 'bfgs_maxiter':3000})
    boozer_surface_pert.need_to_run_code = True
    res = boozer_surface_pert.minimize_boozer_penalty_constraints_LBFGS(tol=1e-16, maxiter=3000, constraint_weight=100., \
            iota=iota, G=G, weight_inv_modB=True, verbose=True)
    
    success = res['success']
    boozer_surface_pert.need_to_run_code = True
    iota, G = res['iota'], res['G']
    res = boozer_surface_pert.minimize_boozer_penalty_constraints_newton(constraint_weight=100., iota=iota, G=G, \
            verbose=True, tol=0, maxiter=0,\
            weight_inv_modB=True)
    iota = res['iota']
    G = res['G']

    lr = np.concatenate([ii/np.arange(1, 16) for ii in range(1, 8)])
    iot = np.abs(res['iota'])
    is_lr = np.any([np.abs(iot-ii)/ii < 0.01 for ii in lr])

    if success and not res['is_self_intersecting']:
        surfaces += [s]
        boozer_surfaces += [boozer_surface_pert]
        nonQSs += [NonQuasiSymmetricRatio(boozer_surface_pert, bs_pert)]
        rg = np.random.Generator(PCG64(inc=0))
        print(f"SAVED {rank}::: SUCCESS {res['success']}, SI {res['is_self_intersecting']}, LOW ORD {is_lr} with {res['iota']:.5f}")
    else:
        print(f"SKIPPED {rank}::: SUCCESS {res['success']}, SI {res['is_self_intersecting']}, LOW ORD {is_lr} with {res['iota']:.5f}, {res_orig['iota']} {res_orig['G']}")
oos_nonqs_list = [i for o in comm.allgather([nqs.J() for nqs in nonQSs]) for i in o]
proc0_print("MEAN STOCH ", np.mean(oos_nonqs_list))
if rank == 0:
    np.savetxt(OUT_DIR + 'nonqs.txt', oos_nonqs_list)
