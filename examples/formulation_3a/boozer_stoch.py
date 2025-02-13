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
boozer_surface = BoozerSurface(bs_orig, s_orig, vol, vol_target, options={'newton_tol':1e-12, 'newton_maxiter':10})


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
nonQSs = [NonQuasiSymmetricRatio(boozer_surface_orig, BiotSavart(coils), sqrtC=sqrtC)]


J_nonQSRatio = nonQSs[0]
print(nonQSs[0].J())
print(nonQSs[0].GN_trace())

import ipdb;ipdb.set_trace()
