#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual, \
    Volume, MajorRadius, CurveLength, NonQuasiIsodynamicRatio, Iotas
from simsopt.mhd import Vmec
from simsopt.objectives import QuadraticPenalty
from simsopt.util import in_github_actions

r"""
This example optimizes the NCSX coils and currents for QI on a single surface. The objective is

    J = J_QI
        + 0.5*(iota - iota_0)**2
        + 0.5*(major_radius - target_major_radius)**2
        + 0.5*max(sum(CurveLength) - CurveLengthTarget, 0)**2

The workflow intentionally stays close to boozerQA.py. We first compute a
single Boozer surface close to the magnetic axis, then optimize the coils using
the milestone-1 quasi-isodynamic objective NonQuasiIsodynamicRatio on that
surface. The rotational transform, major radius, and total coil length are kept
close to their initial values with quadratic penalties.

Environment variables can be used to reduce runtime for testing:

    SIMSOPT_BOOZER_QI_MPOL
    SIMSOPT_BOOZER_QI_NTOR
    SIMSOPT_BOOZER_QI_SDIM
    SIMSOPT_BOOZER_QI_NPHI
    SIMSOPT_BOOZER_QI_NALPHA
    SIMSOPT_BOOZER_QI_NBJ
    SIMSOPT_BOOZER_QI_NPHI_OUT
    SIMSOPT_BOOZER_QI_MAXITER
    SIMSOPT_BOOZER_QI_SKIP_TAYLOR
    SIMSOPT_BOOZER_QI_WRITE_VTK
    SIMSOPT_BOOZER_QI_OUT_DIR
    SIMSOPT_BOOZER_QI_VMEC
"""


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


OUT_DIR = os.environ.get("SIMSOPT_BOOZER_QI_OUT_DIR", "./output/")
WRITE_VTK = _env_bool("SIMSOPT_BOOZER_QI_WRITE_VTK", True)
SKIP_TAYLOR = _env_bool("SIMSOPT_BOOZER_QI_SKIP_TAYLOR", False)
VMEC_GEOMETRY = os.environ.get("SIMSOPT_BOOZER_QI_VMEC")

if WRITE_VTK:
    os.makedirs(OUT_DIR, exist_ok=True)

print("Running 2_Intermediate/boozerQI.py")
print("================================")

base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
all_curves = [c.curve for c in bs.coils]
bs_qi = BiotSavart(bs.coils)
current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

mpol_default = 3 if in_github_actions else 6
ntor_default = 3 if in_github_actions else 6
mpol = _env_int("SIMSOPT_BOOZER_QI_MPOL", mpol_default)
ntor = _env_int("SIMSOPT_BOOZER_QI_NTOR", ntor_default)
stellsym = True

phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
s = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)

if VMEC_GEOMETRY:
    vmec_path = Path(VMEC_GEOMETRY).expanduser().resolve()
    print(f"Using VMEC geometry initializer from {vmec_path}")
    vmec = Vmec(str(vmec_path))
    if vmec.boundary.nfp != nfp:
        raise ValueError(
            f"VMEC geometry nfp={vmec.boundary.nfp} does not match the coil configuration nfp={nfp}."
        )
    s.least_squares_fit(vmec.boundary.gamma())
else:
    s.fit_to_curve(ma, 0.1, flip_theta=True)

iota = -0.406

vol = Volume(s)
vol_target = vol.J()

boozer_surface = BoozerSurface(bs, s, vol, vol_target)
res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=iota, G=G0)

out_res = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)[0]
print(f"NEWTON {res['success']}: iter={res['iter']}, iota={res['iota']:.3f}, vol={s.volume():.3f}, ||residual||={np.linalg.norm(out_res):.3e}")

sDIM_default = 8 if in_github_actions else 20
nphi_default = 31 if in_github_actions else 151
nalpha_default = 5 if in_github_actions else 31
nBj_default = 7 if in_github_actions else 51
nphi_out_default = 61 if in_github_actions else 2000

sDIM = _env_int("SIMSOPT_BOOZER_QI_SDIM", sDIM_default)
nphi_qi = _env_int("SIMSOPT_BOOZER_QI_NPHI", nphi_default)
nalpha = _env_int("SIMSOPT_BOOZER_QI_NALPHA", nalpha_default)
nBj = _env_int("SIMSOPT_BOOZER_QI_NBJ", nBj_default)
nphi_out = _env_int("SIMSOPT_BOOZER_QI_NPHI_OUT", nphi_out_default)

mr = MajorRadius(boozer_surface)
ls = [CurveLength(c) for c in base_curves]

J_major_radius = QuadraticPenalty(mr, mr.J(), 'identity')
J_iotas = QuadraticPenalty(Iotas(boozer_surface), res['iota'], 'identity')
J_nonQIRatio = NonQuasiIsodynamicRatio(
    boozer_surface,
    bs_qi,
    sDIM=sDIM,
    nphi=nphi_qi,
    nalpha=nalpha,
    nBj=nBj,
    nphi_out=nphi_out,
)
Jls = QuadraticPenalty(sum(ls), float(sum(ls).J()), 'max')

JF = J_nonQIRatio + J_iotas + J_major_radius + Jls

if WRITE_VTK:
    curves_to_vtk(all_curves, os.path.join(OUT_DIR, "curves_init"))
    boozer_surface.surface.to_vtk(os.path.join(OUT_DIR, "surf_init"))

base_currents[0].fix_all()


def fun(dofs):
    sdofs_prev = boozer_surface.surface.x
    iota_prev = boozer_surface.res['iota']
    G_prev = boozer_surface.res['G']

    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()

    if not boozer_surface.res['success']:
        J = 1e3
        boozer_surface.surface.x = sdofs_prev
        boozer_surface.res['iota'] = iota_prev
        boozer_surface.res['G'] = G_prev

    cl_string = ", ".join([f"{term.J():.1f}" for term in ls])
    outstr = f"J={J:.1e}, J_nonQIRatio={J_nonQIRatio.J():.2e}, iota={boozer_surface.res['iota']:.2e}, mr={mr.J():.2e}"
    outstr += f", Len=sum([{cl_string}])={sum(term.J() for term in ls):.1f}"
    outstr += f", ||grad J||={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad


if not SKIP_TAYLOR:
    print("""
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
    for eps in [1e-3, 1e-4, 1e-5, 1e-6]:
        J1, _ = f(dofs + 2*eps*h)
        J2, _ = f(dofs + eps*h)
        J3, _ = f(dofs - eps*h)
        J4, _ = f(dofs - 2*eps*h)
        print("err", ((J1*(-1/12) + J2*(8/12) + J3*(-8/12) + J4*(1/12))/eps - dJh)/max(np.linalg.norm(dJh), 1e-15))

print("""
################################################################################
### Run the optimization #######################################################
################################################################################
""")

maxiter_default = 3 if in_github_actions else 1000
maxiter = _env_int("SIMSOPT_BOOZER_QI_MAXITER", maxiter_default)

initial_dofs = JF.x.copy()
initial_value, initial_grad = fun(initial_dofs)
print(f"Initial gradient norm = {np.linalg.norm(initial_grad):.3e}")

res = minimize(fun, initial_dofs, jac=True, method='BFGS', options={'maxiter': maxiter}, tol=1e-15)

if WRITE_VTK:
    curves_to_vtk(all_curves, os.path.join(OUT_DIR, "curves_opt"))
    boozer_surface.surface.to_vtk(os.path.join(OUT_DIR, "surf_opt"))

print(f"Optimization success={res.success}, status={res.status}, nit={res.nit}, initial_J={initial_value:.6e}, final_J={res.fun:.6e}")
print("End of 2_Intermediate/boozerQI.py")
print("================================")