#!/usr/bin/env python3

import argparse
import os
import numpy as np
from scipy.optimize import minimize

from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual, \
    Volume, MajorRadius, CurveLength, NonQuasiSymmetricRatio, Iotas
from simsopt.objectives import QuadraticPenalty
from simsopt.util import in_github_actions

r"""
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
Use --use-jax with --device {cpu,gpu} to select the optional JAX field and Boozer surface.
"""

try:
    from simsopt_jax.backend import set_backend
    from simsopt_jax_adapters.field import JaxBiotSavart
    from simsopt_jax_adapters.geo.boozer_surface import JaxBoozerSurface
except ImportError as error:
    jax_import_error = str(error)
else:
    jax_import_error = None

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--use-jax", action="store_true", help="Use the optional JAX field and Boozer surface")
parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
args = parser.parse_args()
if args.use_jax:
    if jax_import_error is not None:
        parser.error(f"--use-jax requires Python >= 3.12 and jax/jaxlib >= 0.12: {jax_import_error}")
    set_backend("jax", device="gpu" if args.device == "gpu" else "cpu")

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

print("Running 2_Intermediate/boozerQA.py")
print("================================")

base_curves, base_currents, ma, nfp, bs  = get_data("ncsx")
bs = JaxBiotSavart(bs.coils) if args.use_jax else bs
# bs.coils includes all coils after symmetry expansion (not just the base coils).
# You can access them directly like this:
all_curves = [c.curve for c in bs.coils]
bs_tf = JaxBiotSavart(bs.coils) if args.use_jax else BiotSavart(bs.coils)
current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

## COMPUTE THE INITIAL SURFACE ON WHICH WE WANT TO OPTIMIZE FOR QA##
# Resolution details of surface on which we optimize for qa
mpol = 6
ntor = 6
stellsym = True

phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
s = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)

# To generate an initial guess for the surface computation, start with the magnetic axis and extrude outward
s.fit_to_curve(ma, 0.1, flip_theta=True)
iota = -0.406

# Use a volume surface label
vol = Volume(s)
vol_target = vol.J()

## compute the surface
boozer_surface = JaxBoozerSurface(bs, s, vol, vol_target) if args.use_jax else BoozerSurface(bs, s, vol, vol_target)
res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=iota, G=G0)

bs_diagnostic = BiotSavart(bs.coils) if args.use_jax else bs
out_res = boozer_surface_residual(s, res['iota'], res['G'], bs_diagnostic, derivatives=0)[0]
print(f"NEWTON {res['success']}: iter={res['iter']}, iota={res['iota']:.3f}, vol={s.volume():.3f}, ||residual||={np.linalg.norm(out_res):.3e}")
## SET UP THE OPTIMIZATION PROBLEM AS A SUM OF OPTIMIZABLES ##
bs_nonQS = JaxBiotSavart(bs.coils) if args.use_jax else BiotSavart(bs.coils)
mr = MajorRadius(boozer_surface)
ls = [CurveLength(c) for c in base_curves]

J_major_radius = QuadraticPenalty(mr, mr.J(), 'identity')  # target major radius is that computed on the initial surface
J_iotas = QuadraticPenalty(Iotas(boozer_surface), res['iota'], 'identity')  # target rotational transform is that computed on the initial surface
J_nonQSRatio = NonQuasiSymmetricRatio(boozer_surface, bs_nonQS)
Jls = QuadraticPenalty(sum(ls), float(sum(ls).J()), 'max')

# sum the objectives together
JF = J_nonQSRatio + J_iotas + J_major_radius + Jls

curves_to_vtk(all_curves, OUT_DIR + "curves_init")
boozer_surface.surface.to_vtk(OUT_DIR + "surf_init")

# let's fix the coil current
base_currents[0].fix_all()


def progress(J, grad):
    cl_string = ", ".join([f"{J.J():.1f}" for J in ls])
    outstr = f"J={J:.1e}, J_nonQSRatio={J_nonQSRatio.J():.2e}, iota={boozer_surface.res['iota']:.2e}, mr={mr.J():.2e}"
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in ls):.1f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    return outstr


# Print the progress line on every 50th evaluation and after each optimization.
PRINT_EVERY = 50
evaluation_count = 0


def fun(dofs):
    global evaluation_count
    # save these as a backup in case the boozer surface Newton solve fails
    sdofs_prev = boozer_surface.surface.x
    iota_prev = boozer_surface.res['iota']
    G_prev = boozer_surface.res['G']

    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()

    if not boozer_surface.res['success']:
        # failed, so reset back to previous surface and return a large value
        # of the objective.  The purpose is to trigger the line search to reduce
        # the step size.
        J = 1e3
        boozer_surface.surface.x = sdofs_prev
        boozer_surface.res['iota'] = iota_prev
        boozer_surface.res['G'] = G_prev

    evaluation_count += 1
    if evaluation_count % PRINT_EVERY == 0:
        print(progress(J, grad))
    return J, grad


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
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
    J1, _ = f(dofs + 2*eps*h)
    J2, _ = f(dofs + eps*h)
    J3, _ = f(dofs - eps*h)
    J4, _ = f(dofs - 2*eps*h)
    print("err", ((J1*(-1/12) + J2*(8/12) + J3*(-8/12) + J4*(1/12))/eps - dJh)/np.linalg.norm(dJh))

print("""
################################################################################
### Run the optimization #######################################################
################################################################################
""")
# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 1e3

res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER}, tol=1e-15)
print(f"{res.nfev} evaluations: J={res.fun:.1e}, ║∇J║={np.linalg.norm(res.jac):.1e}")
curves_to_vtk(all_curves, OUT_DIR + "curves_opt")
boozer_surface.surface.to_vtk(OUT_DIR + "surf_opt")

print("End of 2_Intermediate/boozerQA.py")
print("================================")
