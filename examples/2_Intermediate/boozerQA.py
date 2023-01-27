#!/usr/bin/env python3
from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual, \
    ToroidalFlux, Volume, MajorRadius, CurveLength, CurveCurveDistance, NonQuasiSymmetricRatio, Iotas
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.configs import get_ncsx_data
from simsopt.objectives import QuadraticPenalty
from scipy.optimize import minimize
import numpy as np
import os

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
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

print("Running 2_Intermediate/boozerQA.py")
print("================================")

base_curves, base_currents, ma = get_ncsx_data()
coils = coils_via_symmetries(base_curves, base_currents, 3, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
bs_tf = BiotSavart(coils)
current_sum = sum(abs(c.current.get_value()) for c in coils)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

## COMPUTE THE INITIAL SURFACE ON WHICH WE WANT TO OPTIMIZE FOR QA##
# Resolution details of surface on which we optimize for qa
mpol = 6  
ntor = 6  
stellsym = True
nfp = 3

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
boozer_surface = BoozerSurface(bs, s, vol, vol_target)
res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=iota, G=G0)

out_res = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)[0]
print(f"NEWTON {res['success']}: iter={res['iter']}, iota={res['iota']:.3f}, vol={s.volume():.3f}, ||residual||={np.linalg.norm(out_res):.3e}")
## SET UP THE OPTIMIZATION PROBLEM AS A SUM OF OPTIMIZABLES ##
bs_nonQS = BiotSavart(coils)
mr = MajorRadius(boozer_surface)
ls = [CurveLength(c) for c in base_curves]

J_major_radius = QuadraticPenalty(mr, mr.J(), 'identity')  # target major radius is that computed on the initial surface
J_iotas = QuadraticPenalty(Iotas(boozer_surface), res['iota'], 'identity')  # target rotational transform is that computed on the initial surface
J_nonQSRatio = NonQuasiSymmetricRatio(boozer_surface, bs_nonQS)
Jls = QuadraticPenalty(sum(ls), float(sum(ls).J()), 'max') 

# sum the objectives together
JF = J_nonQSRatio + J_iotas + J_major_radius + Jls

curves_to_vtk(curves, OUT_DIR + f"curves_init")
boozer_surface.surface.to_vtk(OUT_DIR + "surf_init")

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

    if not boozer_surface.res['success']:
        # failed, so reset back to previous surface and return a large value
        # of the objective.  The purpose is to trigger the line search to reduce
        # the step size.
        J = 1e3
        boozer_surface.surface.x = sdofs_prev
        boozer_surface.res['iota'] = iota_prev
        boozer_surface.res['G'] = G_prev

    cl_string = ", ".join([f"{J.J():.1f}" for J in ls])
    outstr = f"J={J:.1e}, J_nonQSRatio={J_nonQSRatio.J():.2e}, iota={boozer_surface.res['iota']:.2e}, mr={mr.J():.2e}"
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in ls):.1f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
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
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
MAXITER = 50 if ci else 1e3

res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER}, tol=1e-15)
curves_to_vtk(curves, OUT_DIR + f"curves_opt")
boozer_surface.surface.to_vtk(OUT_DIR + "surf_opt")

print("End of 2_Intermediate/boozerQA.py")
print("================================")
