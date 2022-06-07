#!/usr/bin/env python3
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.curve import curves_to_vtk
from simsopt.geo.surfaceobjectives import boozer_surface_residual, ToroidalFlux, Volume, MajorRadius
from simsopt.geo.curveobjectives import CurveLength, CurveCurveDistance
from simsopt.field.coil import coils_via_symmetries
from simsopt.util.zoo import get_ncsx_data
from simsopt.geo.surfaceobjectives import NonQuasiAxisymmetricRatio, Iotas
from simsopt.objectives.utilities import QuadraticPenalty
from scipy.optimize import minimize
import numpy as np
import os

"""
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


## RESOLUTION DETAILS OF SURFACE ON WHICH WE OPTIMIZE FOR QA
mpol = 6  
ntor = 6  
stellsym = True
nfp = 3

phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
s = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
s.fit_to_curve(ma, 0.1, flip_theta=True)
iota = -0.406

vol = Volume(s)
vol_target = vol.J()

## COMPUTE THE SURFACE
boozer_surface = BoozerSurface(bs, s, vol, vol_target)
res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=iota, G=G0)
print(f"NEWTON {res['success']}: iter={res['iter']}, iota={res['iota']:.3f}, vol={s.volume():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")

## SET UP THE OPTIMIZATION PROBLEM AS A SUM OF OPTIMIZABLES
bs_nonQS = BiotSavart(coils)
mr = MajorRadius(boozer_surface)
ls = [CurveLength(c) for c in base_curves]

J_major_radius = QuadraticPenalty(mr, 1.5, 'identity')
J_iotas = QuadraticPenalty(Iotas(boozer_surface), res['iota'], 'identity')
J_nonQSRatio = NonQuasiAxisymmetricRatio(boozer_surface, bs_nonQS)
Jls = QuadraticPenalty(sum(ls), 21., 'max') 

JF = J_nonQSRatio + J_iotas + J_major_radius + Jls

curves_to_vtk(curves, OUT_DIR + f"curves_init")
boozer_surface.surface.to_vtk(OUT_DIR + "surf_init")

# let's fix the coil current
base_currents[0].fix_all()

def fun(dofs):
    # save these in case the boozer surface solve fails
    sdofs_prev = boozer_surface.surface.x
    iota_prev = boozer_surface.res['iota']
    G_prev = boozer_surface.res['G']

    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    
    if not boozer_surface.res['success']:
        # failed, so reset back to previous surface
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
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    print("err", ((J1-J2)/(2*eps) - dJh)/np.linalg.norm(dJh))

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
# Number of iterations to perform:
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
MAXITER = 50 if ci else 1e5

res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER}, tol=1e-15)
curves_to_vtk(curves, OUT_DIR + f"curves_opt")
boozer_surface.surface.to_vtk(OUT_DIR + "surf_opt")

print("End of 2_Intermediate/boozerQA.py")
print("================================")
