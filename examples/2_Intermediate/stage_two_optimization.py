from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux, FOCUSObjective
from simsopt.geo.curve import RotatedCurve, curves_to_vtk
from simsopt.field.biotsavart import BiotSavart, Current, Coil
from simsopt.geo.coilcollection import coils_via_symmetries, create_equally_spaced_curves
from simsopt.geo.curveobjectives import CurveLength
import numpy as np

from pathlib import Path
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'wout_li383_low_res_reference.nc'
nfp = 3
nphi = 32
ntheta = 32
phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
thetas = np.linspace(0, 1., ntheta, endpoint=False)
s = SurfaceRZFourier.from_wout(filename, quadpoints_phi=phis, quadpoints_theta=thetas)

ncoils = 4
R0 = 1.5
R1 = 0.8
order = 6
PPP = 15
ALPHA = 1e-5

base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=True, R0=R0, R1=R1, order=order, PPP=PPP)
base_currents = []
for i in range(ncoils):
    curr = Current(1e5)
    if i == 0:
        curr.fix_all()
    base_currents.append(curr)

coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
curves = [c.curve for c in coils]
curves_to_vtk(curves, "/tmp/curves_init")
s.to_vtk("/tmp/surf")

bs = BiotSavart(coils)

Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]

JF = FOCUSObjective(Jf, Jls, ALPHA)


# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize
def fun(dofs):
    JF.x = dofs
    J = JF.J()
    dJ = JF.dJ()
    grad = dJ(JF)
    cl_string = ", ".join([f"{v:.3f}" for v in JF.vals[1:]])
    mean_AbsB = np.mean(bs.AbsB())
    print(f"J={J:.3e}, Jflux={JF.vals[0]:.3e}, sqrt(Jflux)/Mean(|B|)={np.sqrt(JF.vals[0])/mean_AbsB:.3e}, CoilLengths=[{cl_string}], ||∇J||={np.linalg.norm(grad):.3e}")
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
    print("err", (J1-J2)/(2*eps) - dJh)

print("""    
################################################################################
### Run some optimisation ######################################################
################################################################################
""")
from scipy.optimize import minimize
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': 400, 'maxcor': 400}, tol=1e-15)
curves_to_vtk(curves, "/tmp/curves_opt")
