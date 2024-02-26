from simsopt.configs import get_w7x_data
from scipy.optimize import minimize
from simsopt.geo import FrameRotation, FramedCurveCentroid
from simsopt.field import Coil
from simsopt.field.fieldalignment import CrtitcalCurrentOpt, critical_current
from simsopt.util import in_github_actions
import numpy as np

MAXITER = 50 if in_github_actions else 400

curves, currents, ma = get_w7x_data()

curve = curves[0]
current = currents[0]
coils = [Coil(c, curr) for c, curr in zip(curves[1:], currents[1:])]

rot_order = 10  # order of the Fourier expression for the rotation of the filament pack

curve.fix_all()  # fix curve DOFs -> only optimize winding angle
current.fix_all()
rotation = FrameRotation(curve.quadpoints, rot_order)

framedcurve = FramedCurveCentroid(curve, rotation)
coil = Coil(framedcurve, current)
JF = CrtitcalCurrentOpt(coil, coils, a=0.05, b=0.05)
print("Minimum Ic before Optimization:",
      np.min(critical_current(coil, a=0.05, b=0.05)))


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    return J, grad


f = fun
dofs = JF.x

res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
               options={'maxiter': MAXITER, 'maxcor': 10, 'gtol': 1e-20, 'ftol': 1e-20}, tol=1e-20)

print("Minimum Ic after Optimization:",
      np.min(critical_current(coil, a=0.05, b=0.05)))
