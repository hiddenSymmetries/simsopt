#!/usr/bin/env python3
import sys
import os
import numpy as np
from simsopt.geo.biotsavart import BiotSavart
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.curve import RotatedCurve
from simsopt.geo.curverzfourier import CurveRZFourier 
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curveobjectives import CurveLength 
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import NonQuasiAxisymmetricComponentPenalty, Area, Volume, ToroidalFlux, boozer_surface_residual
from simsopt.geo.coilcollection import CoilCollection
sys.path.append(os.path.join("..", "tests", "geo"))
from surface_test_helpers import get_ncsx_data, get_surface, get_exact_surface
from scipy.optimize import minimize, least_squares
from lbfgs import fmin_lbfgs




coils, currents, ma = get_ncsx_data()
stellarator = CoilCollection(coils, currents, 3, True)
coils = stellarator.coils
currents = stellarator.currents


mpol = 5  # try increasing this to 8 or 10 for smoother surfaces
ntor = 5  # try increasing this to 8 or 10 for smoother surfaces
stellsym = True
nfp = 3
exact = False

if exact:
    phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
    thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
else:
    phis = np.linspace(0, 1/(2*nfp), ntor+5, endpoint=False)
    thetas = np.linspace(0, 1, 2*mpol+5, endpoint=False)

s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
s.fit_to_curve(ma, 0.10, flip_theta=True)
iota0 = -0.4

bs = BiotSavart(stellarator.coils, stellarator.currents)
bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
G0 = 2. * np.pi * np.sum(np.abs(bs.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))


label = Volume(s, stellarator)
label_target = label.J()
boozer_surface = BoozerSurface(bs, s, label, label_target)

weight = 100.
for idx,target in enumerate(np.linspace(-0.3,-1.5,30)):
    boozer_surface = BoozerSurface(bs, s, label, target)
    res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=300, constraint_weight = weight, iota=iota0,G=G0, method='manual')
    print(f"iota={res['iota']:.3f}, label={label.J():.3f}, area={s.area():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}, ||gradient|| = {np.linalg.norm(res['gradient']):.3e}")
    iota0 = res['iota']
    G0 = res['G']


boozer_surface.res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=300, constraint_weight = weight, iota=iota0,G=G0, method='manual')
problem = NonQuasiAxisymmetricComponentPenalty(boozer_surface, stellarator) 
print(boozer_surface.res['success'],problem.J())

def fun_scipy(dofs):
    stellarator.set_dofs(dofs)

    # revert to reference states
    iota0 = problem.boozer_surface_reference["iota"]
    G0 = problem.boozer_surface_reference["G"]
    problem.boozer_surface.surface.set_dofs(problem.boozer_surface_reference["dofs"])

    problem.boozer_surface.res = problem.boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=300, constraint_weight = weight, iota=iota0,G=G0, method='manual')
    J = problem.J()
    dJ = problem.dJ()
    if not boozer_surface.res['success']:
        print("Failed to compute surface")
        J = 2*J
        dJ = -dJ
    return J, dJ
def fun_pylbfgs(dofs,g,*args):
    stellarator.set_dofs(dofs)
    
    # revert to reference states - zeroth order continuation
    iota0 = problem.boozer_surface_reference["iota"]
    G0 = problem.boozer_surface_reference["G"]
    problem.boozer_surface.surface.set_dofs(problem.boozer_surface_reference["dofs"])
    # first order continuation

    problem.boozer_surface.res = problem.boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=300, constraint_weight = weight, iota=iota0,G=G0, method='manual')
    J = problem.J()
    dJ = problem.dJ()
    
    print(f"{boozer_surface.res['success']}, iota={res['iota']:.6e}, J={J:.6e} label={label.J():.3f}, area={s.area():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||={np.linalg.norm(boozer_surface_residual(s, boozer_surface.res['iota'], boozer_surface.res['G'], bs, derivatives=0)):.3e}, ||gradient|| = {np.linalg.norm(res['gradient']):.3e}")
    if not boozer_surface.res['success']:
        print("Failed to compute surface")
        J = 2*J
        dJ = -dJ
    g[:] = dJ
    return J

#import ipdb;ipdb.set_trace()
#tol = 0.
#res = minimize(fun_scipy, coeffs, jac=True, method='L-BFGS-B',options={'maxiter': maxiter, 'maxcor': 200, 'ftol': tol, 'gtol': tol }, callback=problem.callback)
coeffs = stellarator.get_dofs()
maxiter = 50
try:
    res = fmin_lbfgs(fun_pylbfgs, coeffs, line_search='wolfe', epsilon=1e-5, max_linesearch=100, m=5000, progress = problem.callback, max_iterations=maxiter)
except Exception as e:
    print(e)
    pass
#print(boozer_surface.res['success'],problem.J())
#print(res['success'], res['message'])

colors = [
    (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
    (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
    (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
    (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
    (0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
    (0.5764705882352941, 0.47058823529411764, 0.3764705882352941),
    (0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
    (0.5490196078431373, 0.5490196078431373, 0.5490196078431373),
    (0.8, 0.7254901960784313, 0.4549019607843137),
    (0.39215686274509803, 0.7098039215686275, 0.803921568627451)
]
def plot():
    import mayavi.mlab as mlab
    #mlab.figure(bgcolor=(1, 1, 1))
    for i in range(0, len(stellarator.coils)):
        gamma = stellarator.coils[i].gamma()
        gamma = np.concatenate((gamma, [gamma[0,:]]))
        mlab.plot3d(gamma[:, 0], gamma[:, 1], gamma[:, 2], color=colors[i%len(stellarator._base_coils)])
    mlab.plot3d(gamma[:, 0], gamma[:, 1], gamma[:, 2], color=colors[len(stellarator._base_coils)])
    
    gamma = s.gamma()
    mlab.mesh(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2])
    mlab.show()
plot()
