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
from simsopt.geo.curveobjectives import CurveLength, MinimumDistance, LpCurveCurvature
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


mpol = 10  # try increasing this to 8 or 10 for smoother surfaces
ntor = 10  # try increasing this to 8 or 10 for smoother surfaces
stellsym = True
nfp = 3
exact = True

surf_list = []
for idx,target in enumerate(np.linspace(-0.162,-3.206, 20)):
    if exact:
        phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    else:
        phis = np.linspace(0, 1/(2*nfp), ntor+5, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+5, endpoint=False)
    
    s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    bs = BiotSavart(stellarator.coils, stellarator.currents)
    
    if len(surf_list) == 0:
        s.fit_to_curve(ma, 0.10, flip_theta=True)
        iota0 = -0.4
        G0 = 2. * np.pi * np.sum(np.abs(bs.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
    else:
        s.set_dofs(surf_list[-1].surface.get_dofs())
        iota0 = surf_list[-1].res['iota'] 
        G0 = surf_list[-1].res['G']

    label = Volume(s, stellarator)
    boozer_surface = BoozerSurface(bs, s, label, target)
    
    try:
        res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=30,iota=iota0,G=G0)
        if res['success']:
            surf_list += [boozer_surface]
            print(f"iota={res['iota']:.3f}, label={label.J():.3f}, area={s.area():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
        else:
            print("didn't converge")
    except:
        print("error thrown.")

    iota0 = res['iota']
    G0 = res['G']

iota_weight = 1.
distance_weight = 1e-3
curvature_weight = 1e-3

minimum_distance = 0.02
iota_target = -0.4

coil_lengths    = [CurveLength(coil) for coil in coils]
coil_length_targets = [J.J() for J in coil_lengths]
min_dist = MinimumDistance(stellarator.coils, minimum_distance)
coil_curvatures = [LpCurveCurvature(coil, 2., length) for (coil, length) in zip(coils, coil_length_targets)]
problem_list = [NonQuasiAxisymmetricComponentPenalty(boozer_surface, stellarator, iota_target, iota_weight) for boozer_surface in surf_list[::6]]

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
    
    for idx, problem in enumerate(problem_list):
        # revert to reference states - zeroth order continuation
        s = problem.boozer_surface.surface
        gamma = s.gamma()
        mlab.mesh(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2])
    mlab.show()
plot()


prev_J_dJ = [ None, None ]
def callback(x, g, fx, *args):
    J = 0.
    dJ = np.zeros(problem_list[0].dJ().shape)

    for problem in problem_list:
        # update the reference boozer surface
        problem.boozer_surface_reference = {"dofs": problem.boozer_surface.surface.get_dofs(),
                                            "iota": problem.boozer_surface.res["iota"],
                                            "G": problem.boozer_surface.res["G"]}
    
    print(J, np.linalg.norm(dJ))
    prev_J_dJ[0] = fx
    prev_J_dJ[1] = g
    
    print("-------------------------------------------\n")

def fun_pylbfgs(dofs,g,*args):
    stellarator.set_dofs(dofs)
    
    J = 0
    dJ = None
    
    # regularizations
    # surfaces
    for idx, problem in enumerate(problem_list):
        # revert to reference states - zeroth order continuation
        iota0 = problem.boozer_surface_reference["iota"]
        G0 = problem.boozer_surface_reference["G"]
        problem.boozer_surface.surface.set_dofs(problem.boozer_surface_reference["dofs"])
        #print(f"reverting to {iota0:.8f}, {G0:.8f}, {np.linalg.norm(problem.boozer_surface_reference['dofs']):.8f}")

        target = problem.boozer_surface.targetlabel
        label = problem.boozer_surface.label
        s = problem.boozer_surface.surface

        res = problem.boozer_surface.solve_residual_equation_exactly_newton( tol=1e-10, maxiter=10, iota=iota0, G=G0)
        J += problem.J()
        
        if dJ is None:
            dJ  = problem.dJ()
        else:
            dJ += problem.dJ()
         
        print(f"Surface {idx}: {res['success']}, iota={res['iota']:.6e}, J={J:.6e}, label={label.J():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||={np.linalg.norm(res['residual']):.3e}")
        if not res['success']:
            print("Failed to compute surface-----------------------------")
            J = 2*prev_J_dJ[0]
            dJ = -prev_J_dJ[1]
            break
    g[:] = dJ
    return J




coeffs = stellarator.get_dofs()
callback(coeffs)





coeffs = stellarator.get_dofs()
maxiter = 300
try:
    res = fmin_lbfgs(fun_pylbfgs, coeffs, line_search='wolfe', epsilon=1e-5, max_linesearch=100, m=5000, progress = callback, max_iterations=maxiter)
except Exception as e:
    print(e)
    pass

plot()
