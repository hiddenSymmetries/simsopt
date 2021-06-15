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
from simsopt.geo.curveobjectives import CurveLength, MinimumDistance, UniformArclength
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import NonQuasiAxisymmetricComponentPenalty, Area, Volume, ToroidalFlux, boozer_surface_residual, MajorRadius
from simsopt.geo.coilcollection import CoilCollection
sys.path.append(os.path.join("..", "tests", "geo"))
from surface_test_helpers import get_ncsx_data, get_surface, get_exact_surface
from scipy.optimize import minimize, least_squares
from lbfgs import fmin_lbfgs




coils, currents, ma = get_ncsx_data(Nt_coils=6, Nt_ma=10, ppp=20)
stellarator = CoilCollection(coils, currents, 3, True)
coils = stellarator.coils
currents = stellarator.currents


mpol = 8  # try increasing this to 8 or 10 for smoother surfaces
ntor = 8  # try increasing this to 8 or 10 for smoother surfaces
stellsym = True
nfp = 3
exact = True

surf_list = []
#for idx,target in enumerate(np.linspace(-0.162,-4.0, 20)):
#for idx,target in enumerate(np.linspace(-0.162,-2., 20)): NON UNIQUENESS HERE
for idx,target in enumerate(np.linspace(-0.162,-1., 20)):
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
            print(f"iota={res['iota']:.3f}, label={label.J():.3f}, area={s.area():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}, AR={s.aspect_ratio():.3f}")
        else:
            print("didn't converge")
    except:
        print("error thrown.")

    iota0 = res['iota']
    G0 = res['G']

target_minimum_distance = 0.15
min_dist = MinimumDistance(stellarator.coils, target_minimum_distance)

iota_target = -0.4
iota_weight = 1.

problem_list = [NonQuasiAxisymmetricComponentPenalty(boozer_surface, stellarator, target_iota=iota_target, iota_weight=iota_weight, major_radius_weight=mrw) for boozer_surface, mrw in zip([surf_list[0], surf_list[int(len(surf_list)/2)], surf_list[-1]], [1,0,0])]
problem_list = [problem_list[0]]
curvelength_list = [CurveLength(curve) for curve in stellarator.coils]
target_lengths = [c.J() for c in curvelength_list]
arclength_list = [UniformArclength(curve, l) for curve, l in zip(stellarator.coils, target_lengths)]
arclength_weights = [1e-3 for curve in stellarator.coils]



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
    mlab.figure(1, bgcolor=(1,1,1))
    mlab.gcf().scene.parallel_projection = True


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
        mlab.mesh(np.concatenate((gamma[:,:,0], gamma[:,0,0].reshape((-1,1)) ), axis = 1), 
                  np.concatenate((gamma[:,:,1], gamma[:,0,1].reshape((-1,1)) ), axis = 1),
                  np.concatenate((gamma[:,:,2], gamma[:,0,2].reshape((-1,1)) ), axis = 1) )
#        mlab.mesh(np.concatenate((gamma[:,:,0], gamma[:,0,0].reshape((-1,1)) ), axis = 1), 
#                  np.concatenate((gamma[:,:,1], gamma[:,0,1].reshape((-1,1)) ), axis = 1),
#                  np.concatenate((gamma[:,:,2], gamma[:,0,2].reshape((-1,1)) ), axis = 1), representation='wireframe' )
    mlab.view(azimuth=100, elevation=55)
    mlab.show()
plot()


prev_J_dJ = [ None, None ]
def callback(x, g, fx, xnorm, gnorm, step, k, num_eval, *args):
    length_penalties = []
    dlength_penalties = []
    # regularizations
    for l, l0, al, alw in zip(curvelength_list, target_lengths, arclength_list, arclength_weights):
        length_penalties += [0.5*(l.J() - l0)**2 + alw*al.J()]
        dlength_penalties += [(l.J() - l0) * l.dJ() + alw*al.dJ_by_dcoefficients()]
    J = np.sum(length_penalties)
    dJ_coil = stellarator.reduce_coefficient_derivatives(dlength_penalties)
    dJ_curr = np.zeros(len(stellarator.get_currents()))
    
    J  += min_dist.J()
    dJ_coil += stellarator.reduce_coefficient_derivatives(min_dist.dJ())

    for idx,problem in enumerate(problem_list):
        # update the reference boozer surface
        problem.boozer_surface_reference = {"dofs": problem.boozer_surface.surface.get_dofs(),
                                         "iota": problem.boozer_surface.res["iota"],
                                         "G": problem.boozer_surface.res["G"]}
    
        J += problem.J()
        dJ1, dJ2 = problem.dJ()
        dJ_curr+=dJ1
        dJ_coil+=dJ2
        if k%200 == 0:
            np.savetxt(f"surface{idx}_{k}.txt", problem.boozer_surface.surface.get_dofs())
    
    dJ = np.concatenate([dJ_curr, dJ_coil])
    if k%200 == 0:
        np.savetxt(f"stellarator_{k}.txt", stellarator.get_dofs())
#        plot()
#        import ipdb;ipdb.set_trace()

    print(f"iter={k}, J={J:.6e}, ||dJ||={np.linalg.norm(dJ, ord=np.inf):6e}, min_dist={min_dist.J():6e}")
    prev_J_dJ[0] = J
    prev_J_dJ[1] = dJ
    print("-------------------------------------------\n")

def fun_pylbfgs(dofs,g,*args):
    stellarator.set_currents(dofs[idx_curr[0]:idx_curr[1]]*current_fak)
    stellarator.set_dofs(dofs[idx_coil[0]:idx_coil[1]])


    length_penalties = []
    dlength_penalties = []
    # regularizations
    for l, l0, al, alw in zip(curvelength_list, target_lengths, arclength_list, arclength_weights):
        length_penalties += [0.5*(l.J() - l0)**2 + alw*al.J()]
        dlength_penalties += [(l.J() - l0) * l.dJ() + alw*al.dJ_by_dcoefficients()]
    J = np.sum(length_penalties)
    dJ_coil = stellarator.reduce_coefficient_derivatives(dlength_penalties)
        
    dJ_curr = np.zeros(len(stellarator.get_currents()))
    J  += min_dist.J()
    dJ_coil += stellarator.reduce_coefficient_derivatives(min_dist.dJ())

    # surfaces
    for idx, problem in enumerate(problem_list):
        # revert to reference states - zeroth order continuation
        iota0 = problem.boozer_surface_reference["iota"]
        G0 = problem.boozer_surface_reference["G"]
        problem.boozer_surface.surface.set_dofs(problem.boozer_surface_reference["dofs"])
        #print(f"reverting to {iota0:.8f}, {G0:.8f}, {np.linalg.norm(problem.boozer_surface_reference['dofs']):.8f}")

        res = problem.boozer_surface.solve_residual_equation_exactly_newton( tol=1e-10, maxiter=10, iota=iota0, G=G0)
        J += problem.J()
        gradcurr, gradcoil= problem.dJ()
        dJ_curr+=gradcurr
        dJ_coil+=gradcoil

        target = problem.boozer_surface.targetlabel
        label = problem.boozer_surface.label
        s = problem.boozer_surface.surface



        print(f"Surface {idx}: {res['success']}, iota={res['iota']:.6e}, J={J:.6e}, area={s.area():.3f}, label={label.J():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||={np.linalg.norm(res['residual']):.3e}, AR={s.aspect_ratio():.3f}")
        if not res['success']:
            print("Failed to compute surface-----------------------------")
            J = 2*prev_J_dJ[0]
            dJ = -prev_J_dJ[1]
            break
    dJ = np.concatenate([dJ_curr, dJ_coil])
    g[:] = dJ
    return J

current_fak = 1./(4 * np.pi * 1e-7)
coeffs_curr = stellarator.get_currents()/current_fak
coeffs_coil = stellarator.get_dofs()
coeffs = np.concatenate([coeffs_curr, coeffs_coil])
idx_curr = (0,len(stellarator.get_currents()))
idx_coil = (idx_curr[1], idx_curr[1] + len(stellarator.get_dofs()))
 
callback(0,coeffs, 0, 0, 0, 0, 0, 0, 0, 0)


#grad = np.zeros( (192,) )
#f0 = fun_pylbfgs(coeffs, grad)
#lm = np.random.uniform(size=coeffs.size)-0.5
#fd_exact = np.dot(grad, lm) 
#
#err_old = 1e9
#epsilons = np.power(2., -np.asarray(range(7, 20)))
#print("################################################################################")
#for eps in epsilons:
#    f1 = fun_pylbfgs(coeffs + eps * lm, grad)
#    Jfd = (f1-f0)/eps
#    err = np.linalg.norm(Jfd-fd_exact)/np.linalg.norm(fd_exact)
#    print(err/err_old)
##    assert err < err_old * 0.55
#    err_old = err
#print("################################################################################")










maxiter = 8000
try:
    res = fmin_lbfgs(fun_pylbfgs, coeffs, line_search='wolfe', epsilon=1e-12, max_linesearch=100, m=5000, progress = callback, max_iterations=maxiter)
except Exception as e:
    print(e)
    pass

#coeffs = stellarator.get_dofs()
#np.savetxt("stellarator.txt", coeffs)
#import ipdb;ipdb.set_trace()
plot()
