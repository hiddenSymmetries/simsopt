import numpy as np
from math import pi
from simsopt.geo.coilcollection import CoilCollection
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.biotsavart import BiotSavart
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.curve import RotatedCurve
from simsopt.geo.curverzfourier import CurveRZFourier 
from simsopt.geo.curvexyzfourier import CurveXYZFourier 
from simsopt.geo.surfaceobjectives import ToroidalFlux 
from simsopt.geo.surfaceobjectives import Area 
from surface_test_helpers import get_ncsx_data,get_surface2



stellsym = True
surfacetype = "SurfaceXYZFourier"

coils, currents, ma = get_ncsx_data()
stellarator = CoilCollection(coils, currents, 3, True)

bs = BiotSavart(stellarator.coils, stellarator.currents)
bs_tf = BiotSavart(stellarator.coils, stellarator.currents)

s = get_surface2(surfacetype,stellsym)
s.fit_to_curve(ma, 0.2)
iota = -0.4

tf = ToroidalFlux(s, bs_tf)
tf_target = 0.1
boozerSurface = BoozerSurface(bs, s, tf, tf_target) 
print("Initial toroidal flux is :", tf.J(), "Target toroidal flux is: ", tf_target)
print("Surface computed using LBFGS and penalised toroidal flux constraint") 
s,iota,message = boozerSurface.minimize_boozer_penalty_constraints_LBFGS(tol = 1e-12, maxiter = 100, constraint_weight = 100., iota = iota)
s,iota,message = boozerSurface.minimize_boozer_penalty_constraints_ls(tol = 1e-12, maxiter = 100, constraint_weight = 100., iota = iota)
print("Toroidal flux is :", tf.J(), "Target toroidal flux is: ", tf_target, "\n")

print("Surface computed using Newton and penalised toroidal flux constraint") 
s, iota,message = boozerSurface.minimize_boozer_penalty_constraints_newton(tol = 1e-11, maxiter = 10, constraint_weight = 100., iota = iota)
print("Toroidal flux is :", tf.J(), "Target toroidal flux is: ", tf_target, "\n")

for tf_target in np.linspace(0.1,0.5, 10):
    boozerSurface = BoozerSurface(bs, s, tf, tf_target) 
    s, iota, lm, message = boozerSurface.minimize_boozer_exact_constraints_newton(tol = 1e-11, maxiter = 10,iota = iota)
    xi = np.concatenate( (s.get_dofs(), [iota]) )
    res = boozerSurface.boozer_penalty_constraints(xi, derivatives = 0, scalarize = False)
    print("Toroidal flux is :", tf.J(),\
      "\nTarget toroidal flux is: ", tf_target,\
      "\niota is: ", iota,
      "\nresidual is: ", np.linalg.norm(res),
      "\n")

    # Matt this is the SurfaceRZFourier object
    sRZ = s.to_RZFourier()
    sRZ.plot()







