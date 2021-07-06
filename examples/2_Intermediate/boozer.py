#!/usr/bin/env python3
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import boozer_surface_residual, ToroidalFlux, Area
from simsopt.geo.coilcollection import CoilCollection
from simsopt.util.zoo import get_ncsx_data
import numpy as np
import os

"""
This example demonstrate how to compute surfaces in Boozer coordinates for a
magnetic field induced by coils.
We start with an initial guess that is just a tube around the magnetic axis,
then we reduce the Boozer residual with BFGS first, then drive it to machine
zero with a Levenberg-Marquardt algorithm. All of this is done while keeping
the surface area constant.  We then switch the label to the Toroidal Flux, and
aim for a Boozer surface with three times larger flux.
"""


coils, currents, ma = get_ncsx_data()
stellarator = CoilCollection(coils, currents, 3, True)
coils = stellarator.coils
currents = stellarator.currents
bs = BiotSavart(coils, currents)
bs_tf = BiotSavart(coils, currents)
G0 = 2. * np.pi * np.sum(np.abs(bs.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))

mpol = 5  # try increasing this to 8 or 10 for smoother surfaces
ntor = 5  # try increasing this to 8 or 10 for smoother surfaces
stellsym = True
nfp = 3

phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
s = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
s.fit_to_curve(ma, 0.10, flip_theta=True)
iota = -0.4

tf = ToroidalFlux(s, bs_tf)
ar = Area(s)
ar_target = ar.J()

boozer_surface = BoozerSurface(bs, s, ar, ar_target)

# compute surface first using LBFGS, this will just be a rough initial guess
res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-10, maxiter=300, constraint_weight=100., iota=iota, G=G0)
print(f"After LBFGS:   iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
if "DISPLAY" in os.environ:
    s.plot()
# now drive the residual down using a specialised least squares algorithm
res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=100, constraint_weight=100., iota=res['iota'], G=res['G'], method='manual')
print(f"After Lev-Mar: iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
if "DISPLAY" in os.environ:
    s.plot()


# change the label of the surface to toroidal flux and aim for a surface with triple the flux
print('Now aim for a larger surface')
tf_target = 3 * tf.J()
boozer_surface = BoozerSurface(bs, s, tf, tf_target)

res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=100, constraint_weight=100., iota=res['iota'], G=res['G'], method='manual')
print(f"After Lev-Mar: iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
if "DISPLAY" in os.environ:
    s.plot()
