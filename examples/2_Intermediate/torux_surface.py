#!/usr/bin/env python3

import numpy as np
from scipy.optimize import minimize

from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, ToroidalFlux, ToruxSurface, Iotas, MajorRadius, \
    CurveLength
from simsopt.geo.toruxsurface import torux_available
from simsopt.objectives import QuadraticPenalty
from simsopt.util import in_github_actions

"""
This example computes a flux surface of the NCSX coils with torux, as a SurfaceRZFourier in
straight-field-line angles, and then changes the coils to raise the rotational transform on the
surface by 2%, keeping its major radius and the length of the coils, as boozerQA.py does with a
BoozerSurface. The derivatives of Iotas and MajorRadius with respect to the coils come from the
adjoint of torux's equations.

torux is an optional dependency: pip install torux (Python 3.12 or newer).
"""

print("Running 2_Intermediate/torux_surface.py")
print("=======================================")

if not torux_available:
    print("torux is not installed: skipping this example")
    print("End of 2_Intermediate/torux_surface.py")
    print("=======================================")
    raise SystemExit(0)

base_curves, base_currents, ma, nfp, bs = get_data("ncsx")

# the toroidal flux through a tube of minor radius 0.1 m about the magnetic axis
tube = SurfaceXYZTensorFourier(mpol=6, ntor=6, stellsym=True, nfp=nfp)
tube.fit_to_curve(ma, 0.1, flip_theta=True)
flux = abs(ToroidalFlux(tube, BiotSavart(bs.coils)).J())

# the flux surface through which the toroidal flux is the same
s = SurfaceRZFourier(nfp=nfp, stellsym=True, mpol=16, ntor=24)
torux_surface = ToruxSurface(bs, s, ToroidalFlux(s, BiotSavart(bs.coils)), flux, options={'verify': False})
res = torux_surface.run_code()
print(f"{s.x.size} surface dofs (torux chose mpol, ntor = {res['torux_mpol_ntor']}), iota={res['iota']:.6f}")

mr = MajorRadius(torux_surface)
ls = [CurveLength(c) for c in base_curves]
J_iotas = QuadraticPenalty(Iotas(torux_surface), 1.02 * res['iota'], 'identity')
J_major_radius = QuadraticPenalty(mr, mr.J(), 'identity')
Jls = QuadraticPenalty(sum(ls), float(sum(ls).J()), 'max')
JF = 1e2 * J_iotas + J_major_radius + Jls

base_currents[0].fix_all()


def fun(dofs):
    JF.x = dofs
    if not torux_surface.run_code()['success']:
        # no surface: return a large value so that the line search takes a shorter step
        return 1e3, np.zeros_like(dofs)
    J, grad = JF.J(), JF.dJ()
    print(f"J={J:.3e}, iota={torux_surface.res['iota']:.6f}, mr={mr.J():.6f}, ║∇J║={np.linalg.norm(grad):.1e}")
    return J, grad


MAXITER = 5 if in_github_actions else 50
res = minimize(fun, JF.x, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER}, tol=1e-15)
print(f"iota={torux_surface.res['iota']:.6f}, target {J_iotas.cons:.6f}")

print("End of 2_Intermediate/torux_surface.py")
print("=======================================")
