#!/usr/bin/env python3
from simsopt.field import BiotSavart
from simsopt.field import coils_via_symmetries
from simsopt.geo import SurfaceRZFourier
from simsopt.geo import QfmSurface
from simsopt.geo import QfmResidual, ToroidalFlux, Area, Volume
from simsopt.configs import get_ncsx_data
import numpy as np
import os

"""
This example demonstrate how to compute a quadratic flux minimizing surfaces
from a magnetic field induced by coils. We start with an initial guess that
is just a tube around the magnetic axis using the NCSX coils. We first reduce
the penalty objective function to target a given toroidal flux using LBFGS.
The equality constrained objective is then reduced using SLSQP. This is repeated
for fixing the area and toroidal flux.
"""
print("Running 1_Simple/qfm.py")
print("=======================")

curves, currents, ma = get_ncsx_data()
coils = coils_via_symmetries(curves, currents, 3, True)
bs = BiotSavart(coils)
bs_tf = BiotSavart(coils)

mpol = 5
ntor = 5
stellsym = True
nfp = 3
constraint_weight = 1e0

phis = np.linspace(0, 1/nfp, 25, endpoint=False)
thetas = np.linspace(0, 1, 25, endpoint=False)
s = SurfaceRZFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis,
    quadpoints_theta=thetas)
s.fit_to_curve(ma, 0.2, flip_theta=True)

# First optimize at fixed volume

qfm = QfmResidual(s, bs)
qfm.J()

vol = Volume(s)
vol_target = vol.J()

qfm_surface = QfmSurface(bs, s, vol, vol_target)

res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=1000,
                                                         constraint_weight=constraint_weight)
print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=1000)
print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

# Now optimize at fixed toroidal flux

tf = ToroidalFlux(s, bs_tf)
tf_target = tf.J()

qfm_surface = QfmSurface(bs, s, tf, tf_target)

res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=1000,
                                                         constraint_weight=constraint_weight)
print(f"||tf constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=1000)
print(f"||tf constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

# Check that volume is not changed
print(f"||vol constraint||={0.5*(vol.J()-vol_target)**2:.8e}")

# Now optimize at fixed area

ar = Area(s)
ar_target = ar.J()

qfm_surface = QfmSurface(bs, s, ar, ar_target)

res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=1000,
                                                         constraint_weight=constraint_weight)
print(f"||area constraint||={0.5*(ar.J()-ar_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=1000)
print(f"||area constraint||={0.5*(ar.J()-ar_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

# Check that volume is not changed
print(f"||vol constraint||={0.5*(vol.J()-vol_target)**2:.8e}")

if "DISPLAY" in os.environ:
    s.plot()
print("End of 1_Simple/qfm.py")
print("=======================")
