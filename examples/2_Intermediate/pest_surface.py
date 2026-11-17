#!/usr/bin/env python3

import numpy as np
from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import PestSurface, ToroidalFlux, SurfaceRZFourier

"""
This example demonstrates how to compute a flux surface of a magnetic field
induced by coils with a PestSurface, as a SurfaceRZFourier in PEST angles.
We start with an initial guess that is just a tube around the magnetic axis,
and solve at a low resolution by Newton's method, keeping the toroidal flux of
the tube. The solution is then the initial guess at a higher resolution. The
field crosses a flux surface less and less as the resolution is raised, which
is how to check that the surface found is one.
"""

print("Running 2_Intermediate/pest_surface.py")
print("======================================")

base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
bs_tf = BiotSavart(bs.coils)

s = SurfaceRZFourier(nfp=nfp, stellsym=True, mpol=6, ntor=6)
s.fit_to_curve(ma, 0.10, flip_theta=True)
iota = -0.4
tf_target = ToroidalFlux(s, bs_tf).J()

for mpol, ntor in [(6, 6), (10, 10), (14, 14)]:
    s = s.copy(mpol=mpol, ntor=ntor)
    pest_surface = PestSurface(bs, s, ToroidalFlux(s, bs_tf), tf_target)
    res = pest_surface.run_code(iota)
    iota = res['iota']

    # the field across the surface, between the collocation points
    fine = s.copy(quadpoints_phi=np.linspace(0, 1/nfp, 4*ntor, endpoint=False),
                  quadpoints_theta=np.linspace(0, 1, 4*mpol, endpoint=False))
    bs.set_points(fine.gamma().reshape((-1, 3)))
    B = bs.B().reshape(fine.gamma().shape)
    BdotN = np.max(np.abs(np.sum(B * fine.unitnormal(), axis=2)) / np.linalg.norm(B, axis=2))
    print(f"mpol={mpol}, ntor={ntor}: iota={iota:.8f}, max|B.n|/|B|={BdotN:.1e}")

print("End of 2_Intermediate/pest_surface.py")
print("======================================")
