#!/usr/bin/env python
"""
This example illustrates the use of the VirtualCasingField class,
which is a wrapper around the VirtualCasing class that provides a
more convenient interface for computing the magnetic field from the plasma currents.
The existing VirtualCasing class only computes the magnetic field
on the plasma boundary surface, but the VirtualCasingField class
can compute the magnetic field at arbitrary points in space.

This example requires that the python ``virtual_casing`` module is installed.
"""
from simsopt.field import VirtualCasingField
from simsopt.geo import SurfaceRZFourier
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Resolution on the plasma boundary surface:
# nphi is the number of grid points in 1/2 a field period.
nphi = 32
ntheta = 32

# Resolution for the virtual casing calculation:
vc_src_nphi = 80

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = 'wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc'
vmec_file = TEST_DIR / filename

vc = VirtualCasingField.from_vmec(vmec_file, 
src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta, digits=2)

# Get offset surface to evaluate magnetic field on 
surf = SurfaceRZFourier.from_wout(vmec_file, nphi=nphi, ntheta=ntheta)
surf.extend_via_normal(0.1)

# eval_points = (np.random.rand(len(surf.gamma().reshape((-1, 3))), 3) - 0.5) * 1000
# vc.set_points(eval_points)
vc.set_points(surf.gamma().reshape((-1, 3)))
B = vc.B() 

modB = np.linalg.norm(B, axis=-1)
modB = modB.reshape((nphi, ntheta))

plt.figure()
plt.contourf(surf.quadpoints_phi, surf.quadpoints_theta, modB.T, cmap='RdBu')
plt.xlabel('phi')
plt.ylabel('theta')
plt.colorbar()
plt.title("Virtual Casing Magnetic Field Magnitude")
plt.show()