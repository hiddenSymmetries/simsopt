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
from simsopt.field import VmecVirtualCasingField
from simsopt.mhd import Vmec
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Resolution on the plasma boundary surface:
# nphi is the number of grid points in 1/2 a field period.
nphi = 50
ntheta = 50


# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = 'wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc'
vmec_file = TEST_DIR / filename

vmec = Vmec(vmec_file)

vc = VmecVirtualCasingField(vmec, src_nphi=nphi, src_ntheta=ntheta,  trgt_nphi=nphi//2, trgt_ntheta=ntheta//2, digits=5, max_upsampling=2000)
# Get offset surface to evaluate magnetic field on
surf = vc.surf.copy(nphi=nphi, ntheta=ntheta, range='half period')
surf.extend_via_normal(0.2)

vc.set_points(surf.gamma().reshape((-1, 3)), )
B = vc.B() 

modB = np.linalg.norm(B, axis=-1)
modB = modB.reshape((nphi, ntheta))

plt.ion()
plt.figure()
plt.contourf(surf.quadpoints_phi, surf.quadpoints_theta, modB.T, cmap='RdBu')
plt.xlabel('phi')
plt.ylabel('theta')
plt.colorbar()
plt.title("Virtual Casing Magnetic Field Magnitude")
plt.show()

vc.Bnormal_due_ext