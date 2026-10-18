#!/usr/bin/env python
"""
Field of the plasma currents outside the plasma, by virtual casing.

VmecVirtualCasingField evaluates the field of the plasma currents of a VMEC equilibrium at
arbitrary points outside its boundary (simsopt.mhd.VirtualCasing gives it only on the boundary).
Here it is evaluated 10 cm outside the boundary of the Landreman-Buller-Drevlak quasi-helical
equilibrium at 5% beta. Adding a coil field, e.g. a BiotSavart object, gives the total field
outside the plasma.

This example requires the python ``virtual_casing`` package.
"""
from pathlib import Path

import numpy as np

from simsopt.field import VmecVirtualCasingField
from simsopt.mhd import Vmec

TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / "wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc"

# Source grid points per half field period. Within about a grid cell of the boundary the
# accuracy depends on this resolution.
nphi = 64
vc = VmecVirtualCasingField(Vmec(filename), src_nphi=nphi, src_ntheta=nphi, digits=6)

# Points 10 cm outside the plasma boundary:
n = 32
surf = vc.surf.copy(nphi=n, ntheta=n, range="half period")
surf.extend_via_normal(0.1)
vc.set_points(surf.gamma().reshape((-1, 3)))
modB = np.linalg.norm(vc.B(), axis=-1).reshape((n, n))
print(f"|B| of the plasma currents 10 cm outside the boundary: {modB.min():.4f} to {modB.max():.4f} T")

# On the boundary itself the field of the external currents, which a coil set has to supply:
print(f"max |B_external . n| on the boundary: {np.abs(vc.Bnormal_due_ext).max():.4f} T")
