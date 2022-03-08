#!/usr/bin/env python

from pathlib import Path
import numpy as np
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries, ScaledCurrent
from simsopt.geo.curve import curves_to_vtk
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

nphi = 32
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

ncoils = 4
nfp = 2
base_curves = [CurveXYZFourier(200, 16) for i in range(ncoils)]
base_currents = []
for i in range(ncoils):
    curr = Current(1.)
    if i == 0:
        curr.fix_all()
    base_currents.append(ScaledCurrent(curr, 1e5))

coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
bs = BiotSavart(coils)
bs.x = np.loadtxt(TEST_DIR / "coils_wechsung_pnas_qa24.txt")

bs.set_points(s.gamma().reshape((-1, 3)))
np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

curves = [c.curve for c in coils]
curves_to_vtk(curves, "/tmp/curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk("/tmp/surf_init", extra_data=pointData)
