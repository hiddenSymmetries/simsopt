#!/usr/bin/env python

from pathlib import Path
import numpy as np
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries, ScaledCurrent
from simsopt.geo.curve import curves_to_vtk
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier


'''
    This script tests the MGRID module in simsopt/field/magneticfield.py

    This test is forked from Florian Weschung's PNAS Precise QA particle tracing test.
    Contact Tony Qian <tqian@pppl.gov> 17 May 2022

'''

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
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
    base_currents.append(ScaledCurrent(curr, 1e6))

coils = coils_via_symmetries(base_curves, base_currents, nfp, True)

bs = BiotSavart(coils)
bs.x = np.loadtxt(TEST_DIR / "coils_wechsung_pnas_qa24.txt")

bs.set_points(s.gamma().reshape((-1, 3)))
np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

curves = [c.curve for c in coils]
curves_to_vtk(curves, "/tmp/curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk("/tmp/surf_init", extra_data=pointData)

# Write Mgrid
f_test = "/tmp/mgrid.pnas-qa-test-lowres.nc"
bs.to_mgrid(f_test, nphi=6, nr=11, nz=11, rmin=0.5, rmax=1.5, zmin=-0.5, zmax=0.5, nfp=2)  # this low resolution test saves memory (20 KB)
#bs.to_mgrid("mgrid.pnas-qa-test.nc", nphi=72, nr=201, nz=201, rmin=0.5, rmax=1.5, zmin=-0.5, zmax=0.5, nfp=2) # use this high resolution to test free boundary vmec (~5s on login node of Stellar at Princeton, 60MB)


### Compare against standard
from simsopt.field.mgrid import ReadMGRID 
f_standard = str(TEST_DIR / 'mgrid.pnas-qa-test-lowres-standard.nc')

m_standard = ReadMGRID(f_standard)
m_test     = ReadMGRID(f_test)

assert np.allclose(m_test.bvec, m_standard.bvec)

