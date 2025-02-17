#!/usr/bin/env python

"""
This example shows how to take coils in simsopt, write an mgrid file, and then
run free-boundary vmec. No optimization of the plasma or coil shapes is
performed in this example.

You can run this example with one or multiple MPI processes.
"""

from pathlib import Path
import numpy as np
from simsopt.configs import get_w7x_data
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition

nfp = 5

# Load in some coils
curves, currents, magnetic_axis = get_w7x_data()
coils = coils_via_symmetries(curves, currents, nfp, True)
bs = BiotSavart(coils)

# Number of grid points in the toroidal angle:
nphi = 24

mgrid_file = "mgrid.w7x.nc"
bs.to_mgrid(
    mgrid_file,
    nr=64,
    nz=65,
    nphi=nphi,
    rmin=4.5,
    rmax=6.3,
    zmin=-1.0,
    zmax=1.0,
    nfp=nfp,
)

# Create a VMEC object from an input file:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
input_file = str(TEST_DIR / "input.W7-X_standard_configuration")

mpi = MpiPartition(1)
vmec = Vmec(input_file, mpi=mpi)

# That input file was for fixed-boundary. We need to change some of
# the vmec input parameters for a free-boundary calculation:
vmec.indata.lfreeb = True
vmec.indata.mgrid_file = mgrid_file
vmec.indata.nzeta = nphi
# All the coils are written into a single "current group", so we only need to
# set a single entry in vmec's "extcur" array:
vmec.indata.extcur[0] = 1.0

# Lower the resolution, so the example runs faster:
vmec.indata.mpol = 6
vmec.indata.ntor = 6
vmec.indata.ns_array[2] = 0
ftol = 1e-10
vmec.indata.ftol_array[1] = ftol

vmec.run()

assert vmec.wout.fsql < ftol
assert vmec.wout.fsqr < ftol
assert vmec.wout.fsqz < ftol
assert vmec.wout.ier_flag == 0
np.testing.assert_allclose(vmec.wout.volume_p, 28.6017247168422, rtol=0.01)
np.testing.assert_allclose(vmec.wout.rmnc[0, 0], 5.561878306096512, rtol=0.01)
np.testing.assert_allclose(vmec.wout.bmnc[0, 1], 2.78074392223658, rtol=0.01)
