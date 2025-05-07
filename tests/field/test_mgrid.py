#!/usr/bin/env python

import unittest
from pathlib import Path
from monty.tempfile import ScratchDir
import numpy as np
from scipy.io import netcdf_file

try:
    import vmec
except ImportError:
    vmec = None

from simsopt.configs import get_w7x_data
from simsopt.field import BiotSavart, coils_via_symmetries, MGrid
from simsopt.mhd import Vmec

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
test_file = TEST_DIR / 'mgrid.pnas-qa-test-lowres-standard.nc'
test_file2 = TEST_DIR / 'mgrid_ncsx_lowres_test.nc'


class Testing(unittest.TestCase):

    def test_from_file(self):
        mgrid = MGrid.from_file(test_file)

        assert mgrid.rmin == 0.5
        assert mgrid.nphi == 6
        assert mgrid.bvec.shape == (11, 11, 6, 3)

        names = mgrid.coil_names
        assert len(names) == 1
        assert names[0] == '________simsopt_coils_________'

        assert mgrid.br_arr.shape == (1, 6, 11, 11)
        assert mgrid.br[0, 0, 0] == -1.0633399551863771  # -0.9946816978184079

        mgrid = MGrid.from_file(test_file2)
        assert mgrid.rmin == 1.0
        assert mgrid.bvec.shape == (10, 12, 4, 3)
        assert mgrid.bz[1, 1, 1] == -1.012339153040808
        assert mgrid.ap[1, 1, 1] == -0.3719177477496187

    def test_add_field_cylinder(self):
        N_points = 5
        br = np.ones(N_points)
        bp = br*2
        bz = br*3
        name = "test_coil"

        mgrid = MGrid()
        mgrid.add_field_cylindrical(br, bp, bz, name=name)
        assert mgrid.n_ext_cur == 1
        assert mgrid.coil_names[0] == '__________test_coil___________'
        assert np.allclose(mgrid.br_arr[0], br)

    def test_write(self):
        mgrid = MGrid.from_file(test_file)
        with ScratchDir("."):
            filename = 'mgrid.test.nc'
            mgrid.write(filename)

            with netcdf_file(filename, mmap=False) as f:
                np.testing.assert_allclose(f.variables['zmin'][()], -0.5)
                assert f.variables['nextcur'][()] == 1
                assert f.variables['br_001'][:].shape == (6, 11, 11)
                assert f.variables['mgrid_mode'][:][0].decode('ascii') == 'N'

                byte_string = f.variables['coil_group'][:]
                message = "".join([x.decode('ascii') for x in byte_string])
                assert message == '________simsopt_coils_________'

    def test_plot(self):
        mgrid = MGrid.from_file(test_file)
        mgrid.plot(show=False)


@unittest.skipIf(vmec is None, "Interface to VMEC not found")
class VmecTests(unittest.TestCase):
    def test_free_boundary_vmec(self):
        """
        Check that the files written by this MGrid class can be read by
        free-boundary vmec.
        """
        input_file = str(TEST_DIR / "input.W7-X_standard_configuration")

        curves, currents, magnetic_axis = get_w7x_data()
        nfp = 5
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        eq = Vmec(input_file)
        nphi = 24
        with ScratchDir("."):
            filename = "mgrid.bfield.nc"
            bs.to_mgrid(
                filename,
                nphi=nphi,
                rmin=4.5,
                rmax=6.3,
                zmin=-1.0,
                zmax=1.0,
                nr=64,
                nz=65,
                nfp=5,
            )

            eq.indata.lfreeb = True
            eq.indata.mgrid_file = filename
            eq.indata.extcur[0] = 1.0
            eq.indata.nzeta = nphi
            eq.indata.mpol = 6
            eq.indata.ntor = 6
            eq.indata.ns_array[2] = 0
            ftol = 1e-10
            eq.indata.ftol_array[1] = ftol
            eq.run()

            np.testing.assert_allclose(eq.wout.volume_p, 28.6017247168422, rtol=0.01)
            np.testing.assert_allclose(eq.wout.rmnc[0, 0], 5.561878306096512, rtol=0.01)
            np.testing.assert_allclose(eq.wout.bmnc[0, 1], 2.78074392223658, rtol=0.01)
            assert eq.wout.fsql < ftol
            assert eq.wout.fsqr < ftol
            assert eq.wout.fsqz < ftol
            assert eq.wout.ier_flag == 0
